import argparse
import logging
import os
import random
import tqdm
import json
import numpy as np
import pandas as pd
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile
from types import SimpleNamespace
import torch
import transformers
from torch.utils.data import Dataset
from transformers import TrainerCallback
from metrics import get_roc_metric_result
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# from transformers.utils import logging as hf_logging
# from datasets import disable_caching
#
# disable_caching()

custom_cache_dir = "/mnt/hdd-baracuda/pdingfelder/tmp"
os.environ["HF_HOME"] = custom_cache_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(custom_cache_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(custom_cache_dir, "datasets")
os.environ["HF_METRICS_CACHE"] = os.path.join(custom_cache_dir, "metrics")

# Disable console output from Transformers
# hf_logging.disable_default_handler()
#
# # Set up standard logging
# logger = logging.getLogger()  # root logger
# logger.setLevel(logging.WARNING)  # or DEBUG
#
# file_handler = logging.FileHandler("transformers.log", mode="w", encoding="utf-8")
# formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
# file_handler.setFormatter(formatter)
# file_handler.setLevel(logging.INFO)
#
# logger.addHandler(file_handler)
# logging.basicConfig(level=logging.ERROR, format="%(asctime)s %(levelname)s %(message)s")
# hf_logging.set_verbosity_error()


def hash_dataframe_as_parquet(df: pd.DataFrame) -> str:
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Save DataFrame as Parquet
        df.to_parquet(tmp_path, engine="pyarrow", index=True)

        # Read the file content as bytes
        with open(tmp_path, "rb") as f:
            file_bytes = f.read()

        # Calculate SHA-256 hash
        return hashlib.sha256(file_bytes).hexdigest()
    finally:
        # Clean up the temporary file
        os.remove(tmp_path)


def eval_experiment(args, model_path, test_data_path: str = None, test_df: pd.DataFrame | list[pd.DataFrame] = None,
                    optimal_threshold=None) -> tuple[float, dict]:
    # logging.info(f"Loading base model of type {args.model_name}...")
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(model_path).to(args.DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    results_combined = {}

    if test_data_path is not None and len(test_data_path) > 0:
        filenames: list[str] = test_data_path.split(",")
    elif test_df is not None:
        test_df: list[pd.DataFrame] = [test_df] if isinstance(test_df, pd.DataFrame) else test_df
        filenames: list[str] = [f"/{hash_dataframe_as_parquet(_df)}_train" for _df in test_df]
    else:
        raise ValueError("Either test_df or test_data_path has to be defined.")

    for counter, filename in enumerate(filenames):
        # print(f"Test in {filename}")
        if test_data_path is not None and len(test_data_path) > 0:
            test_data = json.load(open(filename, "r"))
        else:
            test_data = test_df[counter].to_dict(orient="records")

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        predictions = {'human': [], 'llm': []}
        with torch.no_grad():
            for item in test_data:
                try:
                    text = item["text"]
                    label = item["label"]
                    tokenized = tokenizer(text, padding=True, truncation=True, max_length=512,
                                          return_tensors="pt").to(args.DEVICE)
                    _prediction = detector(**tokenized).logits.softmax(-1)[:, 0].tolist()[0]
                    item["prediction"] = _prediction
                    if label == "human":
                        predictions["human"].append(_prediction)
                    elif label == "llm":
                        predictions["llm"].append(_prediction)
                    else:
                        raise ValueError(f"Unknown label {label}")
                except Exception as e:
                    print(e)
                    print(f"Text: {text}, label: {label}, item: {item}")

        predictions['human'] = [-i for i in predictions['human'] if np.isfinite(i)]
        predictions['llm'] = [-i for i in predictions['llm'] if np.isfinite(i)]

        result = get_roc_metric_result(
            predictions['human'], predictions['llm'], optimal_threshold
        )

        if "xlm-roberta-base" in args.model_name:
            model_name = "xlm-roberta-base"
        elif "xlm-roberta-large" in args.model_name:
            model_name = "xlm-roberta-large"
        else:
            model_name = args.model_name

        parts = filename.split('/')
        filename = parts[-1]  # 获取文件名 'cross_domains_arxiv_train.json'
        file_base = filename.split('_train')[0]  # 从文件名中分割出基础部分 'cross_domains_arxiv'

        # print(f"{result}")
        with open(f"{model_path}/{file_base}.{model_name}_data.json", "w") as f:
            json.dump(test_data, f, indent=4)

        with open(f"{model_path}/{file_base}.{model_name}_result.json", "w") as f:
            json.dump(result, f, indent=4)
        results_combined[filename] = result

    return optimal_threshold, results_combined


class JSONDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        text = row["text"]
        label = 0 if row["label"] == "human" else 1
        inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=512)
        inputs["labels"] = label
        return inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds,
                                                               average='micro')
    acc = accuracy_score(labels, preds)

    # if (type(accuracy_score) is not float) and (type(accuracy_score) is not int):
    #     acc = acc.item()

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=10, metric_key="eval_loss"):
        self.patience = patience
        self.metric_key = metric_key
        self.best_metric = float("inf")
        self.wait = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_metric = metrics[self.metric_key]
        if current_metric <= self.best_metric:
            self.best_metric = current_metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                control.should_training_stop = True


class EvalAccuracyCallback(TrainerCallback):

    def __init__(self, model_path: str = ""):
        super().__init__()
        self.model_path = model_path

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        epoch = int(state.epoch)
        eval_accuracy = metrics["eval_accuracy"]
        eval_f1 = metrics["eval_f1"]
        eval_precision = metrics["eval_precision"]
        eval_recall = metrics["eval_recall"]
        #
        # print(
        #     f"Epoch: {epoch} - Accuracy: {eval_accuracy:.4f}, F1: {eval_f1:.4f}, Precision: {eval_precision:.4f}, "
        #     f"Recall: {eval_recall:.4f}")

        with open(f"{self.model_path}/eval_result.txt", "a") as f:
            f.write(
                f"Epoch: {epoch} - Accuracy: {eval_accuracy:.4f}, F1: {eval_f1:.4f}, "
                f"Precision: {eval_precision:.4f}, Recall: {eval_recall:.4f}\n")


def train_test_data_split(data: list[dict], seed: int):
    human_data, llm_data = [], []
    for sample in data:
        if sample["label"] == "human":
            human_data.append(sample)

        if sample["label"] == "llm":
            llm_data.append(sample)

    # train_data = human_data[:-200] + llm_data[:-200]
    # valid_data = human_data[-200:] + llm_data[-200:]
    data_available = min(len(human_data), len(llm_data))
    # print(data_available)
    valid_data_len = int(0.2 * data_available)
    train_data = human_data[:-valid_data_len] + llm_data[:-valid_data_len]
    valid_data = human_data[-valid_data_len:] + llm_data[-valid_data_len:]

    random.seed(seed)
    random.shuffle(train_data)
    random.shuffle(valid_data)

    # print(f"Training data size: {len(train_data)} samples")
    # print(f"Validation data size: {len(valid_data)} samples")

    return train_data, valid_data


def generate_args_for_training_roberta(train_data_path: str = None, test_data_path: str = None,
                                       transfer_data_path: list[str] = None, train_df: pd.DataFrame = None,
                                       test_df: pd.DataFrame = None, transfer_df: list[pd.DataFrame] = None,
                                       epochs: int = 3, learning_rate: float = 1e-6,
                                       batch_size: int = 8, seed: int = 2023, mode: str = "train", device: str = "cuda",
                                       model_name: str = "roberta-base",
                                       save_model_path: str = "Detector_Results/roberta_base_classifier"):
    if transfer_df is None:
        transfer_df = []
    if transfer_data_path is None:
        transfer_data_path = []
    args_dict = {
        "model_name": model_name,
        "save_model_path": save_model_path,
        "train_data_path": train_data_path,
        "test_data_path": test_data_path,
        "transfer_test_data_path": ",".join(transfer_data_path),
        "train_df": train_df,
        "test_df": test_df,
        "transfer_df": transfer_df,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seed": seed,
        "mode": mode,
        "DEVICE": device
    }

    # Convert dict to namespace-like object
    created_args = SimpleNamespace(**args_dict)
    return created_args


def train_and_evaluate_roberta(args, train_data: list[dict], valid_data: list[dict], model_path: str):
    # Initialize tokenizer and model
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name, cache_dir=custom_cache_dir)
    # Create data loaders
    train_dataset = JSONDataset(train_data, tokenizer)  # [:2000]
    valid_dataset = JSONDataset(valid_data, tokenizer)

    result_path = f"{model_path}_results"
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=result_path,  # output directory
        num_train_epochs=args.epochs,  # total number of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        logging_dir='./logs',  # directory for storing logs
        learning_rate=args.learning_rate,
        save_strategy="epoch",
        seed=args.seed,
        save_total_limit=1,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
    )

    def model_init():
        return RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=2,
                                                                cache_dir=custom_cache_dir)

    # Initialize Trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EvalAccuracyCallback(model_path=model_path), EarlyStoppingCallback()]
    )

    # Train model
    trainer.train()

    # Evaluate model
    eval_result = trainer.evaluate()

    # Print out the results
    # for key in sorted(eval_result.keys()):
    #     print(f"{key}: {eval_result[key]}")

    with open(f"{model_path}/eval_result.txt", "a") as f:
        f.write(str(eval_result))

    trainer.save_model(model_path)

    # Save tokenizer
    tokenizer.save_pretrained(model_path)

    # Save model config
    trainer.model.config.save_pretrained(model_path)

    # Save eval result
    with open(f"{model_path}/eval_result.json", "w") as f:
        json.dump(eval_result, f)

    # evaluate on test data
    optimal_threshold, _ = eval_experiment(args, model_path=model_path, test_data_path=args.test_data_path,
                                           test_df=args.test_df)
    _, results_combined = eval_experiment(args, model_path=model_path, test_data_path=args.transfer_test_data_path,
                                          test_df=args.transfer_df, optimal_threshold=optimal_threshold)
    return results_combined


def _get_data(data_path: str, df: pd.DataFrame, mode: str = "training") -> list:
    if data_path is not None:
        with open(data_path, "r") as f:
            data = json.load(f)
        f.close()
        return data
    elif df is not None:
        return df.to_dict(orient="records")
    else:
        raise ValueError(f"Either a dataframe or an input data file has to be provided for the {mode} dataset!")


def get_model_path_and_create_result_file(save_model_path: str, train_data_path: str):
    if save_model_path.startswith("/mnt/") or save_model_path.startswith("C:"):
        model_path = save_model_path
    elif train_data_path is not None:
        model_path = f"{train_data_path.split('train')[0]}{save_model_path}"
    else:
        model_path = f"./{save_model_path}"
    os.makedirs(model_path, exist_ok=True)
    with open(f"{model_path}/eval_result.txt", "w") as f:
        pass
    return model_path


def run(args) -> dict:
    if args.mode == "train":
        model_path = get_model_path_and_create_result_file(args.save_model_path, args.train_data_path)
        data = _get_data(args.train_data_path, args.train_df)

        train_data, valid_data = train_test_data_split(data, seed=args.seed)
        results_combined = train_and_evaluate_roberta(args, train_data, valid_data, model_path)

    elif args.mode == "eval":
        model_path = f"{args.train_data_path.split('train')[0]}{args.save_model_path}"
        optimal_threshold, _ = eval_experiment(args, model_path=model_path, test_data_path=args.test_data_path,
                                               test_df=args.test_df)
        _, results_combined = eval_experiment(args, model_path=model_path, test_data_path=args.test_data_path,
                                              test_df=args.test_df, optimal_threshold=optimal_threshold)

    else:
        raise ValueError(f"Mode has to be either 'train' or 'eval'. Mode provided: '{args.mode}' ")

    return results_combined


# def run(save_model_path: str, train_data: pd.DataFrame, mode: str = "train", seed: int = 2023):
#     arguments = generate_args_for_training_roberta()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="roberta-base", type=str)
    parser.add_argument('--save_model_path', default="roberta_base_classifier", type=str)
    parser.add_argument('--train_data_path', default="", type=str, required=True)
    parser.add_argument('--test_data_path', type=str, required=True,
                        help="Path to the test data. could be several files with ','. "
                             "Note that the data should have been perturbed.")
    parser.add_argument('--transfer_test_data_path', type=str, required=True,
                        help="Path to the test data. could be several files with ','. "
                             "Note that the data should have been perturbed.")
    parser.add_argument('--train_df', type=pd.DataFrame, default=None,
                        help="Train dataframe. Can be provided instead of train_data_path")
    parser.add_argument('--test_df', type=pd.DataFrame, default=None,
                        help="Test dataframe. Can be provided instead of test_data_path")
    parser.add_argument('--transfer_df', type=pd.DataFrame, default=None,
                        help="Transfer test dataframe. Can be provided instead of transfer_test_data_path")
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--learning_rate', default=1e-6, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    args = parser.parse_args()
    # run(args)
