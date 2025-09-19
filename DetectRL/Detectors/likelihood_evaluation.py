import logging
import random
import numpy as np
import torch
import tqdm
import argparse
import json
from metrics import get_roc_metrics
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def get_ll(text, args, tokenizer, model):
    with torch.no_grad():
        tokenized = tokenizer(text, return_tensors="pt").to(args.DEVICE)
        labels = tokenized['input_ids']
        if labels.nelement() == 0:
            logging.error(f"Empty input: {text}")
            return 0
        else:
            return -model(**tokenized, labels=labels).loss.item()

def truncate_text_to_sentences(text, min_word_count=100):
    word_count = 0
    end_of_last_sentence = 0

    words = text.split()

    for i, word in enumerate(words):
        if word[-1] in '.!?':
            if word_count >= min_word_count:
                return ' '.join(words[:i + 1])
            end_of_last_sentence = i
        word_count += 1

    if end_of_last_sentence > 0:
        return ' '.join(words[:end_of_last_sentence + 1])
    else:
        return ' '.join(words)


def experiment(args):
    # load model
    logging.info(f"Loading base model of type {args.base_model}...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    base_model.eval()
    base_model.cuda()

    filenames = args.test_data_path.split(",")
    for filename in filenames:
        logging.info(f"Test in {filename}")
        test_data = json.load(open(filename, "r"))

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        predictions = {'human': [], 'llm': []}
        for item in tqdm.tqdm(test_data):
            text = item["text"]
            label = item["label"]

            item["text_ll"] = get_ll(text, args, base_tokenizer, base_model)

            # result
            if label == "human":
                predictions['human'].append(item["text_ll"])
            elif label == "llm":
                predictions['llm'].append(item["text_ll"])
            else:
                raise ValueError(f"Unknown label {label}")

        predictions['human'] = [i for i in predictions['human'] if np.isfinite(i)]
        predictions['llm'] = [i for i in predictions['llm'] if np.isfinite(i)]

        roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy = get_roc_metrics(predictions['human'],
                                                                                                   predictions['llm'])

        result = {
            "roc_auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "conf_matrix": conf_matrix,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }

        logging.info(f"{result}")
        with open(filename.split(".json")[0] + "_likelihood_data.json", "w") as f:
            json.dump(test_data, f, indent=4)

        with open(filename.split(".json")[0] + "_likelihood_result.json", "w") as f:
            json.dump(result, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True,
                        help="Path to the test data. could be several files with ','. "
                             "Note that the data should have been perturbed.")
    parser.add_argument('--base_model', default="EleutherAI/gpt-neo-2.7B", type=str, required=False)
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    parser.add_argument('--seed', default=2023, type=int, required=False)
    args = parser.parse_args()

    experiment(args)
