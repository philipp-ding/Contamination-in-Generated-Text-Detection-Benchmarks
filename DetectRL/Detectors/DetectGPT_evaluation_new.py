import functools
import logging
import math
import random
import numpy as np
import torch
import tqdm
import argparse
import json
from DetectGPT import perturb_texts
from loss import get_ll, get_lls
from metrics import get_roc_metrics,get_roc_by_threshold
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def experiment(args):
    ntrain=args.ntrain
    # load model
    logging.info(f"Loading base model of type {args.base_model}...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)

    mask_tokenizer = AutoTokenizer.from_pretrained(args.mask_model)
    mask_model = AutoModelForSeq2SeqLM.from_pretrained(args.mask_model)

    model_config = {
        "base_tokenizer": base_tokenizer,
        "base_model": base_model,
        "mask_tokenizer": mask_tokenizer,
        "mask_model": mask_model,
    }

    # filenames = args.test_data_path.split(",")
    # for filename in filenames:
    # train
    logging.info(f"Train in {args.train_data_path}")
    test_data = json.load(open(args.train_data_path, "r"))[:ntrain*2]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    perturb_fn = functools.partial(perturb_texts, args=args, model_config=model_config)

    mask_model.eval()
    mask_model.cuda()
    for item in tqdm.tqdm(test_data):
        perturbed_text = perturb_fn([item["text"] for _ in range(max(args.n_perturbation_list))])
        assert len(perturbed_text) == max(
            args.n_perturbation_list), f"Expected {max(args.n_perturbation_list)} perturbed samples, got {len(perturbed_text)}"
        item["perturbed_text"] = perturbed_text
    mask_model.to("cpu")

    base_model.eval()
    base_model.cuda()
    for item in tqdm.tqdm(test_data):
        text = item["text"]
        item["text_ll"] = get_ll(text, args, base_tokenizer, base_model)
        perturbed_text_ll = get_lls(item["perturbed_text"], args, base_tokenizer, base_model)
        item["perturbed_text_ll"] = perturbed_text_ll

        for n_perturbation in args.n_perturbation_list:
            item[f"perturbed_text_ll_{n_perturbation}"] = np.mean(
                [i for i in perturbed_text_ll[:n_perturbation] if i != 0])
            item[f"perturbed_text_ll_std_{n_perturbation}"] = np.std(
                [i for i in perturbed_text_ll[:n_perturbation] if i != 0]) if len(
                [i for i in perturbed_text_ll[:n_perturbation] if i != 0]) > 1 else 1
    base_model.to("cpu")

    results = {}
    for n_perturbation in args.n_perturbation_list:
        predictions = {'human': [], 'llm': []}
        for item in test_data:
            label = item["label"]
            detectgpt_score = (item['text_ll'] - item[f'perturbed_text_ll_{n_perturbation}']) / item[
                f'perturbed_text_ll_std_{n_perturbation}']

            item['detectgpt_score'] = detectgpt_score

            if label == "human":
                predictions['human'].append(detectgpt_score)
            elif label == "llm":
                predictions['llm'].append(detectgpt_score)
            else:
                raise ValueError(f"Unknown label {label}")

        predictions['human'] = [i for i in predictions['human'] if not math.isnan(i)]
        predictions['llm'] = [i for i in predictions['llm'] if not math.isnan(i)]

        predictions['human'] = [i for i in predictions['human'] if np.isfinite(i)]
        predictions['llm'] = [i for i in predictions['llm'] if np.isfinite(i)]

        roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy, tpr_at_fpr_0_01= get_roc_metrics(
            predictions['human'], predictions['llm'])

        result = {
            "roc_auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "conf_matrix": conf_matrix,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tpr_at_fpr_0_01": tpr_at_fpr_0_01
        }

        results[f"{n_perturbation}_perturbation"] = result

    print(f"{results}")

    # test
    logging.info(f"Test in {args.test_data_path}")
    test_data = json.load(open(args.test_data_path, "r"))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    perturb_fn = functools.partial(perturb_texts, args=args, model_config=model_config)

    mask_model.eval()
    mask_model.cuda()
    for item in tqdm.tqdm(test_data):
        perturbed_text = perturb_fn([item["text"] for _ in range(max(args.n_perturbation_list))])
        assert len(perturbed_text) == max(
            args.n_perturbation_list), f"Expected {max(args.n_perturbation_list)} perturbed samples, got {len(perturbed_text)}"
        item["perturbed_text"] = perturbed_text
    mask_model.to("cpu")

    base_model.eval()
    base_model.cuda()
    for item in tqdm.tqdm(test_data):
        text = item["text"]
        item["text_ll"] = get_ll(text, args, base_tokenizer, base_model)
        perturbed_text_ll = get_lls(item["perturbed_text"], args, base_tokenizer, base_model)
        item["perturbed_text_ll"] = perturbed_text_ll

        for n_perturbation in args.n_perturbation_list:
            item[f"perturbed_text_ll_{n_perturbation}"] = np.mean(
                [i for i in perturbed_text_ll[:n_perturbation] if i != 0])
            item[f"perturbed_text_ll_std_{n_perturbation}"] = np.std(
                [i for i in perturbed_text_ll[:n_perturbation] if i != 0]) if len(
                [i for i in perturbed_text_ll[:n_perturbation] if i != 0]) > 1 else 1
    base_model.to("cpu")

    results = {}
    for n_perturbation in args.n_perturbation_list:
        predictions = {'human': [], 'llm': []}
        for item in test_data:
            label = item["label"]
            detectgpt_score = (item['text_ll'] - item[f'perturbed_text_ll_{n_perturbation}']) / item[
                f'perturbed_text_ll_std_{n_perturbation}']

            item['detectgpt_score'] = detectgpt_score

            if label == "human":
                predictions['human'].append(detectgpt_score)
            elif label == "llm":
                predictions['llm'].append(detectgpt_score)
            else:
                raise ValueError(f"Unknown label {label}")

        predictions['human'] = [i for i in predictions['human'] if not math.isnan(i)]
        predictions['llm'] = [i for i in predictions['llm'] if not math.isnan(i)]

        predictions['human'] = [i for i in predictions['human'] if np.isfinite(i)]
        predictions['llm'] = [i for i in predictions['llm'] if np.isfinite(i)]

        roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy, tpr_at_fpr_0_01 = get_roc_by_threshold(
            predictions['human'], predictions['llm'],threshold=optimal_threshold)

        result = {
            "roc_auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "conf_matrix": conf_matrix,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tpr_at_fpr_0_01": tpr_at_fpr_0_01
        }

        results[f"{n_perturbation}_perturbation"] = result

    print(f"{results}")
    with open(args.test_data_path.split(".json")[0] + f"_{ntrain}_DetectGPT_data.json", "w") as f:
        json.dump(test_data, f, indent=4)

    with open(args.test_data_path.split(".json")[0] + f"_{ntrain}_DetectGPT_result.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, required=True,
                        help="Path to the test data. could be several files with ','. "
                             "Note that the data should have been perturbed.")
    parser.add_argument('--test_data_path', type=str, required=True,



















                        help="Path to the test data. could be several files with ','. "
                             "Note that the data should have been perturbed.")
    parser.add_argument('--base_model', default="EleutherAI/gpt-neo-2.7B", type=str)
    parser.add_argument('--mask_model', default="t5-small", type=str)
    parser.add_argument('--ntrain', default=512, type=int, required=True)
    parser.add_argument('--output_file', default="output_file", type=str)
    parser.add_argument('--n_perturbation_list', default=[1, 10, 20, 50, 100], type=list)
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--pct_words_masked', type=float,
                        default=0.3)  # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument('--DEVICE', default="cuda", type=str, required=False)
    parser.add_argument('--seed', default=2023, type=int, required=False)
    args = parser.parse_args()

    experiment(args)
