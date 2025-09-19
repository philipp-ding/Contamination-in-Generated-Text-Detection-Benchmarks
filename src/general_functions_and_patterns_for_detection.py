from typing import Union
import random
import hashlib
import numpy as np
import pandas as pd
import os
import sys
import json
import re
from tqdm import tqdm
import shap
import torch
import transformers

# === CONFIG GLOBAL CONSTANTS ===
BASE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")

# TODO Change detector results dir
DETECT_RL_DIR = f"{BASE_DIR}DetectRL/"
DETECTOR_RESULTS = f"{DETECT_RL_DIR}Detector_Results/"
DETECTOR_RESULTS_NO_SYNC = f"{BASE_DIR}../Training_Results/"

sys.path.append(os.path.join(DETECT_RL_DIR, "Detectors"))
from DetectRL.Detectors import train_roberta

DATA_DIR = f"{DETECT_RL_DIR}Benchmark"
ORIGINAL_DATA_DIR = f"{DATA_DIR}/Original_Dataset/"
BENCHMARK_DIR = f"{DATA_DIR}/Benchmark_Data/"
TASK_DIR = f"{DATA_DIR}/Tasks/"

RESULT_DIR = f"{BASE_DIR}results/"
CLEANED_FILES_DIR = f"{BASE_DIR}/cleaned_DetectRL_files/"
RECLEANED_FILES_DIR = f"{CLEANED_FILES_DIR}/recleaned_files_using_llms/"
REGEX_CLEANED_FILES = f"{CLEANED_FILES_DIR}/regex_cleaned_files/"
# if debug is true, only a subset of the data is processed to test the functionality/ setup
#   (e.g. only the first three folders are considered)
DEBUG = False
SEED = 2023
COLUMNS_DIRECTLY_LLM_GENERATED_DETECT_RL = ("direct_prompt", "prompt_few_shot", "prompt_SICO", "paraphrase_polish_llm",
                                            "paraphrase_polish_human")
DOMAINS = ["arxiv", "xsum", "writing_prompt", "yelp_review"]
LLMs = ["ChatGPT", "Claude-instant", "Llama-2-70b", "Google-PaLM"]
LABELS = ["human", "llm"]
json_path_writing = f"{BASE_DIR}/DetectRL/Benchmark/Original_Dataset/writing_prompt_2800.json"
json_path_abstract = f"{BASE_DIR}/DetectRL/Benchmark/Original_Dataset/arxiv_2800.json"
json_path_review = f"{BASE_DIR}/DetectRL/Benchmark/Original_Dataset/yelp_review_2800.json"
json_path_xsum = f"{BASE_DIR}/DetectRL/Benchmark/Original_Dataset/xsum_2800.json"

# === Different patterns defined as regular expressions ===
# default ending possibilities of an LLM introduction sentence
# PATTERN_ENDING_LLM_SENTENCE = r"[\s\w,\"\']{0,300}([:!.?]+|[:]?[\*]{2})"
PATTERN_ENDING_LLM_SENTENCE = r"[^.:!?]{0,300}([:!.?]+|[:]?[\*]{2})"
PATTERN_BEGINNING_LLM_SENTENCE = (
    r"^((\[SYSTEM\]|\*{0,2}assistant\*{0,2})[: ]?)?"
    r"^((Of course|Sure)[.!,]?)?"
    r"^(?:\w{1,10}![ ]?)?"  # Optional short exclamation at the start
    r"[^.:!?]{0,100}"  # Up to 100 characters excluding terminal punctuation
)

# === general patterns ===
PATTERN_REJECTION = (r"(.*I apologize, upon further reflection.*?|.*a fake review.*|.*((only)|(just)) a language model"
                     r".*|.*I cannot provide.*|.*As an AI language model, I am unable to engage with content that may "
                     r"violate my usage guidelines.*|.*upon reflection I do not.*|.*As an AI.*|.*(I apologize, "
                     r"(but\w?)?(as an AI|upon reflection)).*)")
PATTERN_CLEANUP = (fr"{PATTERN_BEGINNING_LLM_SENTENCE}(Voici un|Here is|Here are|Here's|Sure[,!]?\s?here)"
                   fr"{PATTERN_ENDING_LLM_SENTENCE}")
REMOVE_FIRST_OCCURRENCE = (fr"{PATTERN_BEGINNING_LLM_SENTENCE}(\d+ sentences|sentence|\[assistant\]|summary)"
                           fr"{PATTERN_ENDING_LLM_SENTENCE}")
# REMOVE_ONLY_PATTERN = r"(?:.*)(\[assistant\])\s*\w{0,20}([:]?[\*]{2}|[:])"
REMOVE_ONLY_PATTERN = r"(?:.*)(((\[system\])|(\[user\])|(\[assistant\]))\s*\w{0,20}|(\*\*assistant))([:]?[\*]{2}|[:])"
# typical patterns based on articles
PATTERN_ARTICLE = (fr"{PATTERN_BEGINNING_LLM_SENTENCE}(given article title|provided article title)"
                   fr"{PATTERN_ENDING_LLM_SENTENCE}")
PATTERN_REVIEW = fr"{PATTERN_BEGINNING_LLM_SENTENCE}(review's first sentence|review){PATTERN_ENDING_LLM_SENTENCE}"
PATTERN_ABSTRACT = fr"{PATTERN_BEGINNING_LLM_SENTENCE}(abstract|academic article){PATTERN_ENDING_LLM_SENTENCE}"
PATTERN_XSUM = fr"{PATTERN_BEGINNING_LLM_SENTENCE}(article){PATTERN_ENDING_LLM_SENTENCE}"

# typical patterns based on tasks
PATTERN_SICO = PATTERN_BEGINNING_LLM_SENTENCE + r"(in a human\s?\w{0,20}\s?style)" + PATTERN_ENDING_LLM_SENTENCE
PATTERN_POLISHING = PATTERN_BEGINNING_LLM_SENTENCE + (r"(grammar[\w\s,]{1,40}spelling)|(spelling[\w\s,]{1,40}grammar)|"
                                                      r"(Improved sentence structure)") + PATTERN_ENDING_LLM_SENTENCE

PATTERN_COMBINED = (PATTERN_CLEANUP + "|" + REMOVE_FIRST_OCCURRENCE + "|" + PATTERN_REJECTION + "|" + PATTERN_ARTICLE +
                    "|" + PATTERN_REVIEW + "|" + PATTERN_ABSTRACT + "|" + PATTERN_XSUM + "|" + PATTERN_SICO + "|" +
                    PATTERN_POLISHING)


# === end patterns ===

def seed_everything(seed: int = SEED) -> None:
    """Set random seeds for reproducibility across libraries.

    Args:
        seed (int): The seed value to use.

    Returns:
        None

    Example:
        >>> seed_everything(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # HF convenience function
    transformers.enable_full_determinism(seed=SEED)
    transformers.set_seed(seed, deterministic=True)


def file_hash(filepath: str) -> str:
    """Compute the SHA256 hash of a file.

    Args:
        filepath (str): Path to the file.

    Returns:
        str: SHA256 hash string.

    Raises:
        FileNotFoundError: If the file does not exist.

    Example:
        >>> file_hash("data.json")
        'e3b0c44298fc1c149afbf4c8996fb924...'
    """
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_files(filename: str) -> pd.DataFrame:
    """Read a JSONL file and return its contents as a DataFrame.

    Args:
        filename (str): Path to the JSONL file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.

    Example:
        >>> read_files("data.jsonl")
        pd.DataFrame(...)
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)


def get_info_based_on_counter(counter: int) -> tuple:
    """Map a counter value to domain information.

    Args:
        counter (int): Counter value (0-3).

    Returns:
        tuple: (domain, prompt_key, human_key)

    Raises:
        ValueError: If counter is not in range 0-3.

    Example:
        >>> get_info_based_on_counter(0)
        ('arxiv', 'title', 'abstract')
    """
    if counter == 0:
        return get_info_based_on_input_path("arxiv")
    elif counter == 1:
        return get_info_based_on_input_path("xsum")
    elif counter == 2:
        return get_info_based_on_input_path("writing_prompt")
    elif counter == 3:
        return get_info_based_on_input_path("yelp_review")
    else:
        raise ValueError(f"Cannot find mapping for the counter {counter} (defined for 0-3)")


def get_info_based_on_input_path(input_path: str) -> tuple:
    """Extract domain, prompt key, and human key from an input path.

    Args:
        input_path (str): Path or string containing domain info.

    Returns:
        tuple: (domain, prompt_key, human_key)

    Raises:
        ValueError: If input_path does not match any known domain.

    Example:
        >>> get_info_based_on_input_path("arxiv_2800.json")
        ('arxiv', 'title', 'abstract')
    """
    if "arxiv" in input_path:
        domain = "arxiv"
        prompt_key = "title"
        human_key = "abstract"
    elif "xsum" in input_path:
        domain = "xsum"
        prompt_key = "summary"
        human_key = "document"
    elif "writing_prompt" in input_path:
        domain = "writing_prompt"
        prompt_key = "story_prompt"
        human_key = "story"
    elif "yelp_review" in input_path:
        domain = "yelp_review"
        prompt_key = "start"
        human_key = "content"
    else:
        raise ValueError(f"Cannot find mapping for the input path: {input_path}")
    return domain, prompt_key, human_key


def filter_for_one_llm(_df: pd.DataFrame, column_llm: str = "llm_type", llm_to_filter: str = "Claude-instant") -> pd.DataFrame:
    """Filter DataFrame for rows matching a specific LLM type.

    Args:
        _df (pd.DataFrame): Input DataFrame.
        column_llm (str): Column name for LLM type.
        llm_to_filter (str): LLM type to filter for.

    Returns:
        pd.DataFrame: Filtered DataFrame.

    Example:
        >>> filter_for_one_llm(df, "llm_type", "Claude-instant")
        pd.DataFrame(...)
    """
    return _df[_df[column_llm] == llm_to_filter]


def load_dataframe_from_json(_json_path: str, filter_llm: bool = False, column_llm: str = "llm_type",
                             llm_to_filter: str = "Claude-instant") -> pd.DataFrame:
    """Load a DataFrame from a JSON file, optionally filtering by LLM type.

    Args:
        _json_path (str): Path to the JSON file.
        filter_llm (bool): Whether to filter by LLM type.
        column_llm (str): Column name for LLM type.
        llm_to_filter (str): LLM type to filter for.

    Returns:
        pd.DataFrame: Loaded (and optionally filtered) DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.

    Example:
        >>> load_dataframe_from_json("data.json", filter_llm=True)
        pd.DataFrame(...)
    """
    # Load JSON data
    with open(_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    if filter_llm:
        df = filter_for_one_llm(df, column_llm, llm_to_filter)
    return df


# === Load JSON data from folders ===
def load_json_file_from_all_folders(data_dir: str) -> pd.DataFrame:
    """Load JSON files from all folders in a directory and concatenate them.

    Args:
        data_dir (str): Directory containing folders with JSON files.

    Returns:
        pd.DataFrame: Concatenated DataFrame from all JSON files.

    Raises:
        FileNotFoundError: If a folder or file does not exist.

    Example:
        >>> load_json_file_from_all_folders("Benchmark/")
        pd.DataFrame(...)
    """
    all_data = []
    folder_count = 0

    for folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.endswith('.json'):
                continue
            json_path = os.path.join(folder_path, filename)
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    df = pd.DataFrame(data)
                    df['domain'] = folder
                    df['dataset'] = filename
                    all_data.append(df)
                except Exception as e:
                    print(f"Failed to load {json_path}: {e}")

        folder_count += 1
        if DEBUG and folder_count >= 3:
            break

    benchmark_df = pd.concat(all_data, ignore_index=True)
    return benchmark_df


# === Check for contamination (only in 'text' column) ===
def is_contaminated(text: str, values_to_compare: list) -> bool:
    """Check if a text is present in a list of values.

    Args:
        text (str): Text to check.
        values_to_compare (list): List of values to compare against.

    Returns:
        bool: True if contaminated, False otherwise.

    Example:
        >>> is_contaminated("sample", ["sample", "other"])
        True
    """
    values_to_compare = values_to_compare if isinstance(values_to_compare, list) else list(values_to_compare)
    return text in values_to_compare if pd.notna(text) else False


def check_contamination_in_df(_df: pd.DataFrame, uncleaned_dataframe_values: Union[list, set]) -> tuple:
    """Check contamination in a DataFrame and summarize by domain and dataset.

    Args:
        _df (pd.DataFrame): DataFrame to check.
        uncleaned_dataframe_values (Union[list, set]): Values to compare for contamination.

    Returns:
        tuple: (DataFrame with contamination column, contaminated rows DataFrame, summary DataFrame)

    Example:
        >>> check_contamination_in_df(df, ["text1", "text2"])
        (df, contamination_df, summary_df)
    """
    _df['contaminated'] = _df['text'].apply(is_contaminated, values_to_compare=uncleaned_dataframe_values)

    # === Filter contaminated rows ===
    contamination_df = _df[_df['contaminated']].copy()

    # === Count contamination by domain and dataset ===
    summary = contamination_df.groupby(['domain', 'dataset']).size().reset_index(name='contamination_count')
    return _df, contamination_df, summary


def analyze_df_for_specific_hints_of_llms(
    _df_temp: pd.DataFrame,
    column_generated_text: str = 'direct_prompt',
    print_summary: bool = True,
    print_results: bool = True,
    print_summary_by_llm: bool = True,
    _pattern: str = PATTERN_COMBINED
) -> tuple:
    """Analyze DataFrame for rows matching LLM-specific patterns.

    Args:
        _df_temp (pd.DataFrame): DataFrame to analyze.
        column_generated_text (str): Column to check for patterns.
        print_summary (bool): Print summary statistics.
        print_results (bool): Print matching/non-matching results.
        print_summary_by_llm (bool): Print summary by LLM type.
        _pattern (str): Regex pattern to match.

    Returns:
        tuple: (matching_rows DataFrame, non_matching_rows DataFrame)

    Example:
        >>> analyze_df_for_specific_hints_of_llms(df, "direct_prompt")
        (matching_rows, non_matching_rows)
    """
    matching_rows = _df_temp[_df_temp[column_generated_text].str.match(_pattern, na=False)]
    if print_summary: print("Entries with typical LLM Patterns: ", len(matching_rows))
    # Filter rows NOT matching any of the patterns
    non_matching_rows = _df_temp[~_df_temp[column_generated_text].str.match(_pattern, na=False)]

    if print_summary: print("Entries without typical LLM Patterns: ", len(non_matching_rows))
    if print_summary_by_llm: print(f"""
Entries without typical LLM Patterns: {non_matching_rows["llm_type"].value_counts()}

Entries with typical LLM Patterns: {matching_rows["llm_type"].value_counts()}
    """)
    if print_results:
        print("\nNot contaminated generated strings: ")
        for item in non_matching_rows[column_generated_text]:
            if print_results: print("\n", item)
    return matching_rows, non_matching_rows


# Function to remove the matched part if it starts with the pattern
def remove_prefix_of_text(
    row: pd.Series,
    _pattern: str,
    _pattern_rejection: str,
    text_column: str = "direct_prompt",
    question_column_name: str = "title"
) -> Union[str, None]:
    """Remove prefix from text based on patterns and question column.

    Args:
        row (pd.Series): Row of DataFrame.
        _pattern (str): Regex pattern to remove.
        _pattern_rejection (str): Pattern for rejection.
        text_column (str): Column name for text.
        question_column_name (str): Column name for question.

    Returns:
        Union[str, None]: Cleaned text or None if rejected.

    Example:
        >>> remove_prefix_of_text(row, PATTERN_CLEANUP, PATTERN_REJECTION)
        'Cleaned text'
    """
    text = row[text_column]
    title = row[question_column_name]
    if pd.isna(text) or re.search(_pattern_rejection, text, re.IGNORECASE):
        return None
    # try:
    if not ("given article title" in text or "provided article title" in text):
        text = re.sub(rf'((["]|[\*]{2})*{re.escape(title)}("|\*{2}))', '', text, count=1,
                      flags=re.IGNORECASE).lstrip()
    # except Exception as e:
    #     logging.warning(f"{title}, error: {e}")
    if question_column_name == "start":
        text = re.sub(PATTERN_REVIEW, "", text, count=3, flags=re.IGNORECASE).lstrip()

    return re.sub(_pattern, '', text, count=1).lstrip()


def remove_prefix_of_text_advanced(
    row: pd.Series,
    _pattern: str = PATTERN_CLEANUP,
    _pattern_rejection: str = PATTERN_REJECTION,
    text_column: str = "direct_prompt",
    question_column_name: str = "title"
) -> Union[str, None]:
    """Advanced removal of prefix from text using multiple patterns.

    Args:
        row (pd.Series): Row of DataFrame.
        _pattern (str): Regex pattern to remove.
        _pattern_rejection (str): Pattern for rejection.
        text_column (str): Column name for text.
        question_column_name (str): Column name for question.

    Returns:
        Union[str, None]: Cleaned text or None if rejected.

    Example:
        >>> remove_prefix_of_text_advanced(row)
        'Cleaned text'
    """
    text = row[text_column]
    title = row[question_column_name]
    if pd.isna(text) or re.search(_pattern_rejection, text, re.IGNORECASE):
        return None
    if not ("given article title" in text or "provided article title" in text):
        text = re.sub(rf'((["]|[\*]{2})?{re.escape(title)}(["]|[\*]{2}))', '', text, count=1,
                      flags=re.IGNORECASE).lstrip()
    if question_column_name == "title":
        text = re.sub(PATTERN_ABSTRACT, "", text, count=3, flags=re.IGNORECASE).lstrip()
    if question_column_name == "summary":
        text = re.sub(PATTERN_XSUM, "", text, count=3, flags=re.IGNORECASE).lstrip()
    if question_column_name == "story_prompt":
        text = re.sub(PATTERN_ARTICLE, "", text, count=3, flags=re.IGNORECASE).lstrip()
    if question_column_name == "start":
        text = re.sub(PATTERN_REVIEW, "", text, count=3, flags=re.IGNORECASE).lstrip()

    text = re.sub(_pattern, '', text, count=1, flags=re.IGNORECASE).lstrip()
    text = re.sub(REMOVE_FIRST_OCCURRENCE, '', text, count=1, flags=re.IGNORECASE).lstrip()

    if text_column == "paraphrase_polish_llm" or text_column == "paraphrase_polish_human":
        text = re.sub(PATTERN_POLISHING, "", text, count=1, flags=re.IGNORECASE).lstrip()
    if text_column == "prompt_SICO":
        text = re.sub(PATTERN_SICO, "", text, flags=re.IGNORECASE).lstrip()

    return re.sub(REMOVE_ONLY_PATTERN, '', text, flags=re.IGNORECASE).lstrip()


def print_dataframe_separated_by_llm(
    df_to_be_printed: pd.DataFrame,
    printing: int = 2,
    question_column_name: str = "start",
    start_counter: int = 0,
    end_counter: int = 1000
) -> None:
    """Print DataFrame rows separated by LLM type.

    Args:
        df_to_be_printed (pd.DataFrame): DataFrame to print.
        printing (int): Print mode (1 for full, 2 for cleaned only).
        question_column_name (str): Column name for question.
        start_counter (int): Start index.
        end_counter (int): End index.

    Returns:
        None

    Example:
        >>> print_dataframe_separated_by_llm(df)
    """
    for llm in df_to_be_printed["llm_type"].unique():
        df_to_be_printed_temp = df_to_be_printed[df_to_be_printed["llm_type"] == llm]
        print(f"\n\n\n LLM TYPE: {llm} \n\n")
        for counter, (_, item) in enumerate(df_to_be_printed_temp.iterrows()):
            if counter > end_counter:
                break
            if counter < start_counter:
                continue
            elif printing == 1:
                print(f"""{item[question_column_name]}

PROMPT: {item["direct_prompt"]}

CLEANED: {item["direct_prompt_cleaned"]}

        """)
            else:
                print(item["direct_prompt_cleaned"], "\n")


def balanced_truncate_json_file(
    input_path: str,
    output_path: str,
    num_entries: int = 10,
    labels: list = ["human", "llm"]
) -> None:
    """Truncate a JSON file to a balanced number of entries per label.

    Args:
        input_path (str): Path to input JSON file.
        output_path (str): Path to output JSON file.
        num_entries (int): Total number of entries to keep.
        labels (list): List of labels to balance.

    Returns:
        None

    Example:
        >>> balanced_truncate_json_file("input.json", "output.json", 10)
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Separate by label
    label_data = {label: [] for label in labels}
    for entry in data:
        label = entry.get("label")
        if label in label_data and len(label_data[label]) < num_entries // 2:
            label_data[label].append(entry)
        if all(len(label_data[l]) >= num_entries // 2 for l in labels):
            break

    # Combine balanced entries
    truncated_data = label_data["human"] + label_data["llm"]

    # Only save if we have enough for both labels
    if len(truncated_data) == num_entries:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(truncated_data, f, indent=2)
    else:
        print(f"⚠️ Skipping {input_path} — not enough data for both labels.")


def clean_and_store_df(
    _df: pd.DataFrame,
    path_name: str = "arxiv",
    question_column_name: str = "title",
    printing: int = 0,
    store_dir: str = f"{BASE_DIR}/results/cleaned_detectRL_files",
    suffix_path_name: str = "",
    full_storage_path: str = None,
    columns_to_be_cleaned: tuple = COLUMNS_DIRECTLY_LLM_GENERATED_DETECT_RL
) -> None:
    """Clean specified columns in a DataFrame and store the result as a parquet file.

    Args:
        _df (pd.DataFrame): DataFrame to clean.
        path_name (str): Name for output file.
        question_column_name (str): Column name for question.
        printing (int): Print mode.
        store_dir (str): Directory to store cleaned file.
        suffix_path_name (str): Suffix for output file name.
        full_storage_path (str): Full path for storage (overrides store_dir).
        columns_to_be_cleaned (tuple): Columns to clean.

    Returns:
        None

    Example:
        >>> clean_and_store_df(df, "arxiv", "title")
    """
    for _column in columns_to_be_cleaned:
        _df[_column] = _df.apply(remove_prefix_of_text, _pattern=PATTERN_CLEANUP, _pattern_rejection=PATTERN_REJECTION,
                                 text_column=_column, axis=1,
                                 question_column_name=question_column_name)
        df_claude = _df[_df["llm_type"] == "Claude-instant"]
        if printing != 0:
            for counter, (_, item) in enumerate(df_claude.iterrows()):
                if counter < 20:
                    if printing == 1:
                        print(f"""{item[question_column_name]}

        PROMPT: {item["direct_prompt"]}

        CLEANED: {item[_column]}

                """)
                    else:
                        print(item[_column], "\n")

    if full_storage_path is not None:
        path_to_store_df = full_storage_path
    else:
        path_to_store_df = f'{store_dir}{path_name}_2800_cleaned_all{suffix_path_name}.parquet'
    print("store dataframe in:", path_to_store_df)
    _df.to_parquet(path_to_store_df)


class TrainRobertaHelper:
    """Helper class for preparing data and running predictions with RoBERTa."""

    @staticmethod
    def prepare_df_for_roberta_training(
        df: pd.DataFrame,
        column_to_be_used_for_text: Union[str, list] = "direct_prompt",
        column_to_be_used_for_human: str = "abstract",
        column_title: str = "title",
        paraphrase_polish_human_as_ai: bool = True
    ) -> pd.DataFrame:
        """Prepare DataFrame for RoBERTa training.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column_to_be_used_for_text (Union[str, list]): Column(s) for LLM text.
            column_to_be_used_for_human (str): Column for human text.
            column_title (str): Column for context/title.
            paraphrase_polish_human_as_ai (bool): Treat 'paraphrase_polish_human' as AI if True.

        Returns:
            pd.DataFrame: Prepared DataFrame for training.

        Example:
            >>> TrainRobertaHelper.prepare_df_for_roberta_training(df)
            pd.DataFrame(...)
        """
        column_to_be_used_for_text = [column_to_be_used_for_text] if isinstance(column_to_be_used_for_text,
                                                                                str) else column_to_be_used_for_text
        list_dfs_with_multiple_prompts = []
        for _llm_column_temp in column_to_be_used_for_text:
            # Subset for human annotations
            df_human = df.loc[:, ["id", column_title, "llm_type", column_to_be_used_for_human, "domain"]]
            df_human["label"] = "human"
            df_human = df_human.rename(columns={
                column_to_be_used_for_human: "text",
                column_title: "context"
            })

            df_llm = df.loc[:, ["id", column_title, "llm_type", _llm_column_temp, "domain"]]

            if _llm_column_temp == "paraphrase_polish_human" and not paraphrase_polish_human_as_ai:
                df_llm["label"] = "human"
            else:
                # Subset for LLM annotations
                df_llm["label"] = "llm"

            df_llm = df_llm.rename(columns={
                _llm_column_temp: "text",
                column_title: "context"
            })

            # Combine both
            _df_combined = pd.concat([df_human, df_llm], ignore_index=True)
            _df_combined = _df_combined.dropna(subset=["label", "text"], how="any")
            _df_combined["llm_prompting_strategy"] = _llm_column_temp
            _df_combined["text"].astype(str)
            list_dfs_with_multiple_prompts.append(_df_combined)
        return pd.concat(list_dfs_with_multiple_prompts)

    @staticmethod
    def load_model_and_tokenizer(trained_model_path: str, device: str = "cuda"):
        """Load a trained RoBERTa model and tokenizer.

        Args:
            trained_model_path (str): Path to the trained model.
            device (str): Device to load model on.

        Returns:
            tuple: (detector model, tokenizer)

        Example:
            >>> TrainRobertaHelper.load_model_and_tokenizer("model_path")
            (model, tokenizer)
        """
        detector = transformers.AutoModelForSequenceClassification.from_pretrained(trained_model_path).to(device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(trained_model_path)
        return detector, tokenizer

    @staticmethod
    def predict_example(_tokenizer, _detector, _example) -> list:
        """Predict probabilities for a single example.

        Args:
            _tokenizer: Tokenizer object.
            _detector: Model object.
            _example: Example text(s) to predict.

        Returns:
            list: List of prediction probabilities.

        Example:
            >>> TrainRobertaHelper.predict_example(tokenizer, detector, ["text"])
            [0.95, 0.05]
        """
        tokenized = _tokenizer(_example, padding=True, truncation=True, max_length=512,
                               return_tensors="pt").to("cuda")
        _detector.eval()
        with torch.no_grad():  # good practice for inference
            outputs = _detector(**tokenized).logits.softmax(-1).tolist()
        return outputs

    @staticmethod
    def predict_dataframe(
        predict_data: Union[pd.DataFrame, list, np.array, pd.Series],
        model: transformers.AutoModelForSequenceClassification = None,
        tokenizer: transformers.AutoTokenizer = None,
        trained_model_path: str = None,
        device: str = "cuda",
        column_text: str = "text"
    ) -> list:
        """Predict probabilities for a batch of data.

        Args:
            predict_data (Union[pd.DataFrame, list, np.array, pd.Series]): Data to predict.
            model: Model object.
            tokenizer: Tokenizer object.
            trained_model_path (str): Path to trained model.
            device (str): Device for prediction.
            column_text (str): Column name for text.

        Returns:
            list: List of prediction probabilities.

        Raises:
            ValueError: If model/tokenizer are not properly specified.

        Example:
            >>> TrainRobertaHelper.predict_dataframe(df)
            [0.9, 0.1, ...]
        """
        if isinstance(predict_data, pd.DataFrame):
            predict_data = predict_data[column_text]
        if tokenizer is None and model is None:
            model, tokenizer = TrainRobertaHelper.load_model_and_tokenizer(trained_model_path, device)
        elif (tokenizer is None) or (model is None):
            raise ValueError("Either (tokenizer and model) or model_path has to be defined")
        else:
            tokenizer = tokenizer
            model = model.to(device)

        predictions = []
        for item in tqdm(predict_data):
            predictions.append(TrainRobertaHelper.predict_example(tokenizer, model, [item])[0][0])
        return predictions

    @staticmethod
    def get_predictions_for_dataframes(df_list: list[pd.DataFrame], model_path: str) -> pd.DataFrame:
        """Get predictions for a list of DataFrames using a model.

        Args:
            df_list (list[pd.DataFrame]): List of DataFrames.
            model_path (str): Path to trained model.

        Returns:
            pd.DataFrame: DataFrame with predictions.

        Example:
            >>> TrainRobertaHelper.get_predictions_for_dataframes([df1, df2], "model_path")
            pd.DataFrame(...)
        """
        result_list: list = []
        for counter, _df in enumerate(df_list):
            df_hash = train_roberta.hash_dataframe_as_parquet(_df)
            result_path = f"{model_path}/{df_hash}.roberta-base_data.json"
            if os.path.exists(result_path):
                with open(result_path) as fp:
                    result = json.load(fp)
                fp.close()
                result_list.append(pd.DataFrame(result))
            else:
                _df["prediction"] = TrainRobertaHelper.predict_dataframe(_df, trained_model_path=model_path)
                result_list.append(_df)

        return pd.concat(result_list)

    @staticmethod
    def print_top_x_predictions(_df: pd.DataFrame, x: int = 20, sort_by="prediction", ascending: bool = True) -> None:
        """Print top X predictions from a DataFrame.

        Args:
            _df (pd.DataFrame): DataFrame with predictions.
            x (int): Number of top predictions to print.
            sort_by (str): Column to sort by.
            ascending (bool): Sort order.

        Returns:
            None

        Example:
            >>> TrainRobertaHelper.print_top_x_predictions(df, x=10)
        """
        _df.sort_values(by=sort_by, ascending=ascending, inplace=True)
        for _, item in _df.iloc[:x].iterrows():
            print(f"""{item["id"]}, {item["llm_type"]} ({item.label}), {round(item["prediction"], 4)}, 
Training LLM: {item.llm_training}, prompt: {item["llm_prompting_strategy"]}
{item["context"]}

    {item.text}

        """)

    @staticmethod
    def add_information_of_top_x_predictions(
        to_be_recleaned: list,
        _df: pd.DataFrame,
        x: int = 20,
        sort_by="prediction",
        ascending: bool = True
    ) -> list:
        """Add info of top X predictions to a list.

        Args:
            to_be_recleaned (list): List to append info to.
            _df (pd.DataFrame): DataFrame with predictions.
            x (int): Number of top predictions.
            sort_by (str): Column to sort by.
            ascending (bool): Sort order.

        Returns:
            list: Updated list with top prediction info.

        Example:
            >>> TrainRobertaHelper.add_information_of_top_x_predictions([], df, x=5)
            [{'id': ..., ...}, ...]
        """
        _df.sort_values(by=sort_by, ascending=ascending, inplace=True)

        for _, item in _df.iloc[:x].iterrows():
            to_be_recleaned.append({
                "id": item.id,
                "llm_type_test": item.llm_type,
                "label": item.label,
                "domain": item.domain,
                "text": item.text,
                "prompting_strategy": item.llm_prompting_strategy
            })
        return to_be_recleaned


class TrainingDataHandler:
    """Class for loading and preparing training data across domains and LLMs."""

    @staticmethod
    def load_dataframes_all_llms_all_domains(
        data_dir: str = CLEANED_FILES_DIR,
        domains: list = DOMAINS,
        _suffix_path: str = "_cleaned_all_v3.parquet",
        prompts: Union[list, str] = "direct_prompt",
        prepare_for_training: bool = True,
        paraphrase_polish_human_as_ai: bool = True
    ) -> tuple:
        """Load DataFrames for all LLMs and domains.

        Args:
            data_dir (str): Directory for data files.
            domains (list): List of domains.
            _suffix_path (str): Suffix for file names.
            prompts (Union[list, str]): Prompt columns.
            prepare_for_training (bool): Whether to prepare for training.
            paraphrase_polish_human_as_ai (bool): Treat 'paraphrase_polish_human' as AI if True.

        Returns:
            tuple: DataFrames for Claude, Llama, Palm, ChatGPT.

        Example:
            >>> TrainingDataHandler.load_dataframes_all_llms_all_domains()
            (df_claude, df_llama, df_palm, df_chatgpt)
        """
        _df = TrainingDataHandler.load_list_of_dataframes_across_domains(data_dir=data_dir, domains=domains, _suffix_path=_suffix_path,
                                                     prompts=prompts, prepare_for_training=prepare_for_training,
                                                     paraphrase_polish_human_as_ai=paraphrase_polish_human_as_ai)
        subset = ["label", "text"] if prepare_for_training else list(_df.columns)
        how = "any" if prepare_for_training else "all"
        df_claude, df_llama, df_palm, df_chatgpt = [
            _df[_df["llm_type"] == _llm].dropna(how=how, subset=subset).reset_index(drop=True)
            for _llm in ["Claude-instant", "Llama-2-70b", "Google-PaLM", "ChatGPT"]]
        return df_claude, df_llama, df_palm, df_chatgpt
    @staticmethod
    def load_list_of_dataframes_across_domains(
        data_dir: str = ORIGINAL_DATA_DIR,
        domains: list = DOMAINS,
        _suffix_path: str = ".json",
        prompts: Union[list, str] = "direct_prompt",
        prepare_for_training: bool = True,
        paraphrase_polish_human_as_ai: bool = True
    ) -> pd.DataFrame:
        """Load and concatenate DataFrames across domains.

        Args:
            data_dir (str): Directory for data files.
            domains (list): List of domains.
            _suffix_path (str): Suffix for file names.
            prompts (Union[list, str]): Prompt columns.
            prepare_for_training (bool): Whether to prepare for training.
            paraphrase_polish_human_as_ai (bool): Treat 'paraphrase_polish_human' as AI if True.

        Returns:
            pd.DataFrame: Concatenated DataFrame.

        Example:
            >>> TrainingDataHandler.load_list_of_dataframes_across_domains()
            pd.DataFrame(...)
        """
        training_data_frames: list = []
        for _domain in domains:
            training_data_frames.append(
                TrainingDataHandler.load_single_dataframe(f"{data_dir}/{_domain}_2800{_suffix_path}", prompts=prompts,
                                                          prepare_for_training=prepare_for_training,
                                                          paraphrase_polish_human_as_ai=paraphrase_polish_human_as_ai))
        return pd.concat(training_data_frames)

    @staticmethod
    def normalise_human_and_title_columns(
        df: pd.DataFrame,
        column_to_be_used_for_human: str = "abstract",
        column_title: str = "title"
    ) -> pd.DataFrame:
        """Rename columns for human text and context.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column_to_be_used_for_human (str): Column for human text.
            column_title (str): Column for context/title.

        Returns:
            pd.DataFrame: DataFrame with renamed columns.

        Example:
            >>> TrainingDataHandler.normalise_human_and_title_columns(df)
            pd.DataFrame(...)
        """
        df = df.rename(columns={
            column_to_be_used_for_human: "text_human",
            column_title: "context"
        })
        return df

    @staticmethod
    def load_single_dataframe(
        df_path: str,
        prompts: Union[list, str] = "direct_prompt",
        prepare_for_training: bool = True,
        paraphrase_polish_human_as_ai: bool = True
    ) -> pd.DataFrame:
        """Load a single DataFrame from file and prepare for training.

        Args:
            df_path (str): Path to data file.
            prompts (Union[list, str]): Prompt columns.
            prepare_for_training (bool): Whether to prepare for training.
            paraphrase_polish_human_as_ai (bool): Treat 'paraphrase_polish_human' as AI if True.

        Returns:
            pd.DataFrame: Loaded and prepared DataFrame.

        Raises:
            ValueError: If file format is unsupported.

        Example:
            >>> TrainingDataHandler.load_single_dataframe("arxiv_2800.json")
            pd.DataFrame(...)
        """
        if df_path.endswith(".json"):
            _training_df_temp = load_dataframe_from_json(df_path)
        elif df_path.endswith(".parquet"):
            _training_df_temp = pd.read_parquet(df_path)
        else:
            raise ValueError(f"Cannot load data format of path {df_path}")
        domain, prompt_key, human_key = get_info_based_on_input_path(df_path)
        _training_df_temp["domain"] = domain
        if prepare_for_training:
            return TrainRobertaHelper.prepare_df_for_roberta_training(_training_df_temp, column_to_be_used_for_text=prompts,
                                                   column_to_be_used_for_human=human_key, column_title=prompt_key,
                                                   paraphrase_polish_human_as_ai=paraphrase_polish_human_as_ai)
        else:
            return TrainingDataHandler.normalise_human_and_title_columns(_training_df_temp, column_to_be_used_for_human=human_key,
                                                     column_title=prompt_key)

    @staticmethod
    def split_training_data_frame_and_adjust_transfer_test_df(
        dataframe_to_train: pd.DataFrame,
        dataframe_to_adjust: pd.DataFrame,
        _seed: int = SEED
    ) -> tuple:
        """Split training DataFrame and adjust transfer test DataFrame.

        Args:
            dataframe_to_train (pd.DataFrame): DataFrame to split.
            dataframe_to_adjust (pd.DataFrame): DataFrame to adjust.
            _seed (int): Random seed.

        Returns:
            tuple: (train_df, test_df, adjusted_df, sample_ids)

        Example:
            >>> TrainingDataHandler.split_training_data_frame_and_adjust_transfer_test_df(df1, df2)
            (train_df, test_df, adjusted_df, sample_ids)
        """
        train_df, test_df, sample_ids = TrainingDataHandler.split_by_id(dataframe_to_train, test_frac=0.2, seed=_seed)
        adjusted_df = dataframe_to_adjust[dataframe_to_adjust["id"].isin(sample_ids)]
        return train_df, test_df, adjusted_df, sample_ids

    @staticmethod
    def split_by_id(
        df: pd.DataFrame,
        id_column: str = 'id',
        test_frac: float = 0.2,
        seed: int = SEED
    ) -> tuple:
        """Split DataFrame into train and test sets by random sampling of IDs.

        Args:
            df (pd.DataFrame): Input DataFrame.
            id_column (str): Column for IDs.
            test_frac (float): Fraction for test set.
            seed (int): Random seed.

        Returns:
            train_df (pd.DataFrame): DataFrame containing the training data.
            test_df (pd.DataFrame): DataFrame containing the test data.
            sampled_ids (pd.Series): The sampled IDs used for the test set.

        Example:
            >>> TrainingDataHandler.split_by_id(df)
            (train_df, test_df, sampled_ids)
        """
        # Get unique IDs
        unique_ids = df[id_column].drop_duplicates()

        # Sample test IDs
        sampled_ids = unique_ids.sample(frac=test_frac, random_state=seed)

        # Create test and train DataFrames
        _test_df = df[df[id_column].isin(sampled_ids)]
        _train_df = df[~df[id_column].isin(sampled_ids)]

        return _train_df, _test_df, sampled_ids

    @staticmethod
    def balanced_truncate_json_file(
        input_path: str,
        output_path: str,
        num_entries: int = 10
    ) -> None:
        """Truncate a JSON file to a balanced number of entries per label.

        Args:
            input_path (str): Path to input JSON file.
            output_path (str): Path to output JSON file.
            num_entries (int): Total number of entries to keep.

        Returns:
            None

        Example:
            >>> TrainingDataHandler.balanced_truncate_json_file("input.json", "output.json", 10)
        """
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Separate by label
        label_data = {label: [] for label in LABELS}
        for entry in data:
            label = entry.get("label")
            if label in label_data and len(label_data[label]) < num_entries // 2:
                label_data[label].append(entry)
            if all(len(label_data[l]) >= num_entries // 2 for l in LABELS):
                break

        # Combine balanced entries
        truncated_data = label_data["human"] + label_data["llm"]

        # Only save if we have enough for both labels
        if len(truncated_data) == num_entries:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(truncated_data, f, indent=2)
        else:
            print(f"⚠️ Skipping {input_path} — not enough data for both labels.")


class ModelExplainer:
    """Class for explaining model predictions using SHAP."""

    def __init__(
        self,
        directory_results_claude_uncleaned: str,
        _path_cleaned: str,
        _path_not_cleaned: str,
        _seed: int = SEED
    ):
        """Initialize ModelExplainer with model paths.

        Args:
            directory_results_claude_uncleaned (str): Path to uncleaned model.
            _path_cleaned (str): Path to cleaned model.
            _path_not_cleaned (str): Path to second uncleaned model.
            _seed (int): Random seed.

        Example:
            >>> ModelExplainer("uncleaned/", "cleaned/", "uncleaned2/")
        """
        seed_everything(_seed)
        self.model_claude_uncleaned = transformers.AutoModelForSequenceClassification.from_pretrained(
            directory_results_claude_uncleaned)
        self.tokenizer_claude_uncleaned = transformers.AutoTokenizer.from_pretrained(directory_results_claude_uncleaned)
        self.model_claude_cleaned = transformers.AutoModelForSequenceClassification.from_pretrained(_path_cleaned)
        self.tokenizer_claude_cleaned = transformers.AutoTokenizer.from_pretrained(_path_cleaned)
        self.model_claude_uncleaned2 = transformers.AutoModelForSequenceClassification.from_pretrained(_path_not_cleaned)
        self.tokenizer_claude_uncleaned2 = transformers.AutoTokenizer.from_pretrained(_path_not_cleaned)

        # Set model_claude_uncleaned to eval mode and move to CPU or CUDA as needed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = "cpu"
        self.model_claude_uncleaned.to(self.device)
        self.model_claude_uncleaned.eval()
        self.model_claude_cleaned.to(self.device)
        self.model_claude_cleaned.eval()
        self.model_claude_uncleaned2.to(self.device)
        self.model_claude_uncleaned2.eval()

    # Wrapper function to tokenize and predict probabilities
    def f_predict(self, texts: list, tokenizer, _model) -> np.ndarray:
        """Tokenize and predict probabilities for a list of texts.

        Args:
            texts (list): List of input texts.
            tokenizer: Tokenizer object.
            _model: Model object.

        Returns:
            np.ndarray: Array of prediction probabilities.

        Example:
            >>> mexp.f_predict(["text"], tokenizer, model)
            array([0.8])
        """
        # Ensure input is a list of strings
        texts = [str(t) for t in texts]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = _model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[:, 0]
        return probs.cpu().numpy()

    def f_predict_uncleaned(self, texts: list) -> np.ndarray:
        """Predict probabilities using the uncleaned model.

        Args:
            texts (list): List of input texts.

        Returns:
            np.ndarray: Array of probabilities.

        Example:
            >>> mexp.f_predict_uncleaned(["text"])
            array([0.7])
        """
        return self.f_predict(texts, self.tokenizer_claude_uncleaned, self.model_claude_uncleaned)

    def f_predict_cleaned(self, texts: list) -> np.ndarray:
        """Predict probabilities using the cleaned model.

        Args:
            texts (list): List of input texts.

        Returns:
            np.ndarray: Array of probabilities.

        Example:
            >>> mexp.f_predict_cleaned(["text"])
            array([0.9])
        """
        return self.f_predict(texts, self.tokenizer_claude_cleaned, self.model_claude_cleaned)

    def f_predict_uncleaned2(self, texts: list) -> np.ndarray:
        """Predict probabilities using the second uncleaned model.

        Args:
            texts (list): List of input texts.

        Returns:
            np.ndarray: Array of probabilities.

        Example:
            >>> mexp.f_predict_uncleaned2(["text"])
            array([0.6])
        """
        return self.f_predict(texts, self.tokenizer_claude_uncleaned2, self.model_claude_uncleaned2)

    def generate_shap_values(
        self,
        input_text: Union[str, list],
        classifier_to_use: str = "uncleaned"
    ):
        """Generate SHAP values for input text using specified classifier.

        Args:
            input_text (Union[str, list]): Input text(s) to explain.
            classifier_to_use (str): Which classifier to use ('uncleaned', 'cleaned', 'uncleaned2').

        Returns:
            shap.Explanation: SHAP values object.

        Raises:
            ValueError: If classifier_to_use is invalid.

        Example:
            >>> mexp.generate_shap_values("sample text", "cleaned")
            shap.Explanation(...)
        """
        if classifier_to_use == "uncleaned":
            pred_func, tokenizer = self.f_predict_uncleaned, self.tokenizer_claude_uncleaned
        elif classifier_to_use == "cleaned":
            pred_func, tokenizer = self.f_predict_cleaned, self.tokenizer_claude_cleaned
        elif classifier_to_use == "uncleaned2":
            pred_func, tokenizer = self.f_predict_uncleaned2, self.tokenizer_claude_uncleaned2
        else:
            raise ValueError("selected classifier not available")
        # Create a SHAP explainer for the text classifier
        explainer = shap.Explainer(pred_func, tokenizer)
        input_text = [input_text] if isinstance(input_text, str) else input_text
        # Explain the prediction for the input text
        return explainer(input_text)

    @staticmethod
    def generate_shap_plots(
        values,
        type_of_plot: str = "text",
        display: bool = False,
        max_plot_waterfall: int = 15
    ):
        """Generate SHAP plots for explanation values.

        Args:
            values: SHAP values object.
            type_of_plot (str): Type of plot ('text' or 'waterfall').
            display (bool): Whether to display the plot.
            max_plot_waterfall (int): Max features for waterfall plot.

        Returns:
            matplotlib.axes.Axes or None: Plot object or None.

        Example:
            >>> ModelExplainer.generate_shap_plots(values, "waterfall")
        """
        # Visualize token attributions (highlighted text)
        if type_of_plot == "text":
            return shap.plots.text(values, display=display)
        else:
            return shap.plots.waterfall(values, max_display=max_plot_waterfall, show=display)

    def generate_shap_values_and_plots(
        self,
        input_text: Union[str, list],
        classifier_to_use: str = "uncleaned",
        type_of_plot: str = "text",
        display: bool = False
    ):
        """Generate SHAP values and plot for input text.

        Args:
            input_text (Union[str, list]): Input text(s) to explain.
            classifier_to_use (str): Which classifier to use.
            type_of_plot (str): Type of plot ('text' or 'waterfall').
            display (bool): Whether to display the plot.

        Returns:
            matplotlib.axes.Axes or None: Plot object or None.

        Example:
            >>> mexp.generate_shap_values_and_plots("sample text", "cleaned", "waterfall")
        """
        values = self.generate_shap_values(input_text, classifier_to_use)
        return ModelExplainer.generate_shap_plots(values[0], type_of_plot, display)
