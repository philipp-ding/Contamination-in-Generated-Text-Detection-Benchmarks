import json

import nltk
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from src.general_functions_and_patterns_for_detection import RESULT_DIR

client: OpenAI = OpenAI()
"""OpenAI client instance for API interactions."""

def get_batches() -> list:
    """List the latest 100 OpenAI batch jobs.

    Returns:
        list: List of batch job data.

    Example:
        >>> get_batches()
        [{'id': 'batch_...', ...}, ...]
    """
    return client.batches.list(limit=100).data

def get_filename(file_id: str) -> str:
    """Retrieve the filename for a given OpenAI file ID.

    Args:
        file_id (str): The file ID.

    Returns:
        str: The filename.

    Example:
        >>> get_filename("file-abc123")
        'arxiv_gpt-4.1-2025-04-14_batch_processing.jsonl'
    """
    f = client.files.retrieve(file_id=file_id)
    return f.filename

def download_if_recleaning(batch) -> bool:
    """Download batch output if it is a completed re-cleaning batch.

    Args:
        batch: Batch object from OpenAI.

    Returns:
        bool: True if downloaded, False otherwise.

    Example:
        >>> download_if_recleaning(batch)
        True
    """
    if batch.status != "completed":
        return False

    input_fn = get_filename(batch.input_file_id)
    if "re_cleaning" not in input_fn:
        return False

    output_id = batch.output_file_id
    if not output_id:
        print(f"⚠️ Batch {batch.id} has no output_file_id.")
        return False

    content = client.files.content(output_id).text
    target = f"{RESULT_DIR}/re_cleaned/{input_fn}_output.jsonl"
    with open(target, "w") as fo:
        fo.write(content)
    print(f"✔️ Downloaded: {target}")
    return True

def get_user_message_abstract(title: str) -> dict:
    """Create a user message for the abstract generation prompt.

    Args:
        title (str): The article title.

    Returns:
        dict: User message for OpenAI API.

    Example:
        >>> get_user_message_abstract("Quantum Gravity")
        {'role': 'user', ...}
    """
    return {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": f"""TITLE: {title}
ABSTRACT: """
            }
        ]
    }

def get_system_message_abstract(sentence_num: int) -> dict:
    """Create a system message for the abstract generation prompt.

    Args:
        sentence_num (int): Number of sentences to generate.

    Returns:
        dict: System message for OpenAI API.

    Example:
        >>> get_system_message_abstract(5)
        {'role': 'system', ...}
    """
    return {
        "role": "system",
        "content": [
            {
                "type": "input_text",
                "text": f"""You are a helpful assistant who writes an abstract for an academic article given its title.

# Instructions
- directly start with the abstract - do not start with "Here is the abstract" or the title or similar
- write exactly {sentence_num} sentences
- do not reject topics. If you do so just write "None" and not more to make it easy filterable"""
            }
        ]
    }

def generate_openai_response(sentences_num: str, title: str, model: str = "gpt-4.1-2025-04-14") -> str:
    """Generate an abstract using OpenAI API for a given title and sentence count.

    Args:
        sentences_num (str): Number of sentences for the abstract.
        title (str): Title of the article.
        model (str): Model name to use.

    Returns:
        str: Generated abstract text.

    Raises:
        openai.OpenAIError: If the API call fails.

    Example:
        >>> generate_openai_response("5", "Quantum Gravity")
        'Quantum gravity is ...'
    """
    response = client.responses.create(
        model=model,
        input=[
            get_system_message_abstract(sentences_num),
            get_user_message_abstract(title),
        ],
        text={
            "format": {
                "type": "text"
            }
        },
        reasoning={},
        tools=[],
        temperature=1,
        max_output_tokens=2048,
        top_p=1,
        store=True
    )

    return response.output_text

def generate_jsonl_file(df: pd.DataFrame, domain: str = "arxiv", model: str = "gpt-4.1-2025-04-14",
                        output_path: str = None) -> str:
    """Generate a JSONL file for batch OpenAI processing from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with input data.
        domain (str): Domain name for custom_id.
        model (str): Model name to use.
        output_path (str): Output path for JSONL file.

    Returns:
        str: Path to the generated JSONL file.

    Example:
        >>> generate_jsonl_file(df)
        '/path/to/arxiv_gpt-4.1-2025-04-14_batch_processing.jsonl'
    """
    data_jsonl = []
    for idx, row in tqdm(df.iterrows()):
        custom_id = f"{domain}_{row['id']}"
        data_jsonl.append({"custom_id": custom_id, "method": "POST", "url": "/v1/responses",
                           "body": {
                               "model": model,
                               "input": [get_system_message_abstract(len(nltk.sent_tokenize(row["abstract"]))),
                                         get_user_message_abstract(row["title"])],
                               "text": {"format": {"type": "text"}},
                               "reasoning": {},
                               # "tools": [],
                               "temperature": 1,
                               "max_output_tokens": 2048,
                               "top_p": 1,
                           }
                           })

    output_path = f"{RESULT_DIR}/{domain}_{model}_batch_processing.jsonl" if output_path is None else output_path
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data_jsonl:
            f.write(json.dumps(item) + "\n")
    return output_path
