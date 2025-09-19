# Standard library imports
import json

# Third-party imports
import nltk
import pandas as pd

# Local imports
from general_functions_and_patterns_for_detection import (
    RESULT_DIR, BASE_DIR,
    PATTERN_REJECTION, REMOVE_ONLY_PATTERN, REMOVE_FIRST_OCCURRENCE,
    PATTERN_ABSTRACT, PATTERN_ARTICLE, PATTERN_REVIEW, PATTERN_XSUM,
    PATTERN_SICO, PATTERN_POLISHING,
    PATTERN_CLEANUP,
    get_info_based_on_input_path,
)

# Constants
REGULAR_EXPRESSION = {
    "arxiv": PATTERN_ABSTRACT,
    "yelp_review": PATTERN_REVIEW,
    "writing_prompt": PATTERN_ARTICLE,
    "xsum": PATTERN_XSUM,
    "prompt_SICO": PATTERN_SICO,
    "paraphrase_polish_human": PATTERN_POLISHING,
    "paraphrase_polish_llm": PATTERN_POLISHING,
}


# Few-shot example utilities
def get_fewshot_examples(domain: str, prompt_type: str) -> list:
    """Load few-shot examples for a given domain and prompt type.

    Args:
        domain (str): The domain, e.g. 'arxiv'.
        prompt_type (str): The prompt type.

    Returns:
        list: List of few-shot example dicts.

    Raises:
        FileNotFoundError: If the examples file does not exist.
        KeyError: If domain or prompt_type is not found.

    Example:
        >>> get_fewshot_examples("arxiv", "direct_prompt")
        [{'text': '...', 'cleaned_text': '...'}, ...]
    """
    with open(f"{BASE_DIR}/src/few_shot_prompting_detectrl_examples.json", "r") as fp:
        few_shot_examples = json.load(fp)
    return few_shot_examples[domain][prompt_type]


def format_few_shot_examples(examples: list) -> str:
    """Format few-shot examples into a string for prompt construction.

    Args:
        examples (list): List of dicts with 'text' and 'cleaned_text'.

    Returns:
        str: Formatted examples string.

    Example:
        >>> format_few_shot_examples([{'text': 'A', 'cleaned_text': 'B'}])
        '## Example 1\nORIGINAL TEXT: A\nCLEANED VERSION: B\n\n'
    """
    result = """"""
    for _counter, example in enumerate(examples):
        result += f"""## Example {_counter + 1}
ORIGINAL TEXT: {example["text"]}
CLEANED VERSION: {example["cleaned_text"]}

"""

    return result


def get_and_format(domain: str, prompt_type: str) -> str:
    """Load and format few-shot examples for a domain and prompt type.

    Args:
        domain (str): The domain.
        prompt_type (str): The prompt type.

    Returns:
        str: Formatted few-shot examples string.

    Example:
        >>> get_and_format("arxiv", "direct_prompt")
        '## Example 1\nORIGINAL TEXT: ...'
    """
    _fs_examples = get_fewshot_examples(domain, prompt_type)
    return format_few_shot_examples(_fs_examples)


# Prompt construction
def get_prompt(domain: str, prompt: str, sentences_num: int, prompt_type: str, icl_examples: str) -> str:
    """Generate a prompt string for a given domain and prompt type.

    Args:
        domain (str): The domain, e.g. 'arxiv', 'xsum', etc.
        prompt (str): The input prompt text.
        sentences_num (int): Number of sentences to generate.
        prompt_type (str): Type of prompt, e.g. 'direct_prompt'.
        icl_examples (str): Few-shot examples as a string.

    Returns:
        str: The constructed prompt string.

    Raises:
        KeyError: If the domain or prompt_type is not found.

    Example:
        >>> get_prompt("arxiv", "Quantum Mechanics", 3, "direct_prompt", "")
        'Given the academic article title, write an academic article abstract with 3 sentences:...'
    """
    prompt_center = {
        "arxiv": {
            "direct_prompt": f'Given the academic article title, write an academic article abstract with {sentences_num} sentences:\n' \
                             f'title: {prompt}\n' \
                             f'abstract:',
            "prompt_few_shot": f'Here are the academic articles titles and academic articles abstracts pairs:\n\n{icl_examples}\n\n' \
                               f'Given the academic article title, write an academic article abstract with {sentences_num} sentences:\n' \
                               f'title: {prompt}\n' \
                               f'abstract:',
            "prompt_ICO_step1": f'Here are the writings from AI and human:\n\n{icl_examples}\n\n' \
                                f'Compare and give the key distinct feature (specifically vocabulary, sentence structure) of human’s writings (do not use examples):',
            "prompt_ICO_step2": f'Based on the description, given the academic article title, write an academic article abstract with {sentences_num} sentences in human style writings:\n' \
                                f'title: {prompt}\n' \
                                f'human:',
            "paraphrase_polish": f'Given the article abstract, polish the writing to meet the academic abstract style with {sentences_num} sentences, ' \
                                 f'improve the spelling, grammar, clarity, concision and overall readability:\n' \
                                 f'abstract: {prompt}\n' \
                                 f'polished abstract:'
        },
        "xsum": {
            "direct_prompt": f'Given the news summary, write a news article with {sentences_num} sentences:\n' \
                             f'news summary: {prompt}\n' \
                             f'news article:',
            "prompt_few_shot": f'Here are the news summaries and news articles pairs:\n\n{icl_examples}\n\n' \
                               f'Given the news summary, write a news article with {sentences_num} sentences:\n' \
                               f'news summary: {prompt}\n' \
                               f'news article:',
            "prompt_ICO_step1": f'Here are the writings from AI and human:\n\n{icl_examples}\n\n' \
                                f'Compare and give the key distinct feature (specifically vocabulary, sentence structure) of human’s writings (do not use examples):',
            "prompt_ICO_step2": f'Based on the description, given the news summary, write a news article with {sentences_num} sentences in human style writings:\n' \
                                f'news summary: {prompt}\n' \
                                f'human:',
            "paraphrase_polish": f'Given the news article, polish the writing to meet the news article style with {sentences_num} sentences, ' \
                                 f'improve the spelling, grammar, clarity, concision and overall readability:\n' \
                                 f'news article: {prompt}\n' \
                                 f'polished news article:'
        },
        "writing_prompt": {
            "direct_prompt": f'Given the writing prompt, write a story with {sentences_num} sentences:\n' \
                             f'writing prompt: {prompt}\n' \
                             f'story:',
            "prompt_few_shot": f'Here are the writing prompts and stories pairs:\n\n{icl_examples}\n\n' \
                               f'Given the writing prompt, write a story with {sentences_num} sentences:\n' \
                               f'writing prompt: {prompt}\n' \
                               f'story:',
            "prompt_ICO_step1": f'Here are the writings from AI and human:\n\n{icl_examples}\n\n' \
                                f'Compare and give the key distinct feature (specifically vocabulary, sentence structure) of human’s writings (do not use examples):',
            "prompt_ICO_step2": f'Based on the description, given the writing prompt, write a story with {sentences_num} sentences in human style writings:\n' \
                                f'writing prompt: {prompt}\n' \
                                f'story:',
            "paraphrase_polish": f'Given the story, polish the writing to meet the story style, ' \
                                 f'improve the spelling, grammar, clarity, concision and overall readability:\n' \
                                 f'story: {prompt}\n' \
                                 f'polished story:'
        },
        "yelp_review": {
            "direct_prompt": f'Given the review\'s first sentence, please help to continue the review with {sentences_num} sentences (do not reject me):\n' \
                             f'review\'s first sentence: {prompt}\n' \
                             f'continued review:',
            "prompt_few_shot": f'Here are the reviews\' first sentence and continued reviews pairs:\n\n{icl_examples}\n\n' \
                               f'Given the review\'s first sentence, continue the review with {sentences_num} sentences:\n' \
                               f'review\'s first sentence: {prompt}\n' \
                               f'continued review:',
            "prompt_ICO_step1": f'Here are the writings from AI and human:\n\n{icl_examples}\n\n' \
                                f'Compare and give the key distinct feature (specifically vocabulary, sentence structure) of human’s writings (do not use examples):',
            "prompt_ICO_step2": f'Based on the description, given the review\'s first sentence, continue the review with {sentences_num} sentences in human style writings:\n' \
                                f'review\'s first sentence: {prompt}\n' \
                                f'continued review:',
            "paraphrase_polish": f'Given the review, polish the writing to meet the review style, ' \
                                 f'improve the spelling, grammar, clarity, concision and overall readability:\n' \
                                 f'review: {prompt}\n' \
                                 f'polished review:'
        },
    }
    return prompt_center[domain][prompt_type]


# Message construction for LLM
def get_system_message(
    prompt_provided: str,
    _regular_expression: str,
    context_dependent_examples: str,
    prompt_use_case: str,
    pattern_rejection: str = PATTERN_REJECTION
) -> dict:
    """Construct the system message for data cleaning prompt.

    Args:
        prompt_provided (str): The original prompt text.
        _regular_expression (str): Regular expressions for cleaning.
        context_dependent_examples (str): Context-dependent examples string.
        prompt_use_case (str): Description of the prompt use case.
        pattern_rejection (str, optional): Pattern for rejection. Defaults to PATTERN_REJECTION.

    Returns:
        dict: System message dictionary for LLM.

    Example:
        >>> get_system_message("title", "regex", "examples", "use_case")
        {'role': 'system', 'content': [...]}
    """
    return {
        "role": "system",
        "content": [
            {
                "type": "input_text",
                "text": fr"""You are a helpful data cleaning assistant, that helps to clean frequent LLM patterns out of a dataset. You will receive the original prompt as well as the original LLM answer. The goal is to remove everything that is related to typical LLM answers or related to the prompt and to only return the text answer of the original task.

# Instructions
- remove all patterns that are typically starting phrases of an LLM response, like (^Here is.*?|^Here are.*?|^Here's.*?|^Sure,\s?here.*?)[.:] at the beginning of the text
- remove phrases provided to the LLM, like the {prompt_provided}
- never remove any content, that is used by the LLM to answer the original prompt. Return the full text to answer the original prompt.
- if the original LLM call results in a rejection (e.g. {pattern_rejection}) just return "REJECTION"
- if the llm does not answer the actual task of the prompt but just describes how one can solve the task, just return "TASK_DESCRIPTION"
- if the original LLM call has no typical patterns, just return the original text as it is
- clean the following patterns only at their first occurrence: {REMOVE_FIRST_OCCURRENCE}
- although you are not restricted to those, you can use the following regular expressions as an example of possible "contamination" to be removed {_regular_expression}. Most of the time the parts before or, if you find the pattern at the end of the text, the text parts after the pattern are also LLM typically. Remove these parts as well if needed.
- for the following patterns only remove the pattern and not everything before or after (if not already found in the instructions before): {REMOVE_ONLY_PATTERN} 

# Output Format
- return only the clean text and no additional context or steps you have taken
- in case of rejection only return "REJECTION" or in case of task description "TASK_DESCRIPTION"
- if you cannot spot any contamination, just return the original text

# Examples
{context_dependent_examples}

# Examples RIGHT vs. WRONG
## Example 1
ORIGINAL TEXT: Polymer quantum mechanics emerges as a fascinating framework within which quantum phenomena manifest in systems with discrete or polymerized structures. This article delves into the study of polymer quantum mechanics and explores its continuum limit. The analysis begins by examining the foundations of polymer quantum mechanics, highlighting its departure from traditional quantum mechanics by incorporating notions of quantized elements and finite-dimensional Hilbert spaces. The evolution of wave functions and the associated dynamics are investigated, revealing significant deviations from the smooth and continuous behavior observed in standard quantum mechanics. Moreover, the concept of the continuum limit is introduced, which aims to connect the discrete polymer quantum mechanics to the well-established continuum quantum mechanics framework. The challenges and opportunities in taking the continuum limit of polymer quantum mechanics are discussed, encompassing topics such as the emergence of classical behavior, the calculation of observables, and the preservation of essential quantum features. Finally, current research directions and potential applications of polymer quantum mechanics in various fields, including condensed matter physics and quantum information science, are outlined. This article serves as a comprehensive introduction to the field, shedding light on the intriguing nature of polymer quantum mechanics and its continuum limit.
RIGHT CLEANED VERSION: Polymer quantum mechanics emerges as a fascinating framework within which quantum phenomena manifest in systems with discrete or polymerized structures. This article delves into the study of polymer quantum mechanics and explores its continuum limit. The analysis begins by examining the foundations of polymer quantum mechanics, highlighting its departure from traditional quantum mechanics by incorporating notions of quantized elements and finite-dimensional Hilbert spaces. The evolution of wave functions and the associated dynamics are investigated, revealing significant deviations from the smooth and continuous behavior observed in standard quantum mechanics. Moreover, the concept of the continuum limit is introduced, which aims to connect the discrete polymer quantum mechanics to the well-established continuum quantum mechanics framework. The challenges and opportunities in taking the continuum limit of polymer quantum mechanics are discussed, encompassing topics such as the emergence of classical behavior, the calculation of observables, and the preservation of essential quantum features. Finally, current research directions and potential applications of polymer quantum mechanics in various fields, including condensed matter physics and quantum information science, are outlined. This article serves as a comprehensive introduction to the field, shedding light on the intriguing nature of polymer quantum mechanics and its continuum limit.
WRONG CLEANED VERSION: Polymer quantum mechanics emerges as a fascinating framework within which quantum phenomena manifest in systems with discrete or polymerized structures. The analysis begins by examining the foundations of polymer quantum mechanics, highlighting its departure from traditional quantum mechanics by incorporating notions of quantized elements and finite-dimensional Hilbert spaces. The evolution of wave functions and the associated dynamics are investigated, revealing significant deviations from the smooth and continuous behavior observed in standard quantum mechanics. Moreover, the concept of the continuum limit is introduced, which aims to connect the discrete polymer quantum mechanics to the well-established continuum quantum mechanics framework. The challenges and opportunities in taking the continuum limit of polymer quantum mechanics are discussed, encompassing topics such as the emergence of classical behavior, the calculation of observables, and the preservation of essential quantum features. Finally, current research directions and potential applications of polymer quantum mechanics in various fields, including condensed matter physics and quantum information science, are outlined.

## Example 2
ORIGINAL TEXT: Here's the abstract: In this study, we present a comprehensive calculation of prompt diphoton production cross sections at the energies of the Tevatron and the Large Hadron Collider (LHC). We focus on the process in which two photons are produced directly and promptly in high-energy hadron collisions. Our calculation incorporates the leading order QCD contributions as well as the relevant electroweak corrections. We provide numerical results for both integrated and differential cross sections, considering a wide range of kinematic variables. Moreover, we investigate the dependence of the results on the choice of renormalization and factorization scales to estimate the associated uncertainties. The impact of the radiative corrections, including higher-order effects and multiple photon emissions, is thoroughly discussed. Our findings contribute to a better understanding of prompt diphoton production and can serve as a valuable reference for experimental analyses at the Tevatron and the LHC, thereby advancing our knowledge of fundamental interactions and the properties of the Higgs boson.
RIGHT CLEANED VERSION: In this study, we present a comprehensive calculation of prompt diphoton production cross sections at the energies of the Tevatron and the Large Hadron Collider (LHC). We focus on the process in which two photons are produced directly and promptly in high-energy hadron collisions. Our calculation incorporates the leading order QCD contributions as well as the relevant electroweak corrections. We provide numerical results for both integrated and differential cross sections, considering a wide range of kinematic variables. Moreover, we investigate the dependence of the results on the choice of renormalization and factorization scales to estimate the associated uncertainties. The impact of the radiative corrections, including higher-order effects and multiple photon emissions, is thoroughly discussed. Our findings contribute to a better understanding of prompt diphoton production and can serve as a valuable reference for experimental analyses at the Tevatron and the LHC, thereby advancing our knowledge of fundamental interactions and the properties of the Higgs boson.
WRONG CLEANED VERSION: We focus on the process in which two photons are produced directly and promptly in high-energy hadron collisions. Our calculation incorporates the leading order QCD contributions as well as the relevant electroweak corrections. We provide numerical results for both integrated and differential cross sections, considering a wide range of kinematic variables. Moreover, we investigate the dependence of the results on the choice of renormalization and factorization scales to estimate the associated uncertainties. The impact of the radiative corrections, including higher-order effects and multiple photon emissions, is thoroughly discussed. Our findings contribute to a better understanding of prompt diphoton production and can serve as a valuable reference for experimental analyses at the Tevatron and the LHC, thereby advancing our knowledge of fundamental interactions and the properties of the Higgs boson. 


# Context
original prompt used for generation: 
{prompt_use_case}"""}
        ]
    }


def get_user_message(LLM_output: str) -> dict:
    """Construct the user message with the LLM output for cleaning.

    Args:
        LLM_output (str): The original LLM output to be cleaned.

    Returns:
        dict: User message dictionary for LLM.

    Example:
        >>> get_user_message("Text to clean")
        {'role': 'user', 'content': [...]}
    """
    return {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": f"""Output of the original LLM to be cleaned:"
{LLM_output}

Cleaned: """}
        ]
    }


# JSONL generation
def generate_jsonl_file(
    df: pd.DataFrame,
    domain: str = "arxiv",
    model: str = "gpt-4.1-mini-2025-04-14",
    output_path: str = None,
    part: int = 0,
    columns_to_be_cleaned: list = None,
    regular_expression: dict = None
) -> None:
    # Set default values for mutable arguments
    if columns_to_be_cleaned is None:
        columns_to_be_cleaned = ['direct_prompt', 'prompt_few_shot', 'prompt_SICO',
                                 'paraphrase_polish_human', 'paraphrase_polish_llm']
    if regular_expression is None:
        regular_expression = REGULAR_EXPRESSION

    """Generate a JSONL file for batch cleaning of LLM outputs.

    Args:
        df (pd.DataFrame): DataFrame containing data to clean.
        domain (str, optional): Domain name. Defaults to "arxiv".
        model (str, optional): Model name. Defaults to "gpt-4.1-mini-2025-04-14".
        output_path (str, optional): Output file path. Defaults to None.
        part (int, optional): Batch index. Defaults to 0.
        columns_to_be_cleaned (list, optional): List of columns to clean.
        regular_expression (dict, optional): Regex patterns per column.

    Returns:
        None

    Raises:
        KeyError: If domain or columns are not found.
        FileNotFoundError: If input files are missing.

    Example:
        >>> generate_jsonl_file(df, domain="arxiv", part=0)
        # Creates a JSONL file with batch cleaning data.
    """
    prompt_provided = \
        {"arxiv": "abstract title", "yelp_review": "first sentence of the review",
         "writing_prompt": "topic of the story",
         "xsum": "news summary"}[domain]
    _, column_prompt, _ = get_info_based_on_input_path(domain)

    reg_exp_domain = PATTERN_CLEANUP + '|' + regular_expression[domain]
    data_jsonl = []

    # indices = [100, 200, 700, 800, 1400, 1500, 2100, 2200]
    # df_temp = df[(df["llm_type"]=="Claude-instant")]        #(df["llm_type"]=="Google-PaLM")|
    df_temp = df.copy(deep=True)
    start = 25 * part
    end = 25 * (part + 1)
    print(f"processing df between {start}:{end}")
    for idx, row in df_temp.iloc[start:end].iterrows():
        for task in columns_to_be_cleaned:
            if task in regular_expression:
                reg_exp_domain += "|" + regular_expression.get(task, "")

            prompt_type = "prompt_polish" if task.startswith(
                "paraphrase_polish") else "prompt_SICO" if task == "prompt_SICO" else "direct_prompt"
            context_dependent_examples = get_and_format("arxiv", prompt_type)

            original_llm_output = row[task]
            if task == "prompt_SICO":
                prompt_use_case = {
                    "prompt_ICO_step1": get_prompt(domain, row[column_prompt],
                                                   len(nltk.sent_tokenize(original_llm_output)), "prompt_ICO_step1",
                                                   "<EXAMPLES>"),
                    "prompt_ICO_step2": get_prompt(domain, row[column_prompt],
                                                   len(nltk.sent_tokenize(original_llm_output)), "prompt_ICO_step2",
                                                   "<EXAMPLES>")
                }
            elif task == "paraphrase_polish_human" or task == "paraphrase_polish_llm":
                prompt_use_case = "paraphrase_polish"
            else:
                prompt_use_case = get_prompt(domain, row[column_prompt], len(nltk.sent_tokenize(original_llm_output)),
                                             task, "<EXAMPLES>")
            custom_id = f"{domain}_{row['id']}_{task}_cleaned"
            data_jsonl.append({"custom_id": custom_id, "method": "POST", "url": "/v1/responses",
                               "body": {
                                   "model": model,
                                   "input": [
                                       get_system_message(prompt_provided, reg_exp_domain, context_dependent_examples,
                                                          prompt_use_case), get_user_message(original_llm_output)],
                                   "text": {"format": {"type": "text"}},
                                   "reasoning": {},
                                   # "tools": [],
                                   "temperature": 0.5,
                                   "max_output_tokens": 3000,
                                   "top_p": 1,
                               }
                               })

    output_path = f"{RESULT_DIR}/data_cleaning_gpt_4-1/{domain}_{model}_batch_processing_cleaning_all_columns_claude_v2_part{part}.jsonl" if output_path is None else output_path
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data_jsonl:
            f.write(json.dumps(item) + "\n")

