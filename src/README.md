
**Folder Structure:**
```
src/
│
├── __init__.py
├── PLACEHOLDER_GOOGLE_AUTHENTIFICATION_FILE.json
├── config.py
├── few_shot_prompting_detectrl_examples.json
├── fewshot_prompting.py
├── general_functions_and_patterns_for_detection.py
├── openAI_client.py
├── README.md
└── TextAttackTextBugger/
    ├── __init__.py
    └── text_bugger.py
```

**README Content:**

# src Directory

This directory contains the core source code, configuration files, and supporting modules for the project, which focuses on data processing, prompt engineering, adversarial attacks, and integration with OpenAI.

## File Summaries

- `__init__.py`: Marks the directory as a Python package.
- `PLACEHOLDER_GOOGLE_AUTHENTIFICATION_FILE.json`: JSON configuration file for using the Google Translate API. Necessary if the paraphrase back translation method shall be executed. Place it in source and add the filename to the .env file.
- `config.py`: TO BE MODIFIED by the user. GPU setup and setup of custom cache directories for the huggingface transformers.
- `.env`: rename `.env copy` to env and add the required environmental variables. These will be loaded using the dotenv package.
- `few_shot_prompting_detectrl_examples.json`: JSON file with few-shot prompting examples for DetectRL tasks.
- `fewshot_prompting.py`: Generates prompt and jsonl files for few-shot prompting.
- `general_functions_and_patterns_for_detection.py`: Provides general utility functions and detection patterns used throughout the project.
- `openAI_client.py`: Handles communication and API requests to OpenAI services.
- `README.md`: Project documentation and overview (currently empty).

### TextAttackTextBugger/

- `__init__.py`: Marks the subdirectory as a Python package.
- `text_bugger.py`: Implements the TextBugger adversarial attack algorithm for text data.
