# Notebooks Directory

# Description
Includes all notebooks used to analyse the level of contamination, execute the data cleaning process and train and evaluate the RoBERTa models.
Includes one additional notebook for data generation using OpenAI models.

**Folder Structure:**
```
notebooks/
│
├── 00_README.md
├── 01_data_preparation/
│   ├── 00_data_contamination_exploration.ipynb
│   ├── 01_data_preparation_regex.ipynb
│   ├── 02_train_roberta_identify_samples_for_recleaning.ipynb
│   ├── 03_data_preparation_clean_using_openai.ipynb
│   └── 04_data_prep_regenerate_data_other_attacks.ipynb
├── 02_training_and_eval/
│   ├── T00_train_RoBERTa_basics.ipynb
│   ├── T01_train_RoBERTa_to_identify_samples_for_recleaning.ipynb
│   ├── T02_adversarial_attacks.ipynb
│   ├── T03_RoBERTa_generalisation_results.ipynb
│   ├── TE04_evaluate_roberta_detect_RL_generalisation_results.ipynb
│   └── TE05_explain_RoBERTa_model_outputs_DetectRL.ipynb
└── 09_miscellaneous/
    └── data_generation_openai.ipynb
```

## Folder Overview

- **01_data_preparation/**: Notebooks for exploring, cleaning, and preparing data, including contamination analysis and sample selection for further cleaning.
- **02_training_and_eval/**: Notebooks for training RoBERTa models, performing adversarial attacks, evaluating model generalisation, and explaining model outputs.
- **09_miscellaneous/**: Additional notebooks for data generation and other supporting tasks.

## File Summaries

### 01_data_preparation/
- `00_data_contamination_exploration.ipynb`: Explores and visualizes data contamination issues in the dataset.
- `01_data_preparation_regex.ipynb`: Cleans and preprocesses data using regular expressions.
- `02_train_roberta_identify_samples_for_recleaning.ipynb`: Trains a RoBERTa model to identify samples that require further cleaning.
- `03_data_preparation_clean_using_openai.ipynb`: Uses OpenAI models to clean and improve data quality.
- `04_data_prep_regenerate_data_other_attacks.ipynb`: Regenerates data to simulate other types of adversarial attacks.

### 02_training_and_eval/
- `T00_train_RoBERTa_basics.ipynb`: Basic training of the RoBERTa model on the prepared dataset.
- `T01_train_RoBERTa_to_identify_samples_for_recleaning.ipynb`: Trains RoBERTa specifically to detect samples needing recleaning.
- `T02_adversarial_attacks.ipynb`: Implements and evaluates adversarial attacks on the trained models.
- `T03_RoBERTa_generalisation_results.ipynb`: Analyses the generalisation capabilities of RoBERTa across different datasets.
- `TE04_evaluate_roberta_detect_RL_generalisation_results.ipynb`: Evaluates RoBERTa's performance in detecting RL generalisation results.
- `TE05_explain_RoBERTa_model_outputs_DetectRL.ipynb`: Provides explanations for RoBERTa model outputs in the context of RL detection.

### 09_miscellaneous/
- `data_generation_openai.ipynb`: Script to regenerate the DetectRL data using new OpenAI models (right now GPT-4.1 is used).
