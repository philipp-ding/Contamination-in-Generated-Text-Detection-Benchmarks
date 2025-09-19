## DetectRL fork

## Description
Integration of the DetectRL repository into our work, as it contains essential components for our project.
Some parts of the code have been modified, as those were not executable in our environment.
Other parts of the repository have been left unchanged and untested. Those might still be wrong, as we only executed
and changed the parts we needed.

## Project Structure

```
DetectRL/
│
├── README.md
├── README_DetectRL.md
├── Analysis/
│   ├── benchmark_statistics/
│   │   ├── statistics_lexical_diversity.py
│   │   ├── statistics_n_gram.py
│   │   ├── statistics_readability.py
│   │   └── statistics_text_length.py
│   ├── heatmap/
│   │   ├── heatmap_cross_attacks.py
│   │   ├── heatmap_cross_domains.py
│   │   └── heatmap_cross_models.py
│   ├── leaderboard/
│   │   └── radar.py
│   └── text_length_impact/
│       └── training_time_test_time_text_length_impact.py
├── Benchmark/
│   ├── Benchmark_Data/
│   │   ├── Data_Mixing/
│   │   ├── Data_Mixing_Human/
│   │   ├── Direct_Prompt/
│   │   ├── Multi_Domain/
│   │   ├── Multi_LLM/
│   │   ├── Paraphrase_Attacks/
│   │   ├── Paraphrase_Attacks_Human/
│   │   ├── Perturbation_Attacks/
│   │   ├── Perturbation_Attacks_Human/
│   │   ├── Prompt_Attacks/
│   │   └── Varying_Length/
│   └── Original_Dataset/
│       ├── arxiv_2800.json
│       ├── get_arxiv_data.py
│       ├── get_writing_prompt_data.py
│       ├── get_xsum_data.py
│       ├── get_yelp_review_data.py
│       ├── writing_prompt_2800.json
│       ├── xsum_2800.json
│       └── yelp_review_2800.json
├── Tasks/
│   ├── Task1/
│   ├── Task2/
│   ├── Task3/
│   └── Task4/
├── Data_Generation/
│   ├── __init__.py
│   ├── benchmark construction.py
│   ├── Chat_API.py
│   ├── data_generation.py
│   ├── data_mixing.py
│   ├── DIPPER.py
│   ├── Archive_chromedriver_win32/
│   └── Archive_TextAttack/
├── Detectors/
│   ├── __init__.py
│   ├── binoculars_detector.py
│   ├── binoculars_evaluation.py
│   ├── DetectGPT_evaluation_new.py
│   ├── DetectGPT_evaluation.py
│   ├── DetectGPT.py
│   ├── dna_gpt_evaluation.py
│   ├── entropy_evaluation.py
│   ├── entropy.py
│   ├── evaluate_roberta_detect_RL_generalisation_results.ipynb
│   ├── Fast_DetectGPT_evaluation.py
│   ├── Fast_DetectGPT.py
│   ├── likelihood_evaluation.py
│   ├── logRank_evaluation.py
│   ├── loss.py
│   ├── LRR_evaluation.py
│   ├── metrics.py
│   ├── NPR_evaluation.py
│   ├── rank_evaluation.py
│   ├── rank.py
│   ├── revise.py
│   ├── roberta_evaluation.py
│   ├── train_roberta_detect_RL.py
│   ├── train_roberta.py
│   └── zero_shot_transfer_evaluation.py
├── image/
│   ├── detectrl-framework.png
│   ├── detectrl-png.png
│   ├── detectrl-svg.svg
│   ├── shap_waterfall_human_attacked_cleaned.svg
│   ├── shap_waterfall_human_attacked_uncleaned.svg
│   ├── shap_waterfall_llm_cleaned.svg
│   ├── shap_waterfall_llm_input_text_cleaned.svg
│   ├── shap_waterfall_llm_uncleaned_v2.svg
│   ├── shap_waterfall_llm_uncleaned.svg
│   └── shap_waterfall_llm_uncleaned2_v2.svg
└── sh_scripts/
    ├── attacks_evaluation.sh
    ├── data_generation.sh
    ├── DIPPER.sh
    ├── domains_evaluation.sh
    ├── human_writing_evaluation.sh
    ├── llms_evaluation.sh
    ├── load_dataset.sh
    └── varying_length_evaluation.sh
```

---

## Folder Descriptions

- **Analysis/**: Contains scripts for statistical analysis and visualization of benchmark data.
  - *benchmark_statistics/*: Calculates metrics such as lexical diversity, n-grams, readability, and text length.
  - *heatmap/*: Visualizes cross-attack, domain, and model heatmaps.
  - *leaderboard/*: Radar visualization for evaluation.
  - *text_length_impact/*: Analyzes the impact of text length on training and test times.

- **Benchmark/**: Collection and organization of benchmark datasets.
  - *Benchmark_Data/*: Contains various datasets and attack scenarios grouped by the tasks mentioned in the DetectRL paper.
  - *Original_Dataset/*: Original data and scripts for loading and preprocessing (e.g., arXiv, Writing Prompts, XSum, Yelp).

- **Tasks/**: Folder for tasks and experiments (e.g., Task1).

- **Data_Generation/**: Scripts for data generation and mixing.
  - *Archive_chromedriver_win32/*: Archived as translation has been shifted to using Google Workspace API.
  - *Archive_TextAttack/*: Archived as TextAttack is used as a package now.

- **Detectors/**: Implementation and evaluation of various detectors for generated texts.
  - Contains scripts for different methods (DetectGPT, Binoculars, Entropy, Roberta, etc.), training and evaluation scripts, and metrics.

- **image/**: Model explanation visualizations (shapley values) and framework graphics.

- **sh_scripts/**: Shell scripts for automating evaluation and data generation processes from the original DetectRL dataset. Seem to have several inconsistencies and errors.

---

## File Descriptions (Selection)

- `README_DetectRL.md`: Original Readme file from DetectRL.
- `statistics_lexical_diversity.py`: Calculates the lexical diversity of texts.
- `statistics_n_gram.py`: Analyzes n-gram distributions.
- `statistics_readability.py`: Assesses the readability of texts.
- `statistics_text_length.py`: Measures the length of texts.
- `heatmap_cross_attacks.py`, `heatmap_cross_domains.py`, `heatmap_cross_models.py`: Create heatmaps for various comparison dimensions.
- `radar.py`: Visualizes benchmark results as a radar chart.
- `training_time_test_time_text_length_impact.py`: Investigates the impact of text length on training and test times.
- `get_arxiv_data.py`, `get_writing_prompt_data.py`, `get_xsum_data.py`, `get_yelp_review_data.py`: Load and preprocess original datasets.
- `benchmark construction.py`, `data_generation.py`, `data_mixing.py`: Generation and mixing of benchmark data.
- `Chat_API.py`: Interface to chat APIs.
- `DIPPER.py`: Implementation of the DIPPER detector.
- `binoculars_detector.py`, `DetectGPT.py`, `entropy.py`, `Fast_DetectGPT.py`, `rank.py`, `loss.py`: Various detectors and utility functions.
- `train_roberta.py`, `train_roberta_detect_RL.py`: Training Roberta models for text detection.
- `metrics.py`: Calculation of evaluation metrics.
- `evaluate_roberta_detect_RL_generalisation_results.ipynb`: Jupyter notebook for generalization analysis.
- `detectrl-framework.png`, `detectrl-svg.svg`: Framework visualizations.
- `attacks_evaluation.sh`, `data_generation.sh`, `domains_evaluation.sh`, etc.: Automation scripts for various evaluation and generation tasks.
