# Contamination in Generated Text Detection Benchmarks

## Description
This repository contains code and instructions to reproduce the experiments and results presented in the paper:
_**Sure! Here’s a short and concise title for your paper: “Contamination in Generated Text Detection Benchmarks”**_

The aim is to provide an understanding of the flaws of contamination (such as
redundancies and typical phrases of generative chat models that are not directly
relevant to the task at hand) especially in the GTD benchmark DetectRL. 

## Structure
The main folders have an individual README.md file describing the content in more detail.
The high-level structure of the repository is as follows: 
- `cleaned_DetectRL_files/`  
  Contains cleaned DetectRL dataset (Parquet files) for the four domains. Besides the final version, a subfolder is 
  provided with the intermediate results after cleaning using regular expressions.

- `DetectRL/`  
  Copy of the DetectRL repository with minor changes to make it work. Not all files are used in the experiments 
  (e.g. shell scripts, most of the zero-shot detectors, and the data generation process).
  Therefore, some files might still not be executable out of the box. The original repository can be found [here](https://github.com/junchaoIU/DetectRL).

- `notebooks/`  
  Jupyter Notebooks for analyzing and cleansing the contamination. In addition, RoBERTa models are trained and 
 evaluated to identify the effect of contamination on trained detectors using SHAP and adversarial attacks.

- `results/`  
  Results and intermediate data from experiments, especially results of the trained models.

- `src/`  
  Python source code and configuration files for the project.

## Installation
- optional: create a venv to install poetry and to later install this package within the venv using poetry:
    ```
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip setuptools wheel
  pip install poetry
  ```

- Install all dependencies with groups using poetry from the pyproject.toml file: 
```poetry install --no-root --all-groups```

To execute the notebooks, that have call the OpenAI API, you need to set the environment variable `OPENAI_API_KEY`. This
can be done in the `src/.env` file (modify the `src/.env copy` for it). For the paraphrase-back-translation a Google Workspace is used 
(set `GOOGLE_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`). The `HF_HOME` to modify the huggingface locally stores the data.

## Support & Contributing
Please create a GitHub issue if you have any questions or run into problems. Feel free to create a pull request if you 
want to contribute, especially if you found some errors within the code, fixed some class within the DetectRL subfolder,
or re-executed parts of the DetectRL benchmark on the cleaned dataset.
Despite the quality checks and multi-stage cleaning process, frequent artifacts or rejections by the LLM may still occur,
as it was not possible for us to manually review all of the over 50,000 samples. If you find further frequent artifacts,
please let us know via an issue or pull request.

## Authors and acknowledgment
The paper was written in collaboration with and under the supervision of [Christian Riess](https://www.cs1.tf.fau.de/christian-riess/).

## ✏️ Citation

If you are using the data, please cite our paper as well as the original DetectRL publication:
```
@article{wu2024detectrl,
  title={DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios},
  author={Wu, Junchao and Zhan, Runzhe and Wong, Derek F and Yang, Shu and Yang, Xinyi and Yuan, Yulin and Chao, Lidia S},
  journal={arXiv preprint arXiv:2410.23746},
  year={2024}
}
```

