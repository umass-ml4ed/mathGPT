# MathGPT

Code for the paper [Tree-Based Representation and Generation of Natural and Mathematical Language
](https://aclanthology.org/2023.acl-long.205/)

If you end up using this code in your research, please cite us like:

```
@inproceedings{scarlatos-lan-2023-tree,
    title = "Tree-Based Representation and Generation of Natural and Mathematical Language",
    author = "Scarlatos, Alexander and Lan, Andrew",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.205",
    pages = "3714--3730",
}
```

## Setup

### Python Environment
Ensure Python3 is installed (this code was tested on v3.9.1).

Create virtual environment
```
python3 -m venv <env_name>
source <env_name>/bin/activate
```

Install libraries
```
python3 -m pip install -r requirements.txt
```

Make TangentCFT available in Python path
```
export PYTHONPATH=..:../TangentCFT/
```

### External Dependencies

The following need to be installed for full functionality. LaTeXML is only required to run pre-processing.

#### TangentCFT
Download TangentCFT to the folder *above* the root of this repo: https://github.com/BehroozMansouri/TangentCFT/tree/2b189dff67d6d3b15e323358921bdbca779bfcd9

Note that some small fixes were made to TangentCFT, so the file `semantic_symbol.py` is copied from there with some changes and is overloaded automatically.

#### LaTeXML
https://math.nist.gov/~BMiller/LaTeXML/get.html

#### NLG-Eval
https://github.com/Maluuba/nlg-eval

Known installation issue: https://github.com/Maluuba/nlg-eval/issues/61

### Data

Here are links to the datasets required for the following tasks. For each, ensure the dataset's root folder is *above* the root of this repo.

- Pre-Training: https://ntcir-math.nii.ac.jp/data/ 
    - just the Wikipedia Corpus
- Headline Generation: https://github.com/yuankepku/MathSum
- Equation Extraction: https://ai.tencent.com/ailab/nlp/dialogue/#datasets
    - The dataset will have to be translated to English; Google Translate is acceptable.
    - See `pre_process.py->process_mwp_data` for desired final data format.
- Student Action Prediction: https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=660
- GSM8K: https://github.com/openai/grade-school-math
- MATH: https://github.com/hendrycks/math

The datasets for the following tasks cannot be released publicly:
- Answer Scoring
- Feedback Generation

## Run

The starting point for all code is `__main__.py`, and you can see a list of command line options by running:

```
python3 __main__.py --help
```

Default values can be found in the `TrainOptions` constructor in `utils.py`.

Here is the typical workflow to replicate our experiments:
- Pre-process the Wikipedia dataset (this step also constructs the vocabulary, which is needed for all following steps)
- Pre-train a MathGPT model
- Pre-process the downstream dataset
- Run cross-validation on the downstream dataset, which for each fold:
    - Fine-tunes the pre-trained MathGPT model on the downstream dataset
    - Runs evaluation on the downstream dataset's test set
