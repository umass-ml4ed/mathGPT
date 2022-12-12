# MathGPT

## Setup

### Python Environemnt
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
