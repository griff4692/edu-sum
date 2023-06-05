# BRIO: Bringing Order to Abstractive Summarization

This repo contains *slightly modified* code from the ACL paper [BRIO: Bringing Order to Abstractive Summarization](https://arxiv.org/abs/2203.16804).

This code is borrowed from the original [GitHub Repo](https://github.com/yixinL7/BRIO).

## How to Install

**Create a separate conda environment to run BRIO code to avoid conflicts.** 

- `python3.8`
- `conda create --name env --file spec-file.txt`
- Further steps
    - install additional libraries (after activating the conda env) `pip install -r requirements.txt`
    - `compare_mt` -> https://github.com/neulab/compare-mt
        ```console
        git clone https://github.com/neulab/compare-mt.git
        cd ./compare-mt
        pip install -r requirements.txt
        python setup.py install
        ```

- The BRIO authors provide the evaluation script in cal_rouge.py. If you are going to use Perl ROUGE package, please change line 13 into the path of your perl ROUGE package.

    ```_ROUGE_PATH = '/YOUR-ABSOLUTE-PATH/ROUGE-RELEASE-1.5.5/'```
- Download [Stanford CoreNLP](http://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip) and place `stanford-corenlp-3.8.0.jar` in home directory `~`.