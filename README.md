# Identification of Morpheme Origin

This repository is part of a Bachelor thesis on **Identification of Morpheme Origin**. Given morphologically segmented words or sentences, it aims to determine an etymology sequence (i.e., language of origin and possible intermediate languages) for each morph.

## Overview

- **Input:** Morphologically segmented text, where each word is decomposed into morphs.  
- **Task:** Assign a list of languages (e.g., `['ces']`, `['lat','ell']`) to each morph representing from which and though which languages the word was borrowed to Czech. Just `['ces']` for native words.
- **Example (Single Word):**  
  ```text
  antivirový
    anti   → ["ell"]
    vir    → ["deu","lat"]
    ov     → ["ces"]
    ý      → ["ces"]
  ```
### Source code
Directory `src/`
- **`main.py`**: The main script to run experiments.
- **`utils.py`**: Contains utility functions used across the project.
- **`baselines.py`**: Implements baseline methods for comparison.
- **`embedding_transformer.py`**: Handles the use of embeddings.
- **`model.py`**: Parent class for the models.
- **`morph_classifier.py`**: Contains the machine learning model for classification.
- **`inter_annotator.py`**: Measures inter-annotator agreement.
- **`prepare_for_annotation.py`**: Processes file of segmented text and prints it in format suitable for annotation.

## Data

Besides scripts the repository includes the following data in the directory `data/`:

- **`annotations/`**: Annotated files (train, dev, test) plus original SIGMORPHON 2022 data.
- **`etymological_data/`**: Supplementary data (CzEtyL etymological lexicon, affix/root dictionaries).
  
## How to Use

### Running the Experiment
To run the experiment with default settings, use:
```bash
make run
```
It will create virtual enviroment and will install the necessary packages.

### Calculating Inter-Annotator Agreement
To calculate agreement on the default two annotations, use:
```bash
make agreement
```

### Cleaning Up
To clean up generated files, use:
```bash
make clean
```

## Running `main.py` with Arguments

If you want to set up arguments and run `main.py` directly (i.e., outside of `make run`):

1. **Create/Update the Virtual Environment:**
```bash
make venv
```
2. **Activate the Virtual Environment:**
```bash
source MorphOriginVenv/bin/activate
```
3. **Run the script with custom arguments:**
   
Example:
```bash
python3 src/main.py --enable_all --extend_train --multi_label --mlp_hidden_size=40 --target_file=data/annotations/test.tsv
```
This enables all models (baselines + learning model) uses extended train set for the learning model using entries from etymological dictionary. Treats the target as multi labels (oposed to default predict the whole sequence at once as single class) and uses 40 neurons in hidden layer of the MLP classifier (which is used as default). All models are evaluated on file `data/annotations/test.tsv` (default would be `data/annotations/dev.tsv`).
   
## Arguments description

The primary machine learning model in this project is defined in `morph_classifier.py`. It offers a flexible pipeline, configured via command-line arguments when running with the script `main.py`:

- **Classifier Type**  
  Choose between:
  - **SVM** (`--classifier_type svm`), adjustable by the C parameter `--svm_c` and kernel used `--svm_kernel`.  
  - **MLP** (`--classifier_type mlp`), optionally as an ensemble of MLPs via `--mlp_ensemble_size`. One hidden layer of size  `--mlp_hidden_size`, default to 30. Additionaly regularizaton can be controled using `--mlp_alpha` and maximum number of iterations (epochs) to run with `--mlp_max_iter`
  - **Logistic Regression** (`--classifier_type lr`).  

- **Feature Extraction**  
  - **Character n-grams** Used by default, disable by `--disable_char_ngrams`, with a customizable range (`--char_ngram_min`, `--char_ngram_max`).  
  - **Morph Type** (Root/Derivational affix/Inflectional affix) On by default use `-disable_morph_type` to disable.
  - **Morph Position** (Root/Prefix/Interfix/Suffix) On by default use `--disable_morph_position` to disable.
  - **Embeddings** (fastText) for words and/or morphs (`--use_word_embedding`, `--use_morph_embedding`).  
  When using embeddings for the first time, they are automatically downloaded from the [internet](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cs.300.bin.gz).
  - **Vowel Start End** Binary indicators for whether a morph starts/ends with a vowel. Disable by   `-disable_vowels`

- **Multi-Label vs Single-Label**  
  - By default, each etymology sequence is treated as a single label (e.g., "lat,ell").  
  - With `--multi_label`, the sequence is split into separate labels (["lat", "ell"]) and for each language individually decides if it will be in the target or not.  The model uses a `MultiLabelBinarizer` + a `OneVsRestClassifier`.  
  - **Fallback Single-Label**: If multi-label prediction fails (returns an empty set), you can optionally train a single-label pipeline in parallel by setting `--fallback_single_label`; the model then falls back to that pipeline if the multi-label pipeline yields no labels.

- **Extending train set and Filtering Low-Frequency Labels**  
  - To extend the train set using the roots and affixes dictionaries use `--extend_train`.   It increases the training set by adding single-morph sentences from the provided roots and affixes dictionaries. It significantly expands the amount of training data, the quality may not match the regular annotated set.
  - Use `--min_seq_occurrence` to discard rare etymology sequences below a certain threshold of occurrences, cleaning up the training data. Example `--min_seq_occurrence=3` filters all sequences with occurences lower than 3.

- **Case Normalization**  
  - Control whether text is lowercased. Default is to lower the case of all text. Use `--keep_case` to disable this.

- **Early Stopping**  
  -  If enabled sets aside 10 % of training data and use it for evaluation. Stop training when the model is not improving on this evaluation set. Use `--early_stopping` to enable this.

- **Saving & Loading**  
  - You can save the trained model to disk with `--save` which saves the trained model to file `model.name + '.pkl'`. To specify path where to save model use `--save_model_path`
  - You can load the trained model to disk with `--load` which loads the trained model from file `model.name + '.pkl'`. To specify path from which to load the model use `--load_model_path`

- **Generating Additional Information**
Files are by default generated to directory `outputs\`. The output directory can be specified using `--outputs_dir`
  - **Stats Files**  
   To specify paths to TSV files that store language- and morph- statistics for both train and test sets:
    - `--stats_lang_test`: Where to store language counts on the test set (default: `languages_test_stats.tsv`)
    - `--stats_morphs_test`: Where to store morph counts on the test set (default: `morphs_test_stats.tsv`)
    - `--stats_lang_train`: Where to store language counts on the train set (default: `languages_train_stats.tsv`)
    - `--stats_morphs_train`: Where to store morph counts on the train set (default: `morphs_train_stats.tsv`)

    If you enable `--print_stats`, the script writes these files after loading the datasets.

  - **Mistakes File**  
    You can specify `--mistakes_file <filename>` to record all morphs for which the learning model fails to perfectly predict the target.
    If `--print_mistakes` is set, the script will actually produce this mistakes file for all models (including baseline models). Otherwise, mistakes are not logged. With exception of learning `MorphClassifier` model if the `--mistakes_file` is specified.

  - **Predictions File**
  By default, the predictions made by the learning model are saved to `outputs/predictions.tsv`.  
  You can change the path using the `--predictions_file` argument.  
  If `--predictions_file` is set to an empty string or `None`, no predictions will be saved.

- **Training and Target Files**

  The data used for training and evaluation can be specified via arguments:

  - `--train_file`: Path to the training file.  
    Default: `data/annotations/train.tsv`

  - `--target_file`: Path to the target (evaluation) file.  
    Default: `data/annotations/dev.tsv`


To see all available flags and parameters, run:
```bash
python3 src/main.py --help
```

## Baseline Models

We implement four baseline models (see `src/baselines.py`), each offering a different approach:

1. **DummyModel**  
   Always predicts `["ces"]` for any alphabetic morph. Useful as a minimal baseline.
2. **MostFrequentOriginModel**  
   Remembers the most frequent etymology sequence for each morph (from training); defaults to `["ces"]` if unseen.
3. **MorphDictModel**  
   Uses root+affix dictionaries to label morphemes if found, falling back to `["ces"]`.
4. **WordDictModel**  
   Analyzes the word to get its lemma (via MorphoDiTa) and assigns the word-level etymology from a dictionary to root of the word, plus also uses affixes dictionaries.

## Evaluation

Our primary evaluation computes an **F1 score** for each morph’s predicted vs. target etymology sets, then averages them (over all morphs). Additionally:
  
- Also report variant where we split evaluation by categories (e.g. “native” vs. “borrowed”) and return separate F1 scores
- Report realtive error reduction of the standart F-score from the dummy baseline (always predict Czech)
- Report F-score grouped by morph text. Group all morphs with the same text and calculate the average F-score for each group and the average over all groups.
