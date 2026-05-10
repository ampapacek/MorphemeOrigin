# Identification of Morph Origin

This repository focuses on identifying the origin of individual morphs in morphologically segmented Czech words and sentences. For each morph, the goal is to predict an etymology sequence: the source language and, when relevant, intermediate borrowing languages.

## Contents

- [Task Overview](#task-overview)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Running the Project](#running-the-project)
- [Command-Line Arguments](#command-line-arguments)
- [LLM Prediction Pipeline](#llm-prediction-pipeline)
- [Baseline Models](#baseline-models)
- [Evaluation](#evaluation)
- [Project Context](#project-context)

## Task Overview

- **Input:** Morphologically segmented text, where each word is decomposed into morphs.
- **Task:** Assign a list of languages (for example `["ces"]` or `["lat", "ell"]`) to each morph to capture its origin and possible borrowing path into Czech.
- **Native morphs:** Morphs of Czech origin are labeled simply as `["ces"]`.

Example:

```text
antivirovy
  anti   -> ["ell"]
  vir    -> ["deu", "lat"]
  ov     -> ["ces"]
  y      -> ["ces"]
```

## Repository Structure

### Main directories

- [`src/`](./src): training, evaluation, baselines, and LLM prediction code
- [`data/`](./data): annotated datasets and supporting etymological resources
- [`docs/`](./docs): additional documentation and notes
- [`legal_corpus_analysis/`](./legal_corpus_analysis): related corpus analysis utilities

### Important files

- [`src/main.py`](./src/main.py): main entry point for experiments
- [`src/morph_classifier.py`](./src/morph_classifier.py): primary machine learning model
- [`src/baselines.py`](./src/baselines.py): baseline methods
- [`src/data_transformers.py`](./src/data_transformers.py): embedding-related processing
- [`src/inter_annotator.py`](./src/inter_annotator.py): inter-annotator agreement
- [`src/prepare_for_annotation.py`](./src/prepare_for_annotation.py): formatting segmented text for annotation
- [`src/llm_predict.py`](./src/llm_predict.py): LLM-based morpheme-origin prediction
- [`src/run_llm_sweep.py`](./src/run_llm_sweep.py): multi-run LLM sweep utility
- [`src/evaluate_predictions.py`](./src/evaluate_predictions.py): evaluates saved predictions against gold data
- [`prompt_for_ai.txt`](./prompt_for_ai.txt): prompt template for the LLM pipeline
- [`Makefile`](./Makefile): common commands for setup and running experiments
- [`run_win.ps1`](./run_win.ps1): PowerShell entry point for Windows

## Data

The [`data/`](./data) directory contains both annotations and supplementary resources:

- [`data/annotations/`](./data/annotations): train, development, and test annotations, plus original SIGMORPHON 2022 data
- [`data/etymological_data/`](./data/etymological_data): supporting etymological lexicons and affix/root dictionaries

## Running the Project

### Linux and other Unix-like systems

Run the default experiment with:

```bash
make run
```

This creates a virtual environment, installs dependencies, and runs the project with default settings.

To calculate inter-annotator agreement on the default annotation files:

```bash
make agreement
```

To remove generated files:

```bash
make clean
```

### Windows PowerShell

Run the default experiment with:

```powershell
powershell -ExecutionPolicy Bypass -File run_win.ps1
```

This will:

- create a virtual environment named `MorphOriginVenv` if it does not already exist
- install packages from `requirements_win.txt`
- run the default experiment with `--enable_all`

### Running `main.py` directly

If you want to configure arguments manually instead of using `make run`:

1. Create or update the virtual environment:

```bash
make venv
```

2. Activate it:

```bash
source MorphOriginVenv/bin/activate
```

3. Run the script with custom arguments:

```bash
python3 src/main.py --enable_all --extend_train --multi_label --mlp_hidden_size=40 --target_file=data/annotations/test.tsv
```

This example enables all models, extends training data using dictionary-derived examples, treats targets as multi-label sequences, uses an MLP with 40 hidden units, and evaluates on `data/annotations/test.tsv`.

## Command-Line Arguments

The primary machine learning model is implemented in [`src/morph_classifier.py`](./src/morph_classifier.py) and configured through [`src/main.py`](./src/main.py).

- **Classifier type**
  - `--classifier_type svm` with `--svm_c` and `--svm_kernel`
  - `--classifier_type mlp` with options such as `--mlp_hidden_size`, `--mlp_alpha`, `--mlp_max_iter`, and `--mlp_ensemble_size`
  - `--classifier_type lr`

- **Feature extraction**
  - Character n-grams are enabled by default and can be disabled with `--disable_char_ngrams`
  - Morph type is enabled by default and can be disabled with `--disable_morph_type`
  - Morph position is enabled by default and can be disabled with `--disable_morph_position`
  - fastText embeddings can be enabled with `--use_word_embedding` and `--use_morph_embedding`
  - Vowel boundary features can be disabled with `--disable_vowels`

- **Prediction setup**
  - By default, etymology sequences are treated as single labels
  - `--multi_label` predicts individual languages in the sequence separately

- **Training data controls**
  - `--extend_train` augments training data with entries from root and affix dictionaries
  - `--min_seq_occurrence` filters out low-frequency etymology sequences

- **Preprocessing and training**
  - `--keep_case` disables lowercasing
  - `--early_stopping` enables early stopping with a held-out portion of training data

- **Saving and loading**
  - `--save` and `--save_model_path` save a trained model
  - `--load` and `--load_model_path` load a trained model

- **Output files**
  - `--outputs_dir` sets the output directory
  - `--print_stats` writes language and morph statistics
  - `--mistakes_file` and `--print_mistakes` control error logging
  - `--predictions_file` sets where predictions are written

- **Input files**
  - `--train_file` sets the training file
  - `--target_file` sets the evaluation file

To see the full list of options:

```bash
python3 src/main.py --help
```

## LLM Prediction Pipeline

The repository also includes an LLM-based prediction workflow:

- [`src/llm_predict.py`](./src/llm_predict.py): runs one model/configuration
- [`src/run_llm_sweep.py`](./src/run_llm_sweep.py): runs multiple model and context-size combinations

### Single LLM run

```bash
python3 src/llm_predict.py \
  --model openai/gpt-5.2 \
  --input_file data/annotations/dev.tsv \
  --train_file data/annotations/train.tsv \
  --train_context_morphs 2000 \
  --prompt_file prompt_for_ai.txt \
  --log_file outputs/dev_gpt5_2_trainm_2000.log \
  --eval_results_file outputs/dev_gpt5_2_results.tsv
```

Notes:

- The default API provider is OpenRouter (`--api_provider openrouter`)
- API keys are resolved first from `--api_key_env` (default `LLM_API_KEY`), then from provider-specific environment variables
- Existing annotated files such as `data/annotations/dev.tsv` and `data/annotations/test.tsv` can be used directly
- Evaluation runs automatically unless `--skip_eval` is set
- Predictions and mistake files are written automatically unless custom paths are provided
- `--batch_parallelism` can be used to parallelize batches within one model run

To use the OpenAI API directly:

```bash
python3 src/llm_predict.py \
  --api_provider openai \
  --model gpt-5.2 \
  --input_file data/annotations/dev.tsv \
  --train_file data/annotations/train.tsv \
  --train_context_morphs 2000 \
  --prompt_file prompt_for_ai.txt
```

To use another OpenAI-compatible endpoint:

```bash
python3 src/llm_predict.py \
  --api_provider openai \
  --api_endpoint https://your-endpoint.example/v1/chat/completions \
  --model gpt-5.2 \
  --input_file data/annotations/dev.tsv \
  --train_file data/annotations/train.tsv \
  --train_context_morphs 2000 \
  --prompt_file prompt_for_ai.txt
```

### Sweep multiple LLM configurations

```bash
python3 src/run_llm_sweep.py \
  --models openai/gpt-5-mini,openai/gpt-5.2 \
  --train_context_sizes 0,50,200 \
  --parallel 2 \
  --summary_file outputs/llm_sweep_results.tsv \
  -- \
  --input_file data/annotations/test.tsv \
  --train_file data/annotations/train.tsv \
  --prompt_file prompt_for_ai.txt
```

Example with parallel batches inside each run:

```bash
python3 src/run_llm_sweep.py \
  --models openai/gpt-5-mini,openai/gpt-5.2 \
  --train_context_sizes 200 \
  --parallel 2 \
  --summary_file outputs/llm_sweep_results.tsv \
  -- \
  --input_file data/annotations/test.tsv \
  --train_file data/annotations/train.tsv \
  --batch_parallelism 5 \
  --prompt_file prompt_for_ai.txt
```

Notes:

- Arguments before `--` configure `run_llm_sweep.py`
- Arguments after `--` are forwarded to each `llm_predict.py` run
- Logs for sweep runs are stored in `outputs/llm_sweep_logs/`
- Evaluation rows from each run are appended to the shared summary file

### Evaluate an existing prediction file

```bash
python3 src/evaluate_predictions.py \
  --pred_file outputs/llm_predictions_openai_gpt_5_2_trainm_999999_all_50.tsv \
  --gold_file data/annotations/test.tsv \
  --mistakes_file outputs/mistakes_eval_gpt5_2_test.tsv
```

The script prints standard metrics such as `f1score_instance`, `f1score_micro`, native vs. borrowed scores, grouped scores, and morph-type split scores.

## Baseline Models

The project includes four baseline models implemented in [`src/baselines.py`](./src/baselines.py):

1. **DummyModel**: always predicts `["ces"]` for alphabetic morphs
2. **MostFrequentOriginModel**: predicts the most frequent training label for a morph and falls back to `["ces"]`
3. **MorphDictModel**: uses root and affix dictionaries when available
4. **WordDictModel**: uses lemma-based word-level etymology together with affix dictionaries

## Evaluation

The main evaluation computes an F1 score between predicted and target etymology sets for each morph and then averages across morphs.

Additional reported views include:

- separate scores for categories such as native vs. borrowed
- relative error reduction against the dummy baseline
- grouped F1 by morph text

## Project Context

This repository is connected to a bachelor thesis on morph origin identification. The work was also presented in the context of LREC and the LT4HALA workshop, and the repository collects the code, data resources, and experimental utilities used for that line of work.
