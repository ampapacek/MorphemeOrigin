# Morpheme Origin Project

This project is part of a Bachelor thesis on morpheme origin prediction. The task is to determine an etymology sequence of languages for each morph in all words in given sentence(s).

## Overview

- **`main.py`**: The main script to run experiments.
- **`utils.py`**: Contains utility functions used across the project.
- **`baselines.py`**: Implements baseline methods for comparison.
- **`embedding_transformer.py`**: Handles the use of embeddings.
- **`model.py`**: Parent class for the models.
- **`morph_classifier.py`**: Contains the machine learning model for classification.
- **`inter_annotator.py`**: Measures inter-annotator agreement.

## Data

Besides scripts the repository includes the following data:

- **`annotations/`**: Annotated files (train, dev, test) plus original SIGMORPHON 2022 data.
- **`additional_data/`**: Supplementary data (CzEtyL etymological lexicon, affix/root dictionaries).
  
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

## Running main.py with Arguments
The main.py script supports various arguments to customize the experiment. To see all available options, use:
```bash
python3 main.py --help
```
