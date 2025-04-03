#!/usr/bin/env python3
"""
This script shows various approaches for Morph Etymology prediction.
It loads annotated test sentences, computes statistics, removes etymology targets,
and then predicts etymology using various models:
  - DummyModel: Always predicts ["ces"] for each alphabetic morph. (Used as baseline.)
  - MostFrequentOriginModel: Predicts the most frequent target if the morph was in training data, otherwise "ces" or another fallback
  - MorphDictModel: Uses root and affix dictionaries.
  - WordDictModel: Uses morphological analysis to get word-level lemma and look this lemma in Etymological dictionary.
  - MorphClassifier: A learnable model (SVM, MLP, or LogisticRegression) plus optional embeddings.

For each enabled model, the script prints the F-score, accuracy, and 
the relative error reduction compared to the dummy model baseline.
"""

import argparse

from utils import (
    load_annotations,
    evaluate,
    relative_error_reduction,
    remove_targets,
    write_morph_statistics,
    count_sentences_words_morphs
)
from baselines import (
    DummyModel,
    MostFrequentOriginModel,
    MorphDictModel,
    WordDictModel
)
from morph_classifier import MorphClassifier  
from model import Model

def parse_args():
    """
    Parses command-line arguments and returns them as an argparse Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate various Morph Etymology models on annotated data."
    )
    # File paths for training, dev, and dictionaries
    parser.add_argument("--train_file", type=str, default="annotations/train.tsv",
                        help="Path to the training file (default: annotations/train.tsv)")
    parser.add_argument("--target_file", type=str, default="annotations/dev.tsv",
                        help="Path to the dev/test file (default: annotations/dev.tsv)")
    parser.add_argument("--root_etym_file", type=str, default="additional_data/roots_etymology.tsv",
                        help="Path to the root etymology file (default: additional_data/roots_etymology.tsv)")
    parser.add_argument("--word_etym_file", type=str, default="additional_data/czetyl_max.tsv",
                        help="Path to the word-level etymology file (default: additional_data/czetyl_max.tsv)")
    parser.add_argument("--affixes_file", type=str, default="additional_data/affixes_etymology.tsv",
                        help="Path to the affixes file (default: additional_data/affixes_etymology.tsv)")

    # File paths for stats
    parser.add_argument("--stats_lang_dev", type=str, default="languages_dev_stats.tsv",
                        help="File to store language stats for dev (default: languages_dev_stats.tsv)")
    parser.add_argument("--stats_morphs_dev", type=str, default="morphs_dev_stats.tsv",
                        help="File to store morph stats for dev (default: morphs_dev_stats.tsv)")
    parser.add_argument("--stats_lang_train", type=str, default="languages_train_stats.tsv",
                        help="File to store language stats for train (default: languages_train_stats.tsv)")
    parser.add_argument("--stats_morphs_train", type=str, default="morphs_train_stats.tsv",
                        help="File to store morph stats for train (default: morphs_train_stats.tsv)")

    # Model toggles
    parser.add_argument("--enable_dummy", action="store_true",
                        help="Print results for the dummy model. (Always run internally for baseline; prints only if True.)")
    parser.add_argument("--enable_mfo", action="store_true",
                        help="Enable MostFrequentOriginModel evaluation.")
    parser.add_argument("--enable_morph_dict", action="store_true",
                        help="Enable MorphDictModel evaluation.")
    parser.add_argument("--enable_word_dict", action="store_true",
                        help="Enable WordDictModel evaluation.")
    parser.add_argument("--enable_morph_classifier", action="store_true",
                        help="Enable MorphClassifier evaluation.")
    parser.add_argument("--enable_all", action="store_true",
                        help="Enable all baseline models plus the MorphClassifier (unless toggles are overridden).")

    # Morph Classifier config
    parser.add_argument("--model_name", type=str, default=None,
                        help="Name for the MorphClassifier model (default: None)")
    parser.add_argument("--classifier_type", type=str, default="mlp",
                        choices=["svm", "mlp", "lr"],
                        help="Classifier type: 'svm', 'mlp', or 'lr' (default: mlp).")
    parser.add_argument("--mlp_hidden_size", type=int, default=100,
                        help="Hidden layer size for MLP classifier (default: 100).")
    parser.add_argument("--random_state", type=int, default=34867991,
                        help="Random seed for the MorphClassifier (default: 34867991).")
    parser.add_argument("--use_word_embedding", action="store_true",
                        help="Use a word embedding feature in the MorphClassifier.")
    parser.add_argument("--use_morph_embedding", action="store_true",
                        help="Use a morph embedding feature in the MorphClassifier.")
    parser.add_argument("--embedding_dimension", type=int, default=300,
                        help="Embedding dimension if using embeddings (default: 300).")

    return parser.parse_args()

def run_model(
    model: Model,
    model_name: str,
    train_data,
    target_data,
    baseline_f1: float,
    mistakes_file: str = None
) -> None:
    """
    Fits (if applicable), predicts, evaluates, and prints results for a given model.

    Args:
        model (Model): The model to run.
        model_name (str): A name/description for printing.
        train_data: The training data (list of DataSentence).
        target_data: The dev/test data with target labels.
        baseline_f1 (float): The baseline F1 score (dummy model).
        mistakes_file (str): If set, logs mistakes to this file.
    """
    try:
        print(f"----- {model_name} -----")
        model.fit(train_data)
        # Remove targets from the data to simulate unlabeled data
        dev_data = remove_targets(target_data)
        predictions = model.predict(dev_data)

        # Evaluate
        f_score, accuracy = evaluate(predictions, target_data, mistakes_file)
        improvement = relative_error_reduction(baseline_f1, f_score)

        print("Results:")
        print(f"F-score: {f_score:.3f} %, Accuracy: {accuracy:.3f} %")
        print(f"Relative Error Reduction: {improvement:.3f} %\n")

    except WordDictModel.NetworkError as net_err:
        print(f"Network error while running model '{model_name}'.\nThe following exception occured: {net_err}")
        print()
    
    except Exception as e:
        print(f"Error when running model: '{model_name}'")
        print(f"The following exception occurred:\n    {e}\n")
        print()

def main():
    args = parse_args()

    # If user wants --enable_all, set all toggles to True
    if args.enable_all:
        args.enable_dummy = True
        args.enable_mfo = True
        args.enable_morph_dict = True
        args.enable_word_dict = True
        args.enable_morph_classifier = True

    # Load dev/test data
    dev_sentences_target = load_annotations(args.target_file)
    # Load train data
    train_sentences = load_annotations(args.train_file)

    write_morph_statistics(
        dev_sentences_target, args.stats_lang_dev, args.stats_morphs_dev
    )
    sentence_count,word_count,morph_count = count_sentences_words_morphs(dev_sentences_target)
    print(f"Statistics on Dev -- Morphs: {morph_count}, Words: {word_count}, Sentences: {sentence_count}\n")

    write_morph_statistics(
        train_sentences, args.stats_lang_train, args.stats_morphs_train
    )
    sentence_count,word_count,morph_count = count_sentences_words_morphs(train_sentences)
    print(f"Statistics on Train -- Morphs: {morph_count}, Words: {word_count}, Sentences: {sentence_count}\n")

    # Always run dummy internally for baseline
    dummy_model = DummyModel()
    dev_dummy = remove_targets(dev_sentences_target)
    dummy_predictions = dummy_model.predict(dev_dummy)
    f_score_dummy, accuracy_dummy = evaluate(dummy_predictions, dev_sentences_target)
    baseline_f1 = f_score_dummy

    if args.enable_dummy:
        print("----- Dummy Model (baseline) -----")
        print(f"F-score: {f_score_dummy:.3f} %, Accuracy: {accuracy_dummy:.3f} %\n")

    if args.enable_mfo:
        mfo_model = MostFrequentOriginModel()
        run_model(mfo_model, mfo_model.name,
                  train_sentences, dev_sentences_target,
                  baseline_f1, f"mistakes_{mfo_model.name}.tsv")

    # Possibly run MorphDictModel
    if args.enable_morph_dict:
        md_model = MorphDictModel(args.root_etym_file, args.affixes_file)
        run_model(md_model, md_model.name,
                  train_sentences, dev_sentences_target,
                  baseline_f1, f"mistakes_{md_model.name}.tsv")

    # Possibly run WordDictModel
    if args.enable_word_dict:
        wd_model = WordDictModel(args.word_etym_file, args.affixes_file)
        run_model(wd_model, wd_model.name,
                  train_sentences, dev_sentences_target,
                  baseline_f1, f"mistakes_{wd_model.name}.tsv")

    # Possibly run MorphClassifier
    if args.enable_morph_classifier:
        learning_model = MorphClassifier(
            name=args.model_name,
            random_state=args.random_state,
            classifier_type=args.classifier_type,
            mlp_hidden_size=args.mlp_hidden_size,
            use_word_embedding=args.use_word_embedding,
            use_morph_embedding=args.use_morph_embedding,
            embedding_dimension=args.embedding_dimension
        )
        run_model(learning_model, learning_model.name,
                  train_sentences, dev_sentences_target,
                  baseline_f1, f"mistakes_{learning_model.name}.tsv")


if __name__ == "__main__":
    main()