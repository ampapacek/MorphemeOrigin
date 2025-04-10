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
import time
import sys
import argparse

from utils import (
    load_annotations,
    evaluate,
    relative_error_reduction,
    remove_targets,
    write_morph_statistics,
    count_sentences_words_morphs,
    single_morph_sentences_from_dict,
    evaluate_combined
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
    parser.add_argument("--mlp_ensemble_size", type=int, default=1,
                        help="Number of MLP classifiers in an ensemble (default: 1).")
    parser.add_argument("--mlp_hidden_size", type=int, default=100,
                        help="Hidden layer size for MLP classifier (default: 100).")
    parser.add_argument("--svm_c", type=float, default=1.0,
                        help="C parameter for LinearSVC (default: 1.0).")
    parser.add_argument("--random_state", type=int, default=34867991,
                        help="Random seed for the MorphClassifier (default: 34867991).")
    
    parser.add_argument("--save_model_path", type=str, default="",
                        help="Path where to save the trained model. Automaticly enables loading. (default: empty => dont save model).")
    parser.add_argument("--load_model_path", type=str, default="",
                        help="Path with the trained model for loading. Automaticly enables saving. (default: empty => dont load model).")
    parser.add_argument("--save", action="store_true",
                        help="If to save the trained model. If the path is not specified, model.name + .pkl is used.")
    parser.add_argument("--load", action="store_true",
                        help="If to load the trained model from file. If the path is not specified, model.name + .pkl is used.")

    # Feature toggles
    parser.add_argument("--disable_char_ngrams", action="store_true",
                        help="Disable character n-gram features (by default they're ON).")
    parser.add_argument("--char_ngram_min", type=int, default=1,
                        help="Minimum n for character n-grams (default: 1).")
    parser.add_argument("--char_ngram_max", type=int, default=2,
                        help="Maximum n for character n-grams (default: 2).")

    parser.add_argument("--disable_morph_type", action="store_true",
                        help="Disable morph type as a one-hot encoded feature (default: it's ON).")
    parser.add_argument("--disable_morph_position", action="store_true",
                        help="Disable morph position as a one-hot encoded feature (default: it's ON).")
    parser.add_argument("--disable_vowels", action="store_true",
                        help="Disable use of vowel features. If the morph text starts and end with a vowel (default: it's ON).")

    parser.add_argument("--use_word_embedding", action="store_true",
                        help="Use a word embedding feature in the MorphClassifier (default: off).")
    parser.add_argument("--use_morph_embedding", action="store_true",
                        help="Use a morph embedding feature in the MorphClassifier (default: off).")
    parser.add_argument("--embedding_dimension", type=int, default=300,
                        help="Embedding dimension if using embeddings (default: 300).")
    parser.add_argument("--fasttext_model_path", type=str, default="cc.cs.300.bin",
                        help="Path to the fastText .bin model (default: cc.cs.300.bin).")

    parser.add_argument("--multi_label", action="store_true",
                        help="Enable multi_label classification. (default: False).")
    parser.add_argument("--extend_train", action="store_true",
                        help="Use root and affixes dictionaries as extension to training set.")
    parser.add_argument("--min_seq_occurrence", type=int, default=2,
                        help="Minimal number of occurrences for an etymological sequence to keep that morph in the train set (default: 2).")
    parser.add_argument("--keep_case", action="store_true",
                        help="If True, keep (upper/lower) case of all morphs/words. If False convert all text to lowercase (default: False).")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Disables printing of additional information like timing.")

    return parser.parse_args()

def run_model(
    model: Model,
    model_name: str,
    train_data,
    target_data,
    baseline_f1: float,
    mistakes_file: str = None,
    verbose:bool = True,
    load_model_path = ""
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
        load_model_path (str): Path to the trained and saved model. If None or empty train the model from data.
    """
    start_time = time.time()
    print(f"----- {model_name} -----")
    try:
        if load_model_path:
            try:
                model.load(load_model_path)
            except Exception as error:
                print("Error when trying to load model from ", load_model_path)
                print("The following error occured:", error)
                print("Save the model firts or call without load_model_path to train the model instead")
                raise error
        else:
            model.fit(train_data)
        # Remove targets from the data to simulate unlabeled data
        dev_data = remove_targets(target_data)
        if verbose:
            print(f"Computing predictions on test data...")
        predictions = model.predict(dev_data)

        # Evaluate
        evaluation_results = evaluate_combined(predictions, target_data, standard_eval=True,native_borrowed_eval=True,group_by_text_eval=True, file_mistakes=mistakes_file)
        f_score = evaluation_results['standard_fscore']
        f_score_native = evaluation_results['native_f1']
        f_score_borrowed = evaluation_results['borrowed_f1']
        f_score_grouped = evaluation_results['grouped_fscore']
        improvement = relative_error_reduction(baseline_f1, f_score)
        if verbose:
            print(f"Predictions computed and evaluated. Total time {time.time()-start_time:.3f} s")
        if verbose:
            print()
            print("Results:")
        print(f"Standard macro F-score: {f_score:.3f} %")
        print(f"F-score: native: {f_score_native:.3f} %, borrowed: {f_score_borrowed:.3f} %, grouped by unique morphs: {f_score_grouped:.3f} %")
        print(f"Relative Error Reduction over dummy baseline on standard F-score: {improvement:.3f} %\n")
    except WordDictModel.NetworkError as net_err:
        print(f"Network error while running model '{model_name}'.\nThe following exception occured: {net_err}")
        print()
        raise net_err
    
    except Exception as e:
        print(f"Error when running model: '{model_name}'")
        print(f"The following exception occurred:\n    {e}\n")
        print()
        raise e

def main():
    args = parse_args()
    args.enable_morph_classifier = True
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

    write_morph_statistics(dev_sentences_target, args.stats_lang_dev, args.stats_morphs_dev)
    sentence_count,word_count,morph_count = count_sentences_words_morphs(dev_sentences_target)
    if not args.quiet:
        print(f"Statistics on Dev -- Morphs: {morph_count}, Words: {word_count}, Sentences: {sentence_count}\n")

    write_morph_statistics(train_sentences, args.stats_lang_train, args.stats_morphs_train)
    sentence_count,word_count,morph_count = count_sentences_words_morphs(train_sentences)
    if not args.quiet:
        print(f"Statistics on Train -- Morphs: {morph_count}, Words: {word_count}, Sentences: {sentence_count}\n")

    # Always run dummy internally for baseline
    dummy_model = DummyModel()
    dev_dummy = remove_targets(dev_sentences_target)
    dummy_predictions = dummy_model.predict(dev_dummy)

    if args.enable_dummy:
        evaluation_results = evaluate_combined(dummy_predictions, dev_sentences_target, standard_eval=True,native_borrowed_eval=True,group_by_text_eval=True)
        f_score_dummy = evaluation_results['standard_fscore']
        f_score_native = evaluation_results['native_f1']
        f_score_borrowed = evaluation_results['borrowed_f1']
        f_score_grouped = evaluation_results['grouped_fscore']
        print("----- Dummy Model (baseline) -----")
        print(f"F-score: {f_score_dummy:.3f} %\n")
        print(f"Grouped by unique morphs: {f_score_grouped:.3f} %")
        print(f"F-score Native {f_score_native:.3f} %, Borrowed: {f_score_borrowed:.3f} %\n")
    else:
        evaluation_results = evaluate_combined(dummy_predictions, dev_sentences_target, standard_eval=True,native_borrowed_eval=False,group_by_text_eval=False)
        f_score_dummy = evaluation_results['standard_fscore']
    baseline_f1 = f_score_dummy

    if args.enable_mfo:
        mfo_model = MostFrequentOriginModel()
        run_model(mfo_model, mfo_model.name,
                  train_sentences, dev_sentences_target,
                  baseline_f1, f"mistakes_{mfo_model.name}.tsv",verbose=(not args.quiet))

    # Possibly run MorphDictModel
    if args.enable_morph_dict:
        md_model = MorphDictModel(args.root_etym_file, args.affixes_file)
        run_model(md_model, md_model.name,
                  train_sentences, dev_sentences_target,
                  baseline_f1, f"mistakes_{md_model.name}.tsv",verbose=(not args.quiet))

    # Possibly run WordDictModel
    if args.enable_word_dict:
        wd_model = WordDictModel(args.word_etym_file, args.affixes_file)
        run_model(wd_model, wd_model.name,
                  train_sentences, dev_sentences_target,
                  baseline_f1, f"mistakes_{wd_model.name}.tsv",verbose=(not args.quiet))

    # Possibly run MorphClassifier
    if args.enable_morph_classifier:
        if args.extend_train:
            train_sentences += single_morph_sentences_from_dict(args.root_etym_file)
            train_sentences += single_morph_sentences_from_dict(args.affixes_file)
            write_morph_statistics(train_sentences,languages_file='lang_extended.tsv')
            sentence_count,word_count,morph_count = count_sentences_words_morphs(train_sentences)
            print(f"Statistics on extended train -- Morphs: {morph_count}, Words: {word_count}, Sentences: {sentence_count}\n")
        else:
            # if the train is not extended there is no need  to remove etym sequences with low frequency
            if args.min_seq_occurrence == 2:
                # if the argument was kept on default set it to 1 (keep all sequences)
                args.min_seq_occurrence = 1

        char_ngram_range = (args.char_ngram_min, args.char_ngram_max)

        learning_model = MorphClassifier(
            name=args.model_name,
            random_state=args.random_state,
            classifier_type=args.classifier_type,
            mlp_ensemble_size=args.mlp_ensemble_size,
            mlp_hidden_size=args.mlp_hidden_size,
            svm_c=args.svm_c,

            use_char_ngrams=(not args.disable_char_ngrams),
            char_ngram_range=char_ngram_range,
            use_morph_type=(not args.disable_morph_type),
            use_morph_position=(not args.disable_morph_position),
            use_vowel_start_end_features=(not args.disable_vowels),

            fasttext_model_path=args.fasttext_model_path,
            use_morph_embedding= args.use_morph_embedding,
            use_word_embedding= args.use_word_embedding,
            embedding_dimension=args.embedding_dimension,

            lower_case=(not args.keep_case),
            verbose=(not args.quiet),
            multi_label=args.multi_label,
            min_label_freq=args.min_seq_occurrence
        )
        if args.load:
            args.load_model_path = learning_model.name + '.pkl'
        try:
            run_model(learning_model, learning_model.name,
                    train_sentences, dev_sentences_target,
                    baseline_f1, f"mistakes_{learning_model.name}.tsv",verbose=(not args.quiet),load_model_path=args.load_model_path)
            if args.save_model_path:
                learning_model.save(args.save_model_path)
            elif args.save:
                learning_model.save(learning_model.name + '.pkl')
        # TODO: Make the excpetion handling better        
        except Exception as e:
            print("Terminating program after an error.")



if __name__ == "__main__":
    main()