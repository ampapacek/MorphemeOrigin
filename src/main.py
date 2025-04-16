#!/usr/bin/env python3
"""
This script shows various approaches for Morph Etymology prediction.
It loads annotated test sentences, computes statistics, removes etymology targets,
and then predicts etymology using various models:
  - DummyModel: Always predicts ["ces"] for each alphabetic morph. (Used as baseline.)
  - MostFrequentOriginModel: Predicts the most frequent target if the morph was in training data, otherwise "ces" or another fallback
  - MorphDictModel: Uses root and affix dictionaries.
  - WordDictModel: Uses morphological analysis to get word-level lemma and look this lemma in Etymological dictionary.
  - MorphClassifier: A learnable model (SVM, MLP, or LogisticRegression) with various extracted features optionaly embeddings.

For each enabled model, the script prints the F-score averaged over all morph instances, scores on native and borrowed morphs,
score when grouping morphs by ther text (count each unique morph just once), and the relative error reduction compared to the dummy model baseline.
"""
import time
import sys
import argparse
import os

from utils import (
    load_annotations,
    evaluate,
    relative_error_reduction,
    remove_targets,
    write_morph_statistics,
    count_sentences_words_morphs,
    single_morph_sentences_from_dict,
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
    # File paths for training set, test set, and dictionaries for baselines
    parser.add_argument("--train_file", type=str, default="data/annotations/train.tsv",
                        help="Path to the training file (default: data/annotations/train.tsv)")
    parser.add_argument("--target_file", type=str, default="data/annotations/dev.tsv",
                        help="Path to the dev/test file (default: data/annotations/dev.tsv)")
    parser.add_argument("--root_etym_file", type=str, default="data/additional_data/roots_etymology.tsv",
                        help="Path to the root etymology file (default: data/additional_data/roots_etymology.tsv)")
    parser.add_argument("--word_etym_file", type=str, default="data/additional_data/czetyl_max.tsv",
                        help="Path to the word-level etymology file (default: data/additional_data/czetyl_max.tsv)")
    parser.add_argument("--affixes_file", type=str, default="data/additional_data/affixes_etymology.tsv",
                        help="Path to the affixes file (default: data/additional_data/affixes_etymology.tsv)")

    # File paths for stats
    parser.add_argument("--outputs_dir", type=str, default="outputs",
                        help="Directory to save output files from the experiment (default: outputs)")
    parser.add_argument("--stats_lang_test", type=str, default="languages_test_stats.tsv",
                        help="File to store language stats for test (default: languages_test_stats.tsv)")
    parser.add_argument("--stats_morphs_test", type=str, default="morphs_test_stats.tsv",
                        help="File to store morph stats for test (default: morphs_test_stats.tsv)")
    parser.add_argument("--stats_lang_train", type=str, default="languages_train_stats.tsv",
                        help="File to store language stats for train (default: languages_train_stats.tsv)")
    parser.add_argument("--stats_morphs_train", type=str, default="morphs_train_stats.tsv",
                        help="File to store morph stats for train (default: morphs_train_stats.tsv)")
    parser.add_argument("--mistakes_file", type=str, default=None,
                        help="File to print mistakes of the learning model (default: None => 'mistakes' + model.name + '.tsv")
    parser.add_argument("--results_file", type=str, default="results.tsv",
                        help="File to print the results (default: results.tsv)")
    
    parser.add_argument("--print_stats", action="store_true",
                        help="Print statistics for morphs and language sequences in both the train and test sets. (default: False)")
    parser.add_argument("--print_mistakes", action="store_true",
                        help="Print mistakes made by the models on test set. (default: False)")
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
                        help="Enable all baseline models plus the learning model MorphClassifier.")
    parser.add_argument("--enable_baselines", action="store_true",
                        help="Enable all baseline models.")

    # Morph Classifier config
    parser.add_argument("--model_name", type=str, default=None,
                        help="Name for the MorphClassifier model (default: None)")
    parser.add_argument("--classifier_type", type=str, default="mlp",
                        choices=["svm", "mlp", "lr"],
                        help="Classifier type: 'svm', 'mlp', or 'lr' (default: mlp).")
    parser.add_argument("--mlp_ensemble_size", type=int, default=1,
                        help="Number of MLP classifiers in an ensemble (default: 1).")
    parser.add_argument("--mlp_hidden_size", type=int, default=30,
                        help="Hidden layer size for MLP classifier (default: 30).")
    parser.add_argument("--svm_c", type=float, default=1.0,
                        help="C parameter for LinearSVC (default: 1.0).")
    parser.add_argument("--svm_kernel", type=str, default='rbf',
                        help="Kernel for the svm model (rbf,poly,linear,sigmoid,precomputed) (default: 'rbf').")
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

    parser.add_argument("--binary", action="store_true",
                        help="Enables binary classification. Just Native or Borrowed. (default: False)" )

    return parser.parse_args()

def run_model(
    model: Model,
    train_data,
    target_data,
    baseline_f1: float = 0,
    file_mistakes: str = None,
    model_name: str = None,
    verbose:bool = True,
    load_model_path:str = None,
    results_file: str = None
) -> None:
    """
    Fits (if applicable), predicts, evaluates, and prints results for a given model.

    Args:
        model (Model): The model to run.
        train_data: The training data (list of DataSentence).
        target_data: The dev/test data with target labels.
        baseline_f1 (float): The baseline F1 score for relative error computation. If it is not provided skip relative error computation.
        mistakes_file (str): If set, logs mistakes to this file.
        model_name (str): A name/description for printing.
        verbose (bool): Enable verbose output.
        load_model_path (str): Path to the trained and saved model. If None or empty train the model from data.
        results_file (str): If provided, append a one-line TSV with metrics (no header) to this file.
    """
    start_time = time.time()
    if model_name == None: model_name = model.name
    print(f"----- {model_name} -----")
    try:
        if load_model_path:
            try:
                model.load(load_model_path)
            except Exception as error:
                print("Error when trying to load model from ", load_model_path)
                print("The following error occured:", error)
                print("Save the model first or call without load_model_path to train the model instead")
                raise error
        else:
            model.fit(train_data)
        # Remove targets from the data to simulate unlabeled data
        test_data = remove_targets(target_data)
        if verbose:
            print(f"Computing predictions on test data...")
        predictions = model.predict(test_data)

        # Evaluate
        evaluation_results = evaluate(predictions, target_data, instance_eval=True,micro_eval=True,native_borrowed_eval=True,group_by_text_eval=True, file_mistakes=file_mistakes)
        f_score = evaluation_results['f1score_instance']
        f_score_micro = evaluation_results['f1score_micro']
        f_score_on_native = evaluation_results['f1_on_native']
        f_score_on_borrowed = evaluation_results['f1_on_borrowed']
        f_score_grouped = evaluation_results['grouped_fscore']
        improvement = None
        if baseline_f1 and baseline_f1 > 0:
            improvement = relative_error_reduction(baseline_f1, f_score)
        if verbose:
            elapsed = time.time() - start_time
            print(f"Predictions computed and evaluated. Total time {elapsed:.2f} s\n")
            print("Results:")
            print(f"Standard (averaged per instance) F-score: {f_score:.1f} %")
            # print(f"Micro F-score:                         {f_score_micro:.1f} %")
            print(f"F-score on native morphs:              {f_score_on_native:.1f} %")
            print(f"F-score on borrowed morphs:            {f_score_on_borrowed:.1f} %")
            print(f"Grouped by unique morph text F-score:  {f_score_grouped:.1f} %")
            if improvement is not None:
                print(f"Relative Error Reduction vs. baseline: {improvement:.1f} %\n")
        
        # If a results_file was given, append a TSV line with the metrics (no header)
        if results_file:
            directory = os.path.dirname(results_file)
            if directory: 
                os.makedirs(directory, exist_ok=True)
                
            with open(results_file, 'at') as rf:
                # We'll log the improvement or '-' if None
                improvement_str = f"{improvement:.1f}" if improvement is not None else "-"
                rf.write(
                    f"{model_name}\t"
                    f"{f_score:.1f}\t"
                    f"{improvement_str}\t"
                    f"{f_score_on_native:.1f}\t"
                    f"{f_score_on_borrowed:.1f}\t"
                    f"{f_score_grouped:.1f}\n"
                )
    except WordDictModel.NetworkError as net_err:
        print(f"Network error while running model '{model_name}'.\nThe following exception occured: {net_err}")
        raise net_err
    
    except Exception as e:
        print(f"Error when running model: '{model_name}'")
        print(f"The following exception occurred:\n{e}\n")
        raise e

def main():
    args = parse_args()
    # args.enable_morph_classifier = True
    stats_languages_train_file = os.path.join(args.outputs_dir, args.stats_lang_train)
    stats_languages_test_file = os.path.join(args.outputs_dir, args.stats_lang_test)
    stats_morphs_train_file = os.path.join(args.outputs_dir, args.stats_morphs_train)
    stats_morphs_test_file = os.path.join(args.outputs_dir, args.stats_morphs_test)
    args.results_file = os.path.join(args.outputs_dir, args.results_file)

    # Load dev/test data
    test_sentences_target = load_annotations(args.target_file)
    # Load train data
    train_sentences = load_annotations(args.train_file)

    if args.binary:
        all_sentences = train_sentences + test_sentences_target
        for sentence in all_sentences:
            for morph in sentence:
                if morph.etymology:
                    if 'ces' in morph.etymology:
                        morph.etymology = ['ces']
                    else:
                        morph.etymology = ['borrowed']

    sentence_count_test,word_count_test,morph_count_test = count_sentences_words_morphs(test_sentences_target)
    sentence_count_train,word_count_train,morph_count_train = count_sentences_words_morphs(train_sentences)

    if args.print_stats:
        write_morph_statistics(test_sentences_target, stats_languages_test_file, stats_morphs_test_file)
        write_morph_statistics(train_sentences, stats_languages_train_file, stats_morphs_train_file)

    if not args.quiet:
        print(f"Statistics on Train -- Morphs: {morph_count_train}, Words: {word_count_train}, Sentences: {sentence_count_train}")
        print(f"Statistics on Test -- Morphs: {morph_count_test}, Words: {word_count_test}, Sentences: {sentence_count_test}\n")

    # Always run dummy internally for baseline
    dummy_model = DummyModel()
    test_sentences_for_predict = remove_targets(test_sentences_target)
    dummy_predictions = dummy_model.predict(test_sentences_for_predict)

    if args.enable_dummy or args.enable_baselines or args.enable_all:
        mistakes_file = None
        if args.print_mistakes:
            mistakes_file = os.path.join(args.outputs_dir, f"mistakes_{dummy_model.name}.tsv")
        evaluation_results = evaluate(dummy_predictions, test_sentences_target, instance_eval=True, micro_eval=True,native_borrowed_eval=True,group_by_text_eval=True,file_mistakes=mistakes_file)
        f_score_dummy = evaluation_results['f1score_instance']
        f_score_dummy_micro = evaluation_results['f1score_micro']
        f_score_native = evaluation_results['f1_on_native']
        f_score_borrowed = evaluation_results['f1_on_borrowed']
        f_score_grouped = evaluation_results['grouped_fscore']
        print("----- Dummy Model (baseline) -----")
        print(f"Standard (averaged per instance) F-score: {f_score_dummy:.1f} %")
        # print(f"Micro F-score: {f_score_dummy_micro:.1f} %")

        print(f"Grouped by unique morphs: {f_score_grouped:.1f} %")
        print(f"F-score Native {f_score_native:.1f} %, Borrowed: {f_score_borrowed:.1f} %\n")
    else:
        evaluation_results = evaluate(dummy_predictions, test_sentences_target)
        f_score_dummy = evaluation_results['f1score_instance']
    baseline_f1 = f_score_dummy

    if args.enable_mfo or args.enable_baselines or args.enable_all:
        mfo_model = MostFrequentOriginModel()
        mistakes_file = None
        if args.print_mistakes:
            mistakes_file = os.path.join(args.outputs_dir, f"mistakes_{mfo_model.name}.tsv")
        run_model(mfo_model, train_sentences, test_sentences_target,
                  baseline_f1=baseline_f1, file_mistakes=mistakes_file,verbose=(not args.quiet),results_file=args.results_file)

    # Possibly run MorphDictModel
    if args.enable_morph_dict or args.enable_baselines or args.enable_all:
        md_model = MorphDictModel(args.root_etym_file, args.affixes_file, binary=args.binary)
        mistakes_file = None
        if args.print_mistakes:
            mistakes_file = os.path.join(args.outputs_dir, f"mistakes_{md_model.name}.tsv")
        run_model(md_model, train_sentences, test_sentences_target,
                  baseline_f1=baseline_f1, file_mistakes=mistakes_file,verbose=(not args.quiet),results_file=args.results_file)

    # Possibly run WordDictModel
    if args.enable_word_dict or args.enable_baselines or args.enable_all:
        wd_model = WordDictModel(args.word_etym_file, args.affixes_file, binary=args.binary)
        mistakes_file = None
        if args.print_mistakes:
            mistakes_file = os.path.join(args.outputs_dir, f"mistakes_{wd_model.name}.tsv")
        run_model(wd_model, train_sentences, test_sentences_target,
                  baseline_f1=baseline_f1, file_mistakes=mistakes_file, verbose=(not args.quiet),results_file=args.results_file)

    # Possibly run MorphClassifier
    if args.enable_morph_classifier or args.enable_all:
        if args.extend_train:
            train_sentences += single_morph_sentences_from_dict(args.root_etym_file)
            train_sentences += single_morph_sentences_from_dict(args.affixes_file)
            if args.print_stats:
                write_morph_statistics(train_sentences,languages_file=stats_languages_train_file.replace('.tsv','_extended.tsv'),morphs_file=stats_morphs_train_file.replace('.tsv','_extended.tsv'))
            sentence_count_extended,word_count_extended,morph_count_extended = count_sentences_words_morphs(train_sentences)
            if not args.quiet:
                print(f"Statistics on extended train -- Morphs: {morph_count_extended}, Words: {word_count_extended}, Sentences: {sentence_count_extended}\n")
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
            svm_kernel=args.svm_kernel,

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
        if args.mistakes_file == None and args.print_mistakes:
            mistakes_file = os.path.join(args.outputs_dir, f"mistakes_{learning_model.name}.tsv")
        else:
            mistakes_file = args.mistakes_file #  if args.mistakes_file=None (None is default) no printing of mistakes. If mistakes_file is not None, mistakes are printed regardles of args.print_mistakes
        
        try:
            run_model(learning_model, train_sentences, test_sentences_target,
                    baseline_f1=baseline_f1, file_mistakes=mistakes_file, verbose=(not args.quiet),
                    load_model_path=args.load_model_path,results_file=args.results_file)
            if args.save_model_path:
                learning_model.save(args.save_model_path)
            elif args.save:
                learning_model.save(learning_model.name + '.pkl')
        # TODO: Make the excpetion handling better        
        except Exception as e:
            print("Terminating program after an error.")



if __name__ == "__main__":
    main()