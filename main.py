#!/usr/bin/env python3
"""
This script shows various approaches for Morph Etymology prediction.
It loads annotated test sentences, computes statistics, removes etymology targets,
and then predicts etymology using various models from baseline simple models to learning models:
  - DummyEtymologyModel: Always predicts ["ces"] for each alphabetic morph.
  - RootModel: Uses root and affix dictionaries.
  - WordModel: Uses a morphologycal analyser to get lemma for each word and then word-level etymology dictionary plus affix rules.

For each model, the script prints the F-score, accuracy, and the relative error reduction 
compared to the dummy model baseline.
"""

from utils import load_annotations, evaluate, relative_error_reduction, remove_targets, statistics, calculate_cohen_kappa
from baselines import DummyModel, MorphDictModel, WordDictModel, MostFrequentOriginModel

from morph_classifier import MorphClassifier

import sys
import copy
import time

def main():
    # Set file paths.
    train_file = 'annotations/train.tsv'
    dev_file = 'annotations/dev.tsv'

    # These files are used to load dictionaries for the rule models.
    root_etym_file = 'additional_data/roots_etymology.tsv'
    word_etym_file = 'additional_data/czetyl_max.tsv'

    affixes_file = 'additional_data/affixes_etymology.tsv'


    languages_stats_file = 'languages_dev_stats.tsv'
    morphs_file = 'morphs_dev_stats.tsv'
    
    # Load annotated dev sentences (target data).
    dev_sentences_target = load_annotations(dev_file)


    # Load train data
    train_sentences = load_annotations(train_file)



    # Compute and print basic statistics on dev.
    morph_count, word_count, sentence_count = statistics(dev_sentences_target, languages_stats_file, morphs_file)
    print(f"Statistics on Dev -- Morphs: {morph_count}, Words: {word_count}, Sentences: {sentence_count}")
    print()

    # Compute and print basic statistics on train.
    morph_count, word_count, sentence_count = statistics(train_sentences, languages_stats_file.replace('_dev.tsv', '_train.tsv'), morphs_file.replace('_dev.tsv', '_train.tsv'))
    print(f"Statistics on Train -- Morphs: {morph_count}, Words: {word_count}, Sentences: {sentence_count}")
    print()

    # Remove etymology targets to simulate unlabeled data.
    test_sentences = remove_targets(dev_sentences_target)
    
    # ---------------------------
    # Dummy Model Evaluation
    # ---------------------------
    dummy_model = DummyModel()
    dummy_predictions = dummy_model.predict(test_sentences)
    f_score_dummy, accuracy_dummy = evaluate(dummy_predictions, dev_sentences_target)
    print("----- Dummy Model -----")
    print(f"F-score: {f_score_dummy:.3f} %, Accuracy: {accuracy_dummy:.3f} %")
    print()
    # Use dummy as baseline for relative error reduction.
    baseline_f1 = f_score_dummy

    mfo = True
    if mfo:
      # ---------------------------
      # Most Frequent Origin Model Evaluation
      # ---------------------------
      mfo_model = MostFrequentOriginModel()
      mfo_model.fit(train_sentences)
      mfo_predictions = mfo_model.predict(test_sentences)
      f_score_mfo, accuracy_mfo = evaluate(mfo_predictions, dev_sentences_target)
      improvement_mfo = relative_error_reduction(baseline_f1, f_score_mfo)
      print("----- Most Frequent Origin Model -----")
      print(f"F-score: {f_score_mfo:.3f} %, Accuracy: {accuracy_mfo:.3f} %")
      print(f"Relative Error Reduction (MFO): {improvement_mfo:.3f} %")
      print()

    run_morph_dict_model = True
    if run_morph_dict_model:
      morp_dict_model = MorphDictModel(root_etym_file, affixes_file)
      rule_predictions = morp_dict_model.predict(test_sentences)
      f_score_mdp, accuracy_mdp = evaluate(rule_predictions, dev_sentences_target, 'mistakes_mdp.tsv')
      improvement_rule = relative_error_reduction(baseline_f1, f_score_mdp)
      print("----- Morph Dict Model -----")
      print(f"F-score: {f_score_mdp:.3f} %, Accuracy: {accuracy_mdp:.3f} %")
      print(f"Relative Error Reduction (MorphDictModel): {improvement_rule:.3f} %")
      print()

    run_word_dict_model = True
    if run_word_dict_model:
      word_dict_model = WordDictModel(word_etym_file, affixes_file)
      word_predictions = word_dict_model.predict(test_sentences)
      f_score_wdm, accuracy_wdm = evaluate(word_predictions, dev_sentences_target, 'mistakes_wdm.tsv')
      improvement_wdm = relative_error_reduction(baseline_f1, f_score_wdm)
      print("----- Word Dict Model -----")
      print(f"F-score: {f_score_wdm:.3f} %, Accuracy: {accuracy_wdm:.3f} %")
      print(f"Relative Error Reduction (Word Dict Model): {improvement_wdm:.3f} %")
      print()


    # ---------------------------
    # Learning Model
    # ---------------------------
    # Initialize and fit the MorphClassifier on the training data.

    model = MorphClassifier(name="mlp_no_embeddings",use_word_embedding=False,random_state=34867991,classifier_type='mlp')

    model.fit(train_sentences)
    test_sentences = remove_targets(test_sentences)
    # Predict etymologies using the trained model.
    model_predictions = model.predict(test_sentences)
    
    # Evaluate the model's predictions against the target annotations.
    f_score_model, accuracy_model = evaluate(model_predictions, dev_sentences_target,f"mistakes_{model.name}.tsv")
    improvement = relative_error_reduction(baseline_f1, f_score_model)
    
    # Print evaluation results.
    print()
    print("----- MorphClassificationModel Evaluation -----")
    print(f"Model {model.name}, F-score: {f_score_model:.3f} %, Accuracy: {accuracy_model:.3f} %")
    print(f"Relative Error Reduction: {improvement:.3f} %")
    print()


    model_svm = MorphClassifier(name="svm_with_embeddings",use_word_embedding=False,use_morph_embedding=True,embedding_dimension=300,random_state=34867991,classifier_type='svm')

    model_svm.fit(train_sentences)
    test_sentences = remove_targets(test_sentences)
    # Predict etymologies using the trained model.
    model_svm_predictions = model_svm.predict(test_sentences)
    
    # Evaluate the model's predictions against the target annotations.
    f_score_model, accuracy_model = evaluate(model_svm_predictions, dev_sentences_target,f"mistakes_{model_svm.name}.tsv")
    improvement = relative_error_reduction(baseline_f1, f_score_model)
    
    # Print evaluation results.
    print()
    print("----- MorphClassificationModel Evaluation -----")
    print(f"Model {model_svm.name},F-score: {f_score_model:.3f} %, Accuracy: {accuracy_model:.3f} %")
    print(f"Relative Error Reduction: {improvement:.3f} %")
    print()
    
if __name__ == "__main__":
    main()