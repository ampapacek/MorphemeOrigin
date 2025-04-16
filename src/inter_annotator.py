from utils import load_annotations,calculate_cohen_kappa,evaluate

# TODO: make this as argumetns of the script
# Set file paths.
dev_file_ales = 'data/annotations/dev.tsv'
dev_file_tomas = 'data/annotations/dev_annotator2.tsv'
differences_file = 'outputs/annotator_differences.tsv'

# Load annotated test sentences (target data).
dev_sentences_annotator1_ales = load_annotations(dev_file_ales)

dev_sentences_annotator2 = load_annotations(dev_file_tomas)


annotator1 = dev_sentences_annotator1_ales
annotator2 = dev_sentences_annotator2
results = evaluate(annotator2,annotator1,file_mistakes=differences_file)

print("----- Inter-Annotator Agreement -----")
print(f"F-score {results['f1score_instance']:.2f} %, Exact_match {results['accuracy_instance']:.2f} %")

cohen_kappa = calculate_cohen_kappa(annotator1,annotator2)
print(f"Coehns kappa: {cohen_kappa:.2f}")