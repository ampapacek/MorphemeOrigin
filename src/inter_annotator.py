#!/usr/bin/env python3

"""
Compute agreement between two annotators.

Metrics:
    • F1 score 
    • Exact‑match accuracy
    • Cohen's kappa  (label sequence treated as a single class)

Optionally save every disagreement into a TSV file.
"""

from utils import load_annotations,calculate_cohen_kappa,evaluate
import argparse, sys, os

def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Inter‑annotator agreement of morpheme‑level etymologies"
    )
    ap.add_argument("annotator1", help="TSV produced by annotator 1")
    ap.add_argument("annotator2", help="TSV produced by annotator 2")
    ap.add_argument(
        "-o", "--out‑mistakes", metavar="FILE",
        help="Write morphs where annotators disagree to FILE (TSV)"
    )
    return ap.parse_args()

# default paths 
DEF_ANN1   = "data/annotations/dev.tsv"
DEF_ANN2   = "data/annotations/dev_annotator2.tsv"
DEF_MISMAT = "outputs/annotator_differences.tsv"

def get_args():
    ap = argparse.ArgumentParser(
        description="Inter‑annotator agreement of morpheme‑level etymologies"
    )
    ap.add_argument("-a1", "--annotator1", default=DEF_ANN1,
                    help=f"Annotator‑1 TSV (default: {DEF_ANN1})")
    ap.add_argument("-a2", "--annotator2", default=DEF_ANN2,
                    help=f"Annotator‑2 TSV (default: {DEF_ANN2})")
    ap.add_argument("-o", "--out-mistakes", default=DEF_MISMAT,
                    metavar="FILE",
                    help=("Write disagreements to FILE "
                          f"(default: {DEF_MISMAT}). "
                          "Use empty string to skip writing."))
    return ap.parse_args()

def main():
    args = get_args()

    # sanity‑check input files
    for path in (args.annotator1, args.annotator2):
        if not os.path.isfile(path):
            sys.exit(f"ERROR: file '{path}' does not exist.")

    annotator1 = load_annotations(args.annotator1)
    annotator2 = load_annotations(args.annotator2)

    out_file = args.out_mistakes if args.out_mistakes else None

    results = evaluate(
        annotator1, annotator2,  
        instance_eval=True,
        file_mistakes=out_file
    )

    f1      = results["f1score_instance"]
    acc     = results["accuracy_instance"]
    kappa   = calculate_cohen_kappa(annotator1, annotator2)

    print("─── Inter‑Annotator Agreement ─────")
    print(f"F1 Score:            {f1:.2f} %")
    print(f"Exact‑match accuracy:    {acc:.2f} %")
    print(f"Cohen’s κ:               {kappa:.2f}")
    if out_file:
        print(f"Disagreements written to: {out_file}")

if __name__ == "__main__":
    main()