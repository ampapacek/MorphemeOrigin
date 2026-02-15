#!/usr/bin/env python3
import argparse

from utils import evaluate, load_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted morpheme etymology annotations.")
    parser.add_argument("--pred_file", type=str, required=True, help="Predicted annotation file.")
    parser.add_argument("--gold_file", type=str, required=True, help="Gold annotation file.")
    parser.add_argument(
        "--mistakes_file",
        type=str,
        default=None,
        help="Optional path to write per-morph mistakes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_annotations(args.pred_file)
    gold = load_annotations(args.gold_file)

    results = evaluate(
        predictions,
        gold,
        instance_eval=True,
        micro_eval=True,
        native_borrowed_eval=True,
        group_by_text_eval=True,
        morph_type_eval=True,
        file_mistakes=args.mistakes_file,
    )

    print("Evaluation:")
    for key in [
        "f1score_instance",
        "accuracy_instance",
        "f1score_micro",
        "f1_on_native",
        "f1_on_borrowed",
        "grouped_fscore",
        "f1_on_root",
        "f1_on_derivational_affix",
        "f1_on_inflectional_affix",
        "count_root",
        "count_derivational_affix",
        "count_inflectional_affix",
    ]:
        if key in results:
            value = results[key]
            if isinstance(value, int):
                print(f"{key}\t{value}")
            else:
                print(f"{key}\t{value:.2f}")

    if args.mistakes_file:
        print(f"Mistakes written to: {args.mistakes_file}")


if __name__ == "__main__":
    main()
