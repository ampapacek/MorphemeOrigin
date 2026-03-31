



from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace")
    parser.add_argument("repo_dir")
    parser.add_argument("train_file")
    parser.add_argument("target_file")
    parser.add_argument("model_path")
    parser.add_argument("predictions_path")
    parser.add_argument("--model-download-url", default="")
    parser.add_argument("--classifier-type", choices=["mlp", "svm", "lr"], default="mlp")
    parser.add_argument("--mlp-hidden-size", type=int, default=30)
    parser.add_argument("--mlp-ensemble-size", type=int, default=1)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-kernel", default="rbf")
    parser.add_argument("--multi-label", action="store_true")
    parser.add_argument("--extend-train", action="store_true")
    parser.add_argument("--random-state", type=int, default=34867991)
    parser.add_argument("--min-label-freq", type=int, default=2)
    parser.add_argument("--model-name", default="")
    return parser.parse_args()


args = parse_args()

workspace = Path(args.workspace)
repo_dir = Path(args.repo_dir)
train_file = Path(args.train_file)
target_file = Path(args.target_file)
model_path = Path(args.model_path)
predictions_path = Path(args.predictions_path)

sys.path.insert(0, str(repo_dir / "src"))

from morph_classifier import MorphClassifier
from utils import load_annotations, single_morph_sentences_from_dict


def build_model(verbose: bool) -> MorphClassifier:
    return MorphClassifier(
        name=args.model_name or None,
        classifier_type=args.classifier_type,
        mlp_hidden_size=args.mlp_hidden_size,
        mlp_ensemble_size=args.mlp_ensemble_size,
        svm_c=args.svm_c,
        svm_kernel=args.svm_kernel,
        random_state=args.random_state,
        multi_label=args.multi_label,
        min_label_freq=args.min_label_freq,
        verbose=verbose,
    )


def extend_train_data(train_data):
    if not args.extend_train:
        return list(train_data)

    roots_file = repo_dir / "data" / "etymological_data" / "roots_etymology.tsv"
    affixes_file = repo_dir / "data" / "etymological_data" / "affixes_etymology.tsv"
    extended = list(train_data)
    extended.extend(single_morph_sentences_from_dict(str(roots_file)))
    extended.extend(single_morph_sentences_from_dict(str(affixes_file)))
    return extended


def print_training_summary(model: MorphClassifier, data, action_label: str | None) -> None:
    df_train = model._sentences_to_dataframe(data, lowercase=model.lower_case)

    if df_train.empty:
        raise ValueError("No training data with non-empty etymology.")

    label_counts = df_train["label"].value_counts()
    number_frames_before = len(df_train)
    df_train = df_train[df_train["label"].map(label_counts) >= model.min_label_freq]
    if df_train.empty:
        raise ValueError(
            f"All morphs were discarded because their labels' frequency < {model.min_label_freq}."
        )

    number_removed = number_frames_before - len(df_train)
    if number_removed > 0:
        print(f"Removed {number_removed} morphs with low occurence etymology sequences")

    preprocessor = model._build_preprocessor()
    x_train = df_train[["text", "word", "morph_type", "morph_position"]]
    xt_train = preprocessor.fit_transform(x_train)
    print("Shape of transformed data:", xt_train.shape)
    if action_label:
        print(f"{action_label}: {model.name}")
    print("Parameters:")
    print(f"  classifier_type={model.classifier_type}")
    if model.classifier_type.lower() == "svm":
        print(f"  svm_c={model.svm_c}")
        print(f"  svm_kernel={model.svm_kernel}")
    elif model.classifier_type.lower() == "mlp":
        print(f"  mlp_hidden_size={model.mlp_hidden_size}")
        if model.mlp_ensemble_size > 1:
            print(f"  ensemble_size={model.mlp_ensemble_size}")
    elif model.classifier_type.lower() == "lr":
        print("  Using LogisticRegression")
    if model.multi_label:
        print("  multi_label=True (using OneVsRestClassifier)")
    else:
        print("  multi_label=False (single-label)")


def is_default_downloadable_setup() -> bool:
    return (
        args.classifier_type == "mlp"
        and args.mlp_hidden_size == 30
        and args.mlp_ensemble_size == 1
        and args.multi_label
        and args.extend_train
        and args.min_label_freq == 2
    )


def load_annotations_robust(path: Path):
    text = path.read_text(encoding="utf-8")
    if "	" not in text:
        return load_annotations(str(path))

    normalized_lines = []
    for line in text.splitlines():
        prefix_len = len(line) - len(line.lstrip("	"))
        if prefix_len:
            normalized_lines.append((" " * 4 * prefix_len) + line.lstrip("	"))
        else:
            normalized_lines.append(line)
    normalized_text = "\n".join(normalized_lines) + ("\n" if text.endswith("\n") else "")

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".tsv", delete=False) as handle:
        handle.write(normalized_text)
        temp_path = Path(handle.name)
    try:
        return load_annotations(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)


def format_sentence_block(sentence, indent: int = 4) -> str:
    lines: list[str] = [sentence.sentence]
    for word in sentence.words:
        if word.text.isalnum():
            lines.append(" " * indent + word.text)
            for morph in word.morphs:
                lines.append(
                    " " * (indent * 2)
                    + morph.text
                    + "\t"
                    + str(morph.morph_type)
                    + "\t"
                    + ",".join(morph.etymology)
                )
    return "\n".join(lines)


def write_predictions_with_comments(
    template_path: Path, predictions, output_path: Path, indent: int = 4
) -> None:
    blocks = [format_sentence_block(sentence, indent=indent) for sentence in predictions]
    block_idx = 0
    output_lines: list[str] = []
    current_block_lines: list[str] = []

    def flush_block() -> None:
        nonlocal block_idx, current_block_lines
        if not current_block_lines:
            return
        if block_idx >= len(blocks):
            raise ValueError("More sentence blocks in template than predictions.")
        output_lines.append(blocks[block_idx])
        output_lines.append("")
        block_idx += 1
        current_block_lines = []

    with template_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("#"):
                flush_block()
                output_lines.append(line)
                continue
            if not line.strip():
                flush_block()
                if output_lines and output_lines[-1] != "":
                    output_lines.append("")
                continue
            current_block_lines.append(line)

    flush_block()
    if block_idx != len(blocks):
        raise ValueError(
            f"Template/prediction mismatch: consumed {block_idx} blocks but have {len(blocks)} predictions."
        )

    output_text = "\n".join(output_lines).rstrip() + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding="utf-8")


train_data = load_annotations_robust(train_file)
target_data = load_annotations_robust(target_file)
print(f"Loaded training sentences: {len(train_data)}")
print(f"Loaded target sentences: {len(target_data)}")

train_for_model = extend_train_data(train_data)
if args.extend_train:
    print(f"Training sentences after --extend_train: {len(train_for_model)}")

downloaded_model = False
should_try_download = is_default_downloadable_setup() and bool(args.model_download_url)
if model_path.exists():
    action_label = None
elif should_try_download:
    action_label = "Downloading model"
else:
    action_label = "Fiting model"

summary_model = build_model(verbose=False)
print_training_summary(summary_model, train_for_model, action_label)

if not model_path.exists() and should_try_download:
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(args.model_download_url, model_path)
        downloaded_model = True
    except Exception as exc:
        print(f"Could not download pretrained model: {exc}")
        print("Training the model instead...")

model = build_model(verbose=False)
if model_path.exists():
    model.load(str(model_path))
    if downloaded_model:
        print(f"Downloaded and loaded pretrained model: {model_path}")
    else:
        print(f"Loaded stored model from: {model_path}")
else:
    model.fit(train_for_model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"Saved trained model to {model_path}")

print("\nPredicting etymologycal origins...")
predictions = model.predict(target_data)
predictions_path.parent.mkdir(parents=True, exist_ok=True)
write_predictions_with_comments(target_file, predictions, predictions_path)
print("Predictions completed")
print(f"Saved predictions to {predictions_path}")
