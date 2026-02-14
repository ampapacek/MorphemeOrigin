#!/usr/bin/env python3
import argparse
import math
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence

import requests

from data_sentece import DataSentence, Morph
from utils import evaluate, load_annotations, pprint_sentences, remove_targets

try:
    import fcntl
except ImportError:
    fcntl = None


LLM_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MORPH_TAGS = {"R", "D", "I"}


@dataclass
class SentenceBatch:
    start: int
    end: int
    morph_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict morpheme etymology with an LLM in sentence-safe batches."
    )
    parser.add_argument("--model", type=str, required=True, help="Model ID.")
    parser.add_argument("--input_file", type=str, default="data/annotations/dev_for_prediction.tsv")
    parser.add_argument(
        "--sentences",
        type=float,
        default=None,
        help=(
            "How many input sentences to process. "
            "If 0 < value < 1, it is treated as a file portion (e.g. 0.2 = first 20%%). "
            "If value >= 1, it must be an integer sentence count."
        ),
    )
    parser.add_argument("--train_file", type=str, default="data/annotations/train.tsv")
    parser.add_argument(
        "--gold_file",
        type=str,
        default=None,
        help=(
            "Gold annotations for evaluation. "
            "If omitted, tries to derive from --input_file by replacing *_for_prediction.tsv with *.tsv."
        ),
    )
    parser.add_argument("--prompt_file", type=str, default="prompt_for_ai.txt")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help=(
            "Output path. If omitted, it is auto-generated from model, train context, "
            "and selected sentence size."
        ),
    )
    parser.add_argument("--batch_morph_target", type=int, default=350)
    parser.add_argument(
        "--train_context_morphs",
        type=int,
        default=0,
        help="Approximate number of train morphs sent as in-context examples; 0 disables.",
    )
    parser.add_argument("--api_key_env", type=str, default="LLM_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=20000)
    parser.add_argument("--timeout_sec", type=int, default=300)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_wait_sec", type=float, default=3.0)
    parser.add_argument("--sleep_between_batches_sec", type=float, default=0.0)
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip automatic evaluation after predictions are generated.",
    )
    parser.add_argument(
        "--eval_results_file",
        type=str,
        default="outputs/llm_eval_results.tsv",
        help="Common TSV file where one evaluation summary line is appended per run.",
    )
    parser.add_argument(
        "--mistakes_file",
        type=str,
        default=None,
        help="Optional mistakes output path. If omitted, a file in outputs/ is generated automatically.",
    )
    parser.add_argument("--http_referer", type=str, default=None)
    parser.add_argument("--x_title", type=str, default="MorphemeOrigin")
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional log file with sent/received messages and timing info.",
    )
    parser.add_argument(
        "--log_append",
        action="store_true",
        help="Append to log file instead of overwriting it.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log_block(log_stream, title: str, content: str | None = None) -> None:
    if not log_stream:
        return
    log_stream.write(f"[{now_iso()}] {title}\n")
    if content:
        log_stream.write(content)
        if not content.endswith("\n"):
            log_stream.write("\n")
    log_stream.write("\n")
    log_stream.flush()


def flatten_morphs(sentences: Sequence[DataSentence]) -> List[Morph]:
    morphs: List[Morph] = []
    for sentence in sentences:
        for word in sentence.words:
            morphs.extend(word.morphs)
    return morphs


def render_sentences(sentences: Sequence[DataSentence], include_etymology: bool) -> str:
    lines: List[str] = []
    for sentence in sentences:
        lines.append(sentence.sentence)
        for word in sentence.words:
            lines.append(f"    {word.text}")
            for morph in word.morphs:
                morph_line = f"        {morph.text}\t{morph.morph_type}"
                if include_etymology:
                    morph_line += "\t" + ",".join(morph.etymology)
                lines.append(morph_line)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def make_batches(sentences: Sequence[DataSentence], target_morphs: int) -> List[SentenceBatch]:
    batches: List[SentenceBatch] = []
    start = 0
    current_count = 0
    for idx, sentence in enumerate(sentences):
        sentence_morphs = sentence.morph_count
        if current_count > 0 and current_count + sentence_morphs > target_morphs:
            batches.append(SentenceBatch(start=start, end=idx, morph_count=current_count))
            start = idx
            current_count = sentence_morphs
        else:
            current_count += sentence_morphs
    if start < len(sentences):
        batches.append(SentenceBatch(start=start, end=len(sentences), morph_count=current_count))
    return batches


def strip_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_morph_rows(model_output: str) -> List[tuple[str, str, str]]:
    rows: List[tuple[str, str, str]] = []
    for raw_line in strip_fence(model_output).splitlines():
        line = raw_line.lstrip()
        parts: list[str]
        if "\t" in line:
            parts = line.split("\t")
        else:
            # Recovery parser for outputs that lost tab formatting.
            parts = line.split(maxsplit=2)
        if len(parts) < 2:
            continue
        morph_text = parts[0].strip()
        morph_tag = parts[1].strip().upper()
        if morph_tag not in MORPH_TAGS:
            continue
        etymology = parts[2].strip() if len(parts) >= 3 else ""
        rows.append((morph_text, morph_tag, etymology))
    return rows


def normalize_etymology(field: str) -> List[str]:
    if not field:
        return []
    # Accept commas/spaces/semicolons and keep only ISO3-like tokens.
    parts = re.split(r"[,\s;/|]+", field.lower())
    return [code for code in parts if re.fullmatch(r"[a-z]{3}", code)]


def find_matching_rows(
    parsed_rows: Sequence[tuple[str, str, str]], expected_morphs: Sequence[Morph]
) -> Sequence[tuple[str, str, str]] | None:
    expected = [(m.text, str(m.morph_type)) for m in expected_morphs]
    n = len(expected)
    if len(parsed_rows) < n:
        return None
    for start in range(len(parsed_rows) - n, -1, -1):
        window = parsed_rows[start : start + n]
        if all(window[i][0] == expected[i][0] and window[i][1] == expected[i][1] for i in range(n)):
            return window
    return None


def recover_matching_rows(
    parsed_rows: Sequence[tuple[str, str, str]], expected_morphs: Sequence[Morph]
) -> tuple[List[tuple[str, str, str]], dict[str, int]]:
    """
    Best-effort row alignment when strict contiguous matching fails.
    Priority:
    1) text + tag in forward order
    2) text-only in forward order (tag coerced to expected tag)
    3) missing row -> empty etymology (caller defaults to ces for non-numeric)
    """
    recovered: List[tuple[str, str, str]] = []
    row_idx = 0
    exact = 0
    text_only = 0
    missing = 0

    for morph in expected_morphs:
        expected_text = morph.text
        expected_tag = str(morph.morph_type)
        found_idx: int | None = None

        for i in range(row_idx, len(parsed_rows)):
            if parsed_rows[i][0] == expected_text and parsed_rows[i][1] == expected_tag:
                found_idx = i
                break
        if found_idx is not None:
            recovered.append(parsed_rows[found_idx])
            row_idx = found_idx + 1
            exact += 1
            continue

        for i in range(row_idx, len(parsed_rows)):
            if parsed_rows[i][0] == expected_text:
                found_idx = i
                break
        if found_idx is not None:
            recovered.append((expected_text, expected_tag, parsed_rows[found_idx][2]))
            row_idx = found_idx + 1
            text_only += 1
            continue

        recovered.append((expected_text, expected_tag, ""))
        missing += 1

    stats = {"exact": exact, "text_only": text_only, "missing": missing}
    return recovered, stats


def etymology_with_default(morph: Morph, field: str) -> List[str]:
    codes = normalize_etymology(field)
    if codes:
        return codes
    if morph.text.isdigit():
        return []
    return ["ces"]


def read_text(path: str) -> str:
    with open(path, "rt", encoding="utf-8") as f:
        return f.read().strip()


def choose_train_context(sentences: Sequence[DataSentence], target_morphs: int) -> List[DataSentence]:
    if target_morphs <= 0:
        return []
    selected: List[DataSentence] = []
    morphs = 0
    for sentence in sentences:
        selected.append(sentence)
        morphs += sentence.morph_count
        if morphs >= target_morphs:
            break
    return selected


def select_input_subset(sentences: Sequence[DataSentence], selector: float | None) -> List[DataSentence]:
    if selector is None:
        return list(sentences)
    if selector <= 0:
        raise ValueError("--sentences must be > 0.")

    total = len(sentences)
    if total == 0:
        return []

    if 0 < selector < 1:
        count = max(1, math.ceil(total * selector))
        return list(sentences[:count])

    if not float(selector).is_integer():
        raise ValueError("When --sentences >= 1, it must be an integer (e.g. 50, not 50.5).")

    count = int(selector)
    if count <= 0:
        raise ValueError("--sentences must be > 0.")
    return list(sentences[: min(total, count)])


def sanitize_filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return token or "unknown"


def sentence_size_token(selector: float | None, selected_count: int) -> str:
    if selector is None:
        return f"all_{selected_count}"
    if 0 < selector < 1:
        fraction = f"{selector:.4f}".rstrip("0").rstrip(".").replace(".", "p")
        return f"frac_{fraction}_{selected_count}"
    return f"n_{selected_count}"


def resolve_output_file(
    *,
    output_file: str | None,
    model: str,
    train_context_morphs: int,
    sentence_selector: float | None,
    selected_sentence_count: int,
) -> str:
    if output_file:
        return output_file

    model_token = sanitize_filename_token(model)
    train_token = f"trainm_{max(0, train_context_morphs)}"
    size_token = sentence_size_token(sentence_selector, selected_sentence_count)
    filename = f"llm_predictions_{model_token}_{train_token}_{size_token}.tsv"
    return os.path.join("outputs", filename)


def resolve_gold_file(gold_file: str | None, input_file: str) -> str:
    if gold_file:
        return gold_file
    if input_file.endswith("_for_prediction.tsv"):
        return input_file[: -len("_for_prediction.tsv")] + ".tsv"
    else:
        return None


def resolve_mistakes_file(mistakes_file: str | None, output_file: str) -> str:
    if mistakes_file:
        return mistakes_file
    output_base = os.path.splitext(os.path.basename(output_file))[0]
    return os.path.join("outputs", f"mistakes_{output_base}.tsv")


def append_eval_summary_line(
    *,
    eval_results_file: str,
    model: str,
    output_file: str,
    gold_file: str,
    mistakes_file: str,
    selected_sentences: int,
    train_context_morphs: int,
    metrics: dict[str, float],
) -> None:
    directory = os.path.dirname(eval_results_file)
    if directory:
        os.makedirs(directory, exist_ok=True)

    header = (
        "timestamp\tmodel\tselected_sentences\ttrain_context_morphs\t"
        "f1score_instance\taccuracy_instance\tf1score_micro\t"
        "f1_on_native\tf1_on_borrowed\tgrouped_fscore\t"
        "output_file\tgold_file\tmistakes_file\n"
    )

    line = (
        f"{now_iso()}\t"
        f"{model}\t"
        f"{selected_sentences}\t"
        f"{train_context_morphs}\t"
        f"{metrics.get('f1score_instance', 0.0):.2f}\t"
        f"{metrics.get('accuracy_instance', 0.0):.2f}\t"
        f"{metrics.get('f1score_micro', 0.0):.2f}\t"
        f"{metrics.get('f1_on_native', 0.0):.2f}\t"
        f"{metrics.get('f1_on_borrowed', 0.0):.2f}\t"
        f"{metrics.get('grouped_fscore', 0.0):.2f}\t"
        f"{output_file}\t"
        f"{gold_file}\t"
        f"{mistakes_file}\n"
    )

    with open(eval_results_file, "at", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                f.write(header)
            f.write(line)
            f.flush()
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def call_llm_api(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    http_referer: str | None,
    x_title: str | None,
) -> tuple[str, dict]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if x_title:
        headers["X-Title"] = x_title

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=timeout_sec)
    response.raise_for_status()
    data = response.json()
    message_content = data["choices"][0]["message"]["content"]
    if isinstance(message_content, str):
        return message_content, data
    if isinstance(message_content, list):
        chunks: List[str] = []
        for chunk in message_content:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                chunks.append(chunk.get("text", ""))
        return "".join(chunks).strip(), data
    raise ValueError(f"Unsupported response format: {type(message_content)}")


def main() -> None:
    args = parse_args()

    api_key = os.getenv(args.api_key_env) or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"Environment variable {args.api_key_env} is not set (or OPENROUTER_API_KEY fallback is missing)."
        )

    system_prompt = read_text(args.prompt_file)
    all_input_sentences = load_annotations(args.input_file)
    input_sentences = select_input_subset(all_input_sentences, args.sentences)
    output_file = resolve_output_file(
        output_file=args.output_file,
        model=args.model,
        train_context_morphs=args.train_context_morphs,
        sentence_selector=args.sentences,
        selected_sentence_count=len(input_sentences),
    )
    prediction_sentences = remove_targets(input_sentences)
    gold_file = resolve_gold_file(args.gold_file, args.input_file)
    mistakes_file = resolve_mistakes_file(args.mistakes_file, output_file)
    batches = make_batches(input_sentences, args.batch_morph_target)
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # Clear old content so checkpoint writes always reflect this run.
    with open(output_file, "wt", encoding="utf-8"):
        pass
    log_stream = None
    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        mode = "a" if args.log_append else "wt"
        log_stream = open(args.log_file, mode, encoding="utf-8")
        log_block(
            log_stream,
            "Run started",
            (
                f"model={args.model}\n"
                f"input_file={args.input_file}\n"
                f"sentences_selector={args.sentences}\n"
                f"loaded_sentences={len(all_input_sentences)}\n"
                f"selected_sentences={len(input_sentences)}\n"
                f"train_file={args.train_file}\n"
                f"gold_file={gold_file}\n"
                f"output_file={output_file}\n"
                f"mistakes_file={mistakes_file}\n"
                f"eval_results_file={args.eval_results_file}\n"
                f"skip_eval={args.skip_eval}\n"
                f"batch_morph_target={args.batch_morph_target}\n"
                f"max_retries={args.max_retries}\n"
                f"temperature={args.temperature}\n"
                f"max_tokens={args.max_tokens}\n"
            ),
        )

    try:
        train_context_text = ""
        if args.train_context_morphs > 0:
            train_sentences = load_annotations(args.train_file)
            train_context = choose_train_context(train_sentences, args.train_context_morphs)
            train_context_text = render_sentences(train_context, include_etymology=True)
            log_block(
                log_stream,
                "Train context enabled",
                (
                    f"train_context_morphs_target={args.train_context_morphs}\n"
                    f"selected_sentences={len(train_context)}"
                ),
            )

        print(
            f"Loaded {len(all_input_sentences)} sentences from {args.input_file}; "
            f"selected {len(input_sentences)} sentence(s) for processing. "
            f"Running {len(batches)} batch requests with target {args.batch_morph_target} morphs per batch."
        )

        for batch_idx, batch in enumerate(batches, start=1):
            batch_input = input_sentences[batch.start : batch.end]
            batch_prediction = prediction_sentences[batch.start : batch.end]
            expected_morphs = flatten_morphs(batch_input)
            batch_text = render_sentences(batch_input, include_etymology=False)
            log_block(
                log_stream,
                f"Batch {batch_idx}/{len(batches)} prepared",
                (
                    f"sentence_range=[{batch.start},{batch.end})\n"
                    f"sentence_count={batch.end - batch.start}\n"
                    f"morph_count={batch.morph_count}"
                ),
            )

            base_messages = [{"role": "system", "content": system_prompt}]
            if train_context_text:
                base_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Learning examples from train data (already annotated):\n\n"
                            f"{train_context_text}"
                        ),
                    }
                )
            base_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Annotate the following batch and return only the annotated text in the same format. "
                        "Do not add explanations or markdown fences.\n\n"
                        f"{batch_text}"
                    ),
                }
            )

            assigned = False
            for attempt in range(1, args.max_retries + 1):
                try:
                    request_log = {
                        "model": args.model,
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                        "batch_idx": batch_idx,
                        "attempt": attempt,
                        "messages": base_messages,
                    }
                    log_block(
                        log_stream,
                        f"Batch {batch_idx} attempt {attempt} request",
                        json.dumps(request_log, ensure_ascii=False, indent=2),
                    )

                    call_start = time.perf_counter()
                    output_text, response_data = call_llm_api(
                        api_key=api_key,
                        model=args.model,
                        messages=base_messages,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        timeout_sec=args.timeout_sec,
                        http_referer=args.http_referer,
                        x_title=args.x_title,
                    )
                    call_elapsed = time.perf_counter() - call_start
                    response_log = {
                        "batch_idx": batch_idx,
                        "attempt": attempt,
                        "elapsed_seconds": round(call_elapsed, 3),
                        "response": response_data,
                        "assistant_text": output_text,
                    }
                    log_block(
                        log_stream,
                        f"Batch {batch_idx} attempt {attempt} response",
                        json.dumps(response_log, ensure_ascii=False, indent=2),
                    )
                    parsed_rows = parse_morph_rows(output_text)
                    matching = find_matching_rows(parsed_rows, expected_morphs)
                    recovery_stats: dict[str, int] | None = None
                    if matching is None:
                        matching, recovery_stats = recover_matching_rows(parsed_rows, expected_morphs)

                    predicted_morphs = flatten_morphs(batch_prediction)
                    for morph, row in zip(predicted_morphs, matching):
                        morph.etymology = etymology_with_default(morph, row[2])

                    if recovery_stats is not None:
                        log_block(
                            log_stream,
                            f"Batch {batch_idx} formatting/alignment recovery",
                            (
                                f"parsed_rows={len(parsed_rows)}\n"
                                f"expected_rows={len(expected_morphs)}\n"
                                f"exact_matches={recovery_stats['exact']}\n"
                                f"text_only_matches={recovery_stats['text_only']}\n"
                                f"defaulted_missing_rows={recovery_stats['missing']}"
                            ),
                        )
                        print(
                            f"Batch {batch_idx}: recovered non-aligned output "
                            f"(exact={recovery_stats['exact']}, text_only={recovery_stats['text_only']}, "
                            f"defaulted={recovery_stats['missing']})."
                        )

                    assigned = True
                    print(
                        f"Batch {batch_idx}/{len(batches)} done: "
                        f"{batch.end - batch.start} sentences, {batch.morph_count} morphs."
                    )
                    log_block(
                        log_stream,
                        f"Batch {batch_idx} completed",
                        f"status=success\nelapsed_seconds={round(call_elapsed, 3)}",
                    )
                    # Persist progress after each finished batch.
                    pprint_sentences(prediction_sentences[: batch.end], output_file)
                    log_block(
                        log_stream,
                        f"Batch {batch_idx} checkpoint saved",
                        f"output_file={output_file}\nsentences_saved={batch.end}",
                    )
                    break
                except (requests.RequestException, ValueError, KeyError) as exc:
                    log_block(
                        log_stream,
                        f"Batch {batch_idx} attempt {attempt} failed",
                        f"error_type={type(exc).__name__}\nerror={str(exc)}",
                    )
                    if attempt >= args.max_retries:
                        raise RuntimeError(
                            f"Batch {batch_idx} failed after {args.max_retries} attempts."
                        ) from exc
                    wait = args.retry_wait_sec * attempt
                    print(f"Batch {batch_idx} attempt {attempt} failed: {exc}. Retrying in {wait:.1f}s.")
                    time.sleep(wait)

            if not assigned:
                raise RuntimeError(f"Failed to assign predictions for batch {batch_idx}.")

            if args.sleep_between_batches_sec > 0:
                time.sleep(args.sleep_between_batches_sec)

        print(f"Predictions written to: {output_file}")
        log_block(log_stream, "Prediction run finished", f"predictions_file={output_file}")

        if not args.skip_eval and gold_file:
            gold_sentences_full = load_annotations(gold_file)
            gold_sentences = select_input_subset(gold_sentences_full, args.sentences)
            if len(gold_sentences) != len(prediction_sentences):
                raise RuntimeError(
                    "Prediction/gold sentence count mismatch: "
                    f"{len(prediction_sentences)} vs {len(gold_sentences)}."
                )

            eval_results = evaluate(
                prediction_sentences,
                gold_sentences,
                instance_eval=True,
                micro_eval=True,
                native_borrowed_eval=True,
                group_by_text_eval=True,
                file_mistakes=mistakes_file,
            )
            append_eval_summary_line(
                eval_results_file=args.eval_results_file,
                model=args.model,
                output_file=output_file,
                gold_file=gold_file,
                mistakes_file=mistakes_file,
                selected_sentences=len(prediction_sentences),
                train_context_morphs=args.train_context_morphs,
                metrics=eval_results,
            )

            print("Evaluation:")
            print(f"f1score_instance\t{eval_results.get('f1score_instance', 0.0):.2f}")
            print(f"accuracy_instance\t{eval_results.get('accuracy_instance', 0.0):.2f}")
            print(f"f1score_micro\t{eval_results.get('f1score_micro', 0.0):.2f}")
            print(f"f1_on_native\t{eval_results.get('f1_on_native', 0.0):.2f}")
            print(f"f1_on_borrowed\t{eval_results.get('f1_on_borrowed', 0.0):.2f}")
            print(f"grouped_fscore\t{eval_results.get('grouped_fscore', 0.0):.2f}")
            print(f"Mistakes written to: {mistakes_file}")
            print(f"Evaluation summary appended to: {args.eval_results_file}")
            log_block(
                log_stream,
                "Evaluation finished",
                (
                    f"mistakes_file={mistakes_file}\n"
                    f"eval_results_file={args.eval_results_file}\n"
                    f"f1score_instance={eval_results.get('f1score_instance', 0.0):.2f}\n"
                    f"accuracy_instance={eval_results.get('accuracy_instance', 0.0):.2f}\n"
                    f"f1score_micro={eval_results.get('f1score_micro', 0.0):.2f}\n"
                    f"f1_on_native={eval_results.get('f1_on_native', 0.0):.2f}\n"
                    f"f1_on_borrowed={eval_results.get('f1_on_borrowed', 0.0):.2f}\n"
                    f"grouped_fscore={eval_results.get('grouped_fscore', 0.0):.2f}"
                ),
            )
        else:
            log_block(log_stream, "Evaluation skipped", "skip_eval=true")

        log_block(log_stream, "Run finished", f"predictions_file={output_file}")
    finally:
        if log_stream:
            log_stream.close()


if __name__ == "__main__":
    main()
