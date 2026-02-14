#!/usr/bin/env python3
import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence

import requests

from data_sentece import DataSentence, Morph
from utils import load_annotations, pprint_sentences, remove_targets


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
    parser.add_argument("--train_file", type=str, default="data/annotations/train.tsv")
    parser.add_argument("--prompt_file", type=str, default="prompt_for_ai.txt")
    parser.add_argument("--output_file", type=str, default="outputs/llm_predictions.tsv")
    parser.add_argument("--batch_morph_target", type=int, default=350)
    parser.add_argument(
        "--train_context_morphs",
        type=int,
        default=0,
        help="Approximate number of train morphs sent as in-context examples; 0 disables.",
    )
    parser.add_argument("--api_key_env", type=str, default="LLM_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=16000)
    parser.add_argument("--timeout_sec", type=int, default=180)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_wait_sec", type=float, default=3.0)
    parser.add_argument("--sleep_between_batches_sec", type=float, default=0.0)
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
        if "\t" not in line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        morph_text = parts[0].strip()
        morph_tag = parts[1].strip()
        if morph_tag not in MORPH_TAGS:
            continue
        etymology = parts[2].strip() if len(parts) >= 3 else ""
        rows.append((morph_text, morph_tag, etymology))
    return rows


def normalize_etymology(field: str) -> List[str]:
    if not field:
        return []
    return [code.strip() for code in field.split(",") if code.strip()]


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
    input_sentences = load_annotations(args.input_file)
    prediction_sentences = remove_targets(input_sentences)
    batches = make_batches(input_sentences, args.batch_morph_target)
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # Clear old content so checkpoint writes always reflect this run.
    with open(args.output_file, "wt", encoding="utf-8"):
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
                f"train_file={args.train_file}\n"
                f"output_file={args.output_file}\n"
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
            f"Loaded {len(input_sentences)} sentences from {args.input_file}. "
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
                    if matching is None:
                        raise ValueError(
                            f"Response did not contain an aligned morph sequence for batch {batch_idx}."
                        )

                    predicted_morphs = flatten_morphs(batch_prediction)
                    for morph, row in zip(predicted_morphs, matching):
                        morph.etymology = normalize_etymology(row[2])
                        if morph.text.isdigit():
                            morph.etymology = []

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
                    pprint_sentences(prediction_sentences[: batch.end], args.output_file)
                    log_block(
                        log_stream,
                        f"Batch {batch_idx} checkpoint saved",
                        f"output_file={args.output_file}\nsentences_saved={batch.end}",
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

        print(f"Predictions written to: {args.output_file}")
        log_block(log_stream, "Run finished", f"predictions_file={args.output_file}")
    finally:
        if log_stream:
            log_stream.close()


if __name__ == "__main__":
    main()
