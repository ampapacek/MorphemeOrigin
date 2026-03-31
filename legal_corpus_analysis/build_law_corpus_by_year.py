#!/usr/bin/env python3
"""
Merge downloaded law files into one text file with a per-year word budget.

Output format:
- # Year YYYY
- # Law <title>, <code>, <link>
- excerpt text

The script skips the metadata header stored at the top of each source file.
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import re
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-zÀ-ž]+(?:[-'][A-Za-zÀ-ž]+)*", re.UNICODE)
LONG_NUMBER_RE = re.compile(r"\b\d{4,}\b")
PAREN_REFERENCE_RE = re.compile(r"\([^)]*(?:§|\d)[^)]*\)")
SECTION_REFERENCE_RE = re.compile(r"§+\s*[\dA-Za-zÀ-ž—–\-/,.\s]*")
LAW_NUMBER_REFERENCE_RE = re.compile(r"č\.\s*[^,;)\n]+", re.IGNORECASE)
ONLY_LETTERS_RE = re.compile(r"[^A-Za-zÀ-ž\s]+", re.UNICODE)
CITATION_ABBREVIATIONS = {
    "cl",
    "cis",
    "č",
    "odst",
    "pism",
    "písm",
    "par",
    "tr",
    "voj",
    "ř",
}
SECTION_MARKER_RE = re.compile(
    r"^(?:§+\s*\d+[A-Za-z]*\.?|čl\.\s*[IVXLCDM\d]+\.?|\(\d+\)|\d+[.)]|[a-z]\))$",
    re.IGNORECASE,
)
SIGNATURE_RE = re.compile(r"v\.\s*r\.|m\.\s*p\.", re.IGNORECASE)
ALL_CAPS_LINE_RE = re.compile(r"^[^a-zá-ž]*[A-ZÁ-Ž][^a-zá-ž]*$")
LIST_ITEM_PREFIX_RE = re.compile(r"^[A-ZÁ-Ž]\.:\s")
DOT_PLACEHOLDER = "__DOT__"
ABBREVIATION_PATTERNS = (
    r"\bsp\.\s*zn\.",
    r"\bv\.\s*r\.",
    r"\bm\.\s*p\.",
    r"\bčl\.",
    r"\bč\.",
    r"\bodst\.",
    r"\bpísm\.",
    r"\bresp\.",
    r"\bpopř\.",
    r"\bnapř\.",
    r"\btj\.",
    r"\btzn\.",
    r"\batd\.",
    r"\bapod\.",
    r"\bSb\.",
    r"\bIng\.",
    r"\bJUDr\.",
    r"\bMgr\.",
    r"\bBc\.",
    r"\bPhDr\.",
    r"\bMUDr\.",
    r"\bRNDr\.",
    r"\bMVDr\.",
    r"\bDr\.",
    r"\bprof\.",
    r"\bdoc\.",
    r"\blit\.",
    r"\btr\.",
    r"\bz\.",
    r"\ba\s+n\.",
    r"\bmin\.",
    r"\bčís\.",
    r"\bAl\.",
    r"\bFr\.",
    r"\bA\.",
)

@dataclasses.dataclass(frozen=True)
class LawDocument:
    path: Path
    year: int
    code: str
    title: str
    source_url: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge law texts into one file with a fixed number of words per year."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory with per-year subdirectories containing .txt law files.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path of the merged output file.",
    )
    parser.add_argument(
        "--words-per-year",
        type=int,
        required=True,
        help="Maximum number of words to include from each year.",
    )
    parser.add_argument(
        "--sample-by",
        choices=("document", "sentence"),
        default="document",
        help=(
            "How to build the yearly sample. "
            "'document' keeps the current behavior, "
            "'sentence' samples individual sentences without law headers."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --sample-by sentence is selected.",
    )
    parser.add_argument(
        "--exclude-punctuation",
        action="store_true",
        help=(
            "Remove punctuation characters from sentence-mode output lines after sampling. "
            "Year divider lines are kept unchanged."
        ),
    )
    return parser.parse_args()


def parse_law_file(path: Path) -> LawDocument:
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata: dict[str, str] = {}
    body_start_index = 0

    for index, line in enumerate(lines):
        if not line.startswith("# "):
            body_start_index = index
            break
        key_value = line[2:].split(":", 1)
        if len(key_value) == 2:
            key, value = key_value
            metadata[key.strip()] = value.strip()
    else:
        body_start_index = len(lines)

    text = "\n".join(lines[body_start_index:]).strip()
    year = int(path.parent.name)

    return LawDocument(
        path=path,
        year=year,
        code=metadata.get("Code", path.stem),
        title=metadata.get("Title", path.stem),
        source_url=metadata.get("Source", ""),
        text=text,
    )


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def truncate_to_word_limit(text: str, word_limit: int) -> str:
    if word_limit <= 0:
        return ""

    matches = list(WORD_RE.finditer(text))
    if len(matches) <= word_limit:
        return text.strip()

    cutoff = matches[word_limit - 1].end()
    return text[:cutoff].rstrip()


def sanitize_excerpt_text(text: str) -> str:
    cleaned_lines: list[str] = []

    for line in text.splitlines():
        line = PAREN_REFERENCE_RE.sub(" ", line)
        line = SECTION_REFERENCE_RE.sub(" ", line)
        line = LAW_NUMBER_REFERENCE_RE.sub(" ", line)
        line = LONG_NUMBER_RE.sub("", line)
        line = ONLY_LETTERS_RE.sub(" ", line)
        tokens = [token for token in line.split() if token.lower() not in CITATION_ABBREVIATIONS]
        line = " ".join(tokens).strip()

        if not line:
            continue
        if not re.search(r"[A-Za-zÀ-ž]", line):
            continue
        cleaned_lines.append(line)

    sanitized = "\n".join(cleaned_lines)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized.strip()


def load_documents(input_dir: Path) -> list[LawDocument]:
    documents: list[LawDocument] = []
    for path in sorted(input_dir.rglob("*.txt")):
        try:
            int(path.parent.name)
        except ValueError:
            continue
        documents.append(parse_law_file(path))
    return documents


def is_heading_like_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.isdigit():
        return True
    if SECTION_MARKER_RE.fullmatch(stripped):
        return True

    word_count = count_words(stripped)
    if SIGNATURE_RE.search(stripped):
        return True
    if word_count == 0:
        return True
    if word_count <= 8 and not re.search(r"[.!?]", stripped):
        if ALL_CAPS_LINE_RE.fullmatch(stripped):
            return True
        if not re.search(r"[,;:]", stripped):
            return True
    return False


def normalize_text_for_sentence_split(text: str) -> str:
    kept_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if is_heading_like_line(stripped):
            continue
        kept_lines.append(stripped)
    return re.sub(r"\s+", " ", " ".join(kept_lines)).strip()


def mask_abbreviations(text: str) -> str:
    masked = text
    for pattern in ABBREVIATION_PATTERNS:
        masked = re.sub(
            pattern,
            lambda match: match.group(0).replace(".", DOT_PLACEHOLDER),
            masked,
            flags=re.IGNORECASE,
        )
    return masked


def clean_sentence(sentence: str) -> str:
    sentence = sentence.replace(DOT_PLACEHOLDER, ".")
    sentence = re.sub(
        r"^(?:§+\s*\d+[A-Za-z]*\.?\s*|čl\.\s*[IVXLCDM\d]+\.?\s*|\(\d+\)\s*|\d+[.)]\s*|[a-z]\)\s*)+",
        "",
        sentence,
        flags=re.IGNORECASE,
    )
    sentence = re.sub(r"([.!?])\s+\d{1,3}\.$", r"\1", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip(" \t\r\n-–—,;:")
    return sentence.strip()


def format_sentence_output(sentence: str, exclude_punctuation: bool) -> str:
    if not exclude_punctuation:
        return sentence

    without_punctuation = re.sub(r"[^\w\s]", " ", sentence, flags=re.UNICODE)
    without_punctuation = without_punctuation.replace("_", " ")
    return re.sub(r"\s+", " ", without_punctuation).strip()


def extract_sentences(text: str) -> list[str]:
    normalized = normalize_text_for_sentence_split(text)
    if not normalized:
        return []

    masked = mask_abbreviations(normalized)
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=(?:[A-ZÁ-Ž„"(\']))', masked)

    sentences: list[str] = []
    for raw_sentence in raw_sentences:
        sentence = clean_sentence(raw_sentence)
        word_count = count_words(sentence)
        first_alpha = next((char for char in sentence if char.isalpha()), "")
        if word_count < 5:
            continue
        if LIST_ITEM_PREFIX_RE.match(sentence):
            continue
        if SIGNATURE_RE.search(sentence):
            continue
        if first_alpha and first_alpha.islower():
            continue
        if not re.search(r"[.!?]$", sentence):
            continue
        if re.search(r"\bze dne \d{1,2}\.$", sentence, re.IGNORECASE):
            continue
        if re.search(r"\b(?:do\s+)?dne \d{1,3}\.$", sentence, re.IGNORECASE):
            continue
        if ":" in sentence and re.search(r"\b[A-Za-zÀ-ž]{1,4}\.$", sentence):
            continue
        if ALL_CAPS_LINE_RE.fullmatch(sentence) and word_count <= 12:
            continue
        sentences.append(sentence)
    return sentences


def build_document_merged_output(
    documents: list[LawDocument], words_per_year: int
) -> tuple[str, int]:
    documents_by_year: dict[int, list[LawDocument]] = {}
    for document in documents:
        documents_by_year.setdefault(document.year, []).append(document)

    output_parts: list[str] = []
    used_laws = 0

    for year in sorted(documents_by_year):
        remaining_words = words_per_year
        if remaining_words <= 0:
            continue

        year_parts = [f"# Year {year}"]
        for document in documents_by_year[year]:
            if remaining_words <= 0:
                break

            excerpt = truncate_to_word_limit(document.text, remaining_words)
            excerpt = sanitize_excerpt_text(excerpt)
            excerpt_word_count = count_words(excerpt)
            if excerpt_word_count == 0:
                continue

            year_parts.append(f"# Law {document.title}, {document.code}, {document.source_url}")
            year_parts.append(excerpt)
            remaining_words -= excerpt_word_count
            used_laws += 1

        if len(year_parts) > 1:
            output_parts.append("\n\n".join(year_parts))

    return "\n\n".join(output_parts).strip() + "\n", used_laws


def sample_sentences_for_year(
    sentences: list[str], words_per_year: int, rng: random.Random
) -> tuple[list[str], int]:
    shuffled = list(sentences)
    rng.shuffle(shuffled)

    selected: list[str] = []
    total_words = 0
    fallback_sentence: str | None = None
    fallback_word_count: int | None = None

    for sentence in shuffled:
        sentence_word_count = count_words(sentence)
        if sentence_word_count == 0:
            continue

        if total_words >= words_per_year:
            break

        proposed_total = total_words + sentence_word_count
        if total_words == 0 or proposed_total <= words_per_year:
            selected.append(sentence)
            total_words = proposed_total
            continue

        overshoot = proposed_total - words_per_year
        remaining = words_per_year - total_words
        if overshoot <= remaining:
            selected.append(sentence)
            total_words = proposed_total
            break

        if fallback_sentence is None or sentence_word_count < fallback_word_count:
            fallback_sentence = sentence
            fallback_word_count = sentence_word_count

    if fallback_sentence is not None and total_words < words_per_year:
        proposed_total = total_words + fallback_word_count
        if proposed_total - words_per_year < words_per_year - total_words:
            selected.append(fallback_sentence)
            total_words = proposed_total

    return selected, total_words


def build_sentence_merged_output(
    documents: list[LawDocument],
    words_per_year: int,
    seed: int,
    exclude_punctuation: bool,
) -> tuple[str, int]:
    documents_by_year: dict[int, list[LawDocument]] = {}
    for document in documents:
        documents_by_year.setdefault(document.year, []).append(document)

    output_parts: list[str] = []
    used_sentences = 0

    for year in sorted(documents_by_year):
        year_sentences: list[str] = []
        for document in documents_by_year[year]:
            year_sentences.extend(extract_sentences(document.text))

        if not year_sentences:
            continue

        year_rng = random.Random(f"{seed}:{year}")
        selected_sentences, _ = sample_sentences_for_year(
            year_sentences, words_per_year, year_rng
        )
        if not selected_sentences:
            continue

        rendered_sentences = [
            format_sentence_output(sentence, exclude_punctuation)
            for sentence in selected_sentences
        ]
        output_parts.append("\n\n".join((f"# Year {year}", "\n".join(rendered_sentences))))
        used_sentences += len(selected_sentences)

    return "\n\n".join(output_parts).strip() + "\n", used_sentences


def main() -> int:
    args = parse_args()
    if args.words_per_year < 1:
        raise SystemExit("--words-per-year must be at least 1")

    documents = load_documents(args.input_dir)
    if not documents:
        raise SystemExit(f"No .txt files found in {args.input_dir}")

    if args.sample_by == "sentence":
        merged_output, used_units = build_sentence_merged_output(
            documents, args.words_per_year, args.seed, args.exclude_punctuation
        )
        unit_label = "Sentences used"
    else:
        merged_output, used_units = build_document_merged_output(
            documents, args.words_per_year
        )
        unit_label = "Laws used"

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(merged_output, encoding="utf-8")

    print(
        f"Wrote merged file to {args.output_file} "
        f"using up to {args.words_per_year} words per year. "
        f"{unit_label}: {used_units}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
