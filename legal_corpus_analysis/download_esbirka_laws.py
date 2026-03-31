#!/usr/bin/env python3
"""
Download Czech law texts from e-Sbirka into per-year directories.

Example:
    python3 esbirka_downloader/download_esbirka_laws.py --years 2020-2022 --limit-per-year 10
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import re
import sys
import time
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

import requests


API_BASE_URL = "https://www.e-sbirka.cz/sbr-externi"
DEFAULT_OUTPUT_DIR = Path("esbirka_downloader/data")
DEFAULT_COLLECTIONS = ("sb",)


@dataclasses.dataclass(frozen=True)
class DocumentRecord:
    year: int
    month: int
    date: str
    title: str
    code: str
    stale_url: str
    status: str
    collection_code: str
    set_code: str


@dataclasses.dataclass(frozen=True)
class DownloadedDocument:
    record: DocumentRecord
    canonical_stale_url: str
    text: str


@dataclasses.dataclass(frozen=True)
class SelectionOptions:
    collections: set[str]
    limit_per_year: int | None
    exclude_title_substrings: tuple[str, ...]
    include_kinds: set[str]
    exclude_kinds: set[str]


class XhtmlToTextParser(HTMLParser):
    BLOCK_TAGS = {
        "div",
        "p",
        "section",
        "article",
        "header",
        "footer",
        "li",
        "tr",
        "table",
        "blockquote",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "br":
            self.parts.append("\n")
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def get_text(self) -> str:
        text = "".join(self.parts)
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class EsbirkaClient:
    def __init__(self, timeout: float = 30.0, user_agent: str | None = None) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": user_agent or "MorphOrigin e-Sbirka downloader/1.0",
            }
        )
        self.timeout = timeout

    def list_year_documents(self, year: int) -> list[DocumentRecord]:
        response = self.session.post(
            f"{API_BASE_URL}/chronologicke-rejstriky/dokumenty-sbirky-po-mesicich",
            json={"rok": year},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        documents: list[DocumentRecord] = []
        for month_info in payload.get("dokumentySbirkyVMesici", []):
            month = month_info["mesic"]
            for set_info in month_info.get("sbirkyDokumentuSbirky", []):
                set_code = set_info["sadaDokumentuKod"]
                for document_info in set_info.get("dokumentySbirkyRejstrikInfo", []):
                    stale_url = document_info["staleUrl"]
                    documents.append(
                        DocumentRecord(
                            year=year,
                            month=month,
                            date=document_info["datum"],
                            title=document_info["nazev"],
                            code=document_info["kodDokumentuSbirky"],
                            stale_url=stale_url,
                            status=document_info["stavDokumentuSbirky"],
                            collection_code=stale_url.strip("/").split("/", 1)[0],
                            set_code=set_code,
                        )
                    )
        return documents

    def fetch_document_text(self, record: DocumentRecord) -> DownloadedDocument:
        encoded_stale_url = requests.utils.quote(record.stale_url, safe="")
        response = self.session.get(
            f"{API_BASE_URL}/dokumenty-sbirky/{encoded_stale_url}/fragmenty",
            params={"cisloStranky": 0},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        fragments = payload.get("seznam", [])
        canonical_stale_url = record.stale_url
        lines: list[str] = []

        for fragment in fragments:
            if fragment.get("staleUrl"):
                canonical_stale_url = fragment["staleUrl"].split("#", 1)[0]
            xhtml = fragment.get("xhtml")
            if not xhtml:
                continue
            text = self.xhtml_to_text(xhtml)
            if text:
                lines.append(text)

        merged_text = "\n".join(lines)
        merged_text = re.sub(r"\n{3,}", "\n\n", merged_text).strip()
        return DownloadedDocument(
            record=record,
            canonical_stale_url=canonical_stale_url,
            text=merged_text,
        )

    @staticmethod
    def xhtml_to_text(value: str) -> str:
        parser = XhtmlToTextParser()
        parser.feed(html.unescape(value))
        return parser.get_text()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Czech law texts from the e-Sbirka chronological register."
    )
    parser.add_argument(
        "--years",
        required=True,
        help="Inclusive year range in the form YYYY-YYYY, for example 2020-2022.",
    )
    parser.add_argument(
        "--limit-per-year",
        type=int,
        default=None,
        help="Maximum number of documents to save for each year. Default: all documents.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where downloaded files are stored. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--collections",
        default=",".join(DEFAULT_COLLECTIONS),
        help="Comma-separated top-level collections to include. Default: sb",
    )
    parser.add_argument(
        "--only-new-laws",
        action="store_true",
        help="Exclude amendment-like titles such as novela, změna, změní, mění.",
    )
    parser.add_argument(
        "--exclude-title-substring",
        action="append",
        default=[],
        help="Exclude documents whose normalized title contains this substring. Can be used multiple times.",
    )
    parser.add_argument(
        "--include-kind",
        action="append",
        default=[],
        help="Only include document kinds inferred from title, e.g. zakon, vyhlaska, narizeni, sdeleni. Can be used multiple times or as comma-separated values.",
    )
    parser.add_argument(
        "--exclude-kind",
        action="append",
        default=[],
        help="Exclude document kinds inferred from title, e.g. zakon, vyhlaska, narizeni, sdeleni. Can be used multiple times or as comma-separated values.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds. Default: 30",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Delay between document downloads in seconds. Default: 0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information.",
    )
    return parser.parse_args()


def parse_year_range(value: str) -> range:
    match = re.fullmatch(r"(\d{4})-(\d{4})", value.strip())
    if not match:
        raise ValueError(f"Invalid year range '{value}'. Expected YYYY-YYYY.")
    start_year = int(match.group(1))
    end_year = int(match.group(2))
    if start_year > end_year:
        raise ValueError(f"Invalid year range '{value}'. Start year must be <= end year.")
    return range(start_year, end_year + 1)


def parse_collections(value: str) -> set[str]:
    collections = {part.strip() for part in value.split(",") if part.strip()}
    if not collections:
        raise ValueError("At least one collection must be specified.")
    return collections


def normalize_for_matching(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_value = ascii_value.lower()
    ascii_value = re.sub(r"\s+", " ", ascii_value)
    return ascii_value.strip()


def build_exclude_title_substrings(
    raw_substrings: Iterable[str],
    only_new_laws: bool,
) -> tuple[str, ...]:
    substrings = [normalize_for_matching(value) for value in raw_substrings if value.strip()]
    if only_new_laws:
        substrings.extend(("novel", "zmen", "meni"))
    deduplicated: list[str] = []
    for substring in substrings:
        if substring and substring not in deduplicated:
            deduplicated.append(substring)
    return tuple(deduplicated)


def parse_kind_values(raw_values: Iterable[str]) -> set[str]:
    values: set[str] = set()
    for raw_value in raw_values:
        for part in raw_value.split(","):
            normalized = normalize_for_matching(part)
            if normalized:
                values.add(normalized)
    return values


def infer_document_kind(title: str) -> str:
    normalized = normalize_for_matching(title)
    if not normalized:
        return "unknown"

    prefix_map = (
        ("narizeni vlady", "narizeni"),
        ("rozhodnuti prezidenta republiky", "rozhodnuti"),
        ("rozhodnuti", "rozhodnuti"),
        ("vyhlaska", "vyhlaska"),
        ("zakon", "zakon"),
        ("sdeleni", "sdeleni"),
        ("nalez", "nalez"),
        ("usneseni", "usneseni"),
        ("oznameni", "oznameni"),
    )
    for prefix, kind in prefix_map:
        if normalized.startswith(prefix):
            return kind
    return normalized.split(" ", 1)[0]


def sanitize_filename_part(value: str, max_length: int = 120) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_value = ascii_value.lower()
    ascii_value = re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-")
    ascii_value = re.sub(r"-{2,}", "-", ascii_value)
    if not ascii_value:
        ascii_value = "document"
    return ascii_value[:max_length].rstrip("-")


def render_output_text(downloaded: DownloadedDocument) -> str:
    record = downloaded.record
    header = [
        f"# Code: {record.code}",
        f"# Title: {record.title}",
        f"# Date: {record.date}",
        f"# Status: {record.status}",
        f"# Original stale URL: {record.stale_url}",
        f"# Canonical stale URL: {downloaded.canonical_stale_url}",
        f"# Source: https://www.e-sbirka.cz{downloaded.canonical_stale_url}",
        "",
    ]
    return "\n".join(header) + downloaded.text + "\n"


def select_documents(
    documents: Iterable[DocumentRecord],
    options: SelectionOptions,
) -> list[DocumentRecord]:
    filtered: list[DocumentRecord] = []
    for document in documents:
        if document.collection_code not in options.collections:
            continue
        document_kind = infer_document_kind(document.title)
        if options.include_kinds and document_kind not in options.include_kinds:
            continue
        if document_kind in options.exclude_kinds:
            continue
        normalized_title = normalize_for_matching(document.title)
        if any(substring in normalized_title for substring in options.exclude_title_substrings):
            continue
        filtered.append(document)
    filtered.sort(key=lambda document: (document.date, document.code))

    if options.limit_per_year is not None:
        filtered = filtered[: options.limit_per_year]
    return filtered


def make_output_path(output_dir: Path, record: DocumentRecord) -> Path:
    year_dir = output_dir / str(record.year)
    code_part = sanitize_filename_part(record.code.replace(" ", "_"))
    title_part = sanitize_filename_part(record.title)
    filename = f"{code_part}__{title_part}.txt"
    return year_dir / filename


def download_year(
    client: EsbirkaClient,
    year: int,
    output_dir: Path,
    selection_options: SelectionOptions,
    overwrite: bool,
    sleep_seconds: float,
    verbose: bool,
) -> tuple[int, int, int]:
    year_documents = client.list_year_documents(year)
    selected_documents = select_documents(year_documents, selection_options)
    downloaded_count = 0
    skipped_existing_count = 0

    if verbose:
        print(
            f"Year {year}: found {len(year_documents)} documents, selected {len(selected_documents)}.",
            file=sys.stderr,
        )

    for index, record in enumerate(selected_documents, start=1):
        output_path = make_output_path(output_dir, record)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not overwrite:
            if verbose:
                print(f"Skipping existing file: {output_path}", file=sys.stderr)
            skipped_existing_count += 1
            continue

        downloaded = client.fetch_document_text(record)
        output_path.write_text(render_output_text(downloaded), encoding="utf-8")
        downloaded_count += 1

        if verbose:
            print(
                f"[{year} {index}/{len(selected_documents)}] Saved {record.code} -> {output_path}",
                file=sys.stderr,
            )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return len(selected_documents), downloaded_count, skipped_existing_count


def main() -> int:
    args = parse_args()

    try:
        years = parse_year_range(args.years)
        collections = parse_collections(args.collections)
        if args.limit_per_year is not None and args.limit_per_year < 1:
            raise ValueError("--limit-per-year must be at least 1.")
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2

    selection_options = SelectionOptions(
        collections=collections,
        limit_per_year=args.limit_per_year,
        exclude_title_substrings=build_exclude_title_substrings(
            args.exclude_title_substring,
            args.only_new_laws,
        ),
        include_kinds=parse_kind_values(args.include_kind),
        exclude_kinds=parse_kind_values(args.exclude_kind),
    )
    client = EsbirkaClient(timeout=args.timeout)
    total_selected = 0
    total_downloaded = 0
    total_skipped_existing = 0

    for year in years:
        try:
            selected_count, downloaded_count, skipped_existing_count = download_year(
                client=client,
                year=year,
                output_dir=args.output_dir,
                selection_options=selection_options,
                overwrite=args.overwrite,
                sleep_seconds=args.sleep_seconds,
                verbose=args.verbose,
            )
            total_selected += selected_count
            total_downloaded += downloaded_count
            total_skipped_existing += skipped_existing_count
        except requests.RequestException as exc:
            print(f"Failed while processing year {year}: {exc}", file=sys.stderr)
            return 1

    print(
        f"Selected {total_selected} documents across {len(list(years))} year(s); "
        f"downloaded {total_downloaded} file(s), skipped {total_skipped_existing} existing "
        f"file(s) in {args.output_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
