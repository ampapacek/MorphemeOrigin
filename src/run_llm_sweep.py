#!/usr/bin/env python3
import argparse
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SweepRun:
    model: str
    train_context_morphs: int
    run_name: str
    process_log_file: Path


@dataclass
class SweepResult:
    run: SweepRun
    returncode: int
    duration_sec: float
    command: list[str]


def parse_csv_list(value: str) -> list[str]:
    parts = [item.strip() for item in value.split(",")]
    return [item for item in parts if item]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in parse_csv_list(value)]


def sanitize_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return token or "unknown"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run llm_predict.py for a grid of models and train-context sizes."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="openai/gpt-5-mini,openai/gpt-5.2",
        help="Comma-separated model IDs to sweep.",
    )
    parser.add_argument(
        "--train_context_sizes",
        type=str,
        default="0,50,200",
        help="Comma-separated --train_context_morphs values to sweep.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel runs (1 = sequential).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run llm_predict.py.",
    )
    parser.add_argument(
        "--llm_script",
        type=str,
        default="src/llm_predict.py",
        help="Path to llm_predict.py.",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="outputs/llm_sweep_results.tsv",
        help="Common TSV file where one line per successful run is appended.",
    )
    parser.add_argument(
        "--run_log_dir",
        type=str,
        default="outputs/llm_sweep_logs",
        help="Directory for per-run subprocess logs.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop launching new runs after the first failure (sequential mode only).",
    )
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    return args, forwarded


def build_runs(
    models: list[str],
    train_sizes: list[int],
    run_log_dir: Path,
) -> list[SweepRun]:
    runs: list[SweepRun] = []
    for model in models:
        for train_size in train_sizes:
            run_name = f"{sanitize_token(model)}_trainm_{train_size}"
            runs.append(
                SweepRun(
                    model=model,
                    train_context_morphs=train_size,
                    run_name=run_name,
                    process_log_file=run_log_dir / f"{run_name}.log",
                )
            )
    return runs


def build_command(args: argparse.Namespace, forwarded: list[str], run: SweepRun) -> list[str]:
    cmd = [
        args.python,
        args.llm_script,
        *forwarded,
        "--model",
        run.model,
        "--train_context_morphs",
        str(run.train_context_morphs),
        "--eval_results_file",
        str(args.summary_file),
    ]
    if "--log_file" not in forwarded:
        cmd.extend(["--log_file", str(run.process_log_file)])
    return cmd


def run_one(repo_root: Path, command: list[str], run: SweepRun) -> SweepResult:
    run.process_log_file.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    with open(run.process_log_file, "wt", encoding="utf-8") as log_f:
        log_f.write(f"$ {shlex.join(command)}\n\n")
        log_f.flush()
        completed = subprocess.run(
            command,
            cwd=repo_root,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    duration = time.perf_counter() - start
    return SweepResult(
        run=run,
        returncode=completed.returncode,
        duration_sec=duration,
        command=command,
    )

def main() -> None:
    args, forwarded = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    models = parse_csv_list(args.models)
    train_sizes = parse_csv_ints(args.train_context_sizes)
    parallel = max(1, args.parallel)

    if not models:
        raise ValueError("No models provided.")
    if not train_sizes:
        raise ValueError("No train_context_sizes provided.")
    if len(models) * len(train_sizes) > 1 and "--output_file" in forwarded:
        raise ValueError("Do not pass --output_file when running multiple sweep jobs.")
    if len(models) * len(train_sizes) > 1 and "--mistakes_file" in forwarded:
        raise ValueError("Do not pass --mistakes_file when running multiple sweep jobs.")

    summary_file = Path(args.summary_file)
    run_log_dir = Path(args.run_log_dir)
    runs = build_runs(models, train_sizes, run_log_dir=run_log_dir)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Planned runs: {len(runs)} (parallel={parallel})")
    for run in runs:
        command = build_command(args, forwarded, run)
        print(f"- {run.run_name}: {shlex.join(command)}")

    if args.dry_run:
        print("Dry run complete.")
        return

    results: list[SweepResult] = []
    if parallel == 1:
        for run in runs:
            command = build_command(args, forwarded, run)
            print(f"Running: {run.run_name}")
            result = run_one(repo_root, command, run)
            results.append(result)
            status = "OK" if result.returncode == 0 else "FAIL"
            print(f"Finished: {run.run_name} [{status}] in {result.duration_sec:.1f}s")
            if args.fail_fast and result.returncode != 0:
                break
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            future_map = {}
            for run in runs:
                command = build_command(args, forwarded, run)
                future = pool.submit(run_one, repo_root, command, run)
                future_map[future] = run
            for future in as_completed(future_map):
                result = future.result()
                results.append(result)
                status = "OK" if result.returncode == 0 else "FAIL"
                print(f"Finished: {result.run.run_name} [{status}] in {result.duration_sec:.1f}s")

    ok_count = sum(1 for r in results if r.returncode == 0)
    fail_count = sum(1 for r in results if r.returncode != 0)

    print(f"Runs finished: ok={ok_count}, failed={fail_count}")
    print(f"Evaluation summary file: {summary_file}")
    print(f"Run logs directory: {run_log_dir}")

    if fail_count > 0:
        failed_names = [r.run.run_name for r in results if r.returncode != 0]
        print("Failed runs:")
        for name in failed_names:
            print(f"- {name}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
