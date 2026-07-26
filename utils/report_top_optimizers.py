#!/usr/bin/env python3
"""Report top optimizers from AutoLR dump folders.

The script scans folders shaped like:

    dumps/<setup>/run_<n>/iteration_<generation>.json

It reports the top N unique smart phenotypes per setup, optional architecture
splits, and per-task rankings for multi-task setups.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.smart_phenotype import advanced_readable_phenotype


ITERATION_PATTERN = "iteration_*.json"
EXCLUDED_SETUP_DIRS = {"benchmarks", "logs"}
TASKS = ["fmnist", "cifar10", "cifar100", "tiny_imagenet", "tiny_imagenet_custom"]
DEFAULT_CIFAR100_CUMULATIVE_GATE = 0.829154 + 0.754643 + 0.556333
THRESHOLD_KEYS = ["FMNIST_THRESHOLD", "CIFAR10_THRESHOLD", "CIFAR100_THRESHOLD"]
ARCHITECTURAL_VARIABLES = [
    "strides",
    "kernel_size",
    "filters",
    "dilation_rate",
    "padding",
    "units",
    "pool_size",
    "layer_count",
    "layer_num",
]
ARCHITECTURAL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(variable) for variable in ARCHITECTURAL_VARIABLES) + r")\b"
)


@dataclass
class OptimizerRecord:
    setup: str
    run: int
    generation: int
    genetic_id: Any
    phen_id: str
    smart_phenotype: str
    phenotype: str | None
    advanced_readable_phenotype: str | None
    fitness: float | None
    score: float
    operation: str | None
    source_file: str
    has_architecture: bool
    is_evaluated: bool
    task_scores: dict[str, float] = field(default_factory=dict)


def generation_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def run_number_from_path(path: Path) -> int:
    return int(path.parent.name.split("_")[-1])


def load_iteration(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    return json.loads(text)


def discover_iteration_files(root: Path) -> dict[str, list[Path]]:
    setups: dict[str, list[Path]] = {}
    for setup_dir in sorted(root.iterdir()):
        if not setup_dir.is_dir() or setup_dir.name in EXCLUDED_SETUP_DIRS:
            continue
        files = sorted(
            setup_dir.glob(f"run_*/{ITERATION_PATTERN}"),
            key=lambda path: (run_number_from_path(path), generation_from_path(path)),
        )
        if files:
            setups[setup_dir.name] = files
    return setups


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def last_numeric(values: Any) -> float | None:
    if isinstance(values, list):
        for value in reversed(values):
            numeric = finite_float(value)
            if numeric is not None:
                return numeric
    return finite_float(values)


def extract_task_score(task_info: Any) -> float | None:
    if not isinstance(task_info, dict):
        return None
    for key in ["test_score", "fitness_score", "score", "accuracy", "val_accuracy"]:
        if key not in task_info:
            continue
        if key in {"accuracy", "val_accuracy"}:
            score = last_numeric(task_info[key])
        else:
            score = finite_float(task_info[key])
        if score is not None:
            return score
    return None


def extract_task_scores(setup: str, other_info: Any) -> dict[str, float]:
    if not isinstance(other_info, dict):
        return {}

    scores: dict[str, float] = {}
    multi_task = other_info.get("multi_task")
    if isinstance(multi_task, dict) and isinstance(multi_task.get("scores"), dict):
        for task, value in multi_task["scores"].items():
            score = finite_float(value)
            if score is not None:
                scores[task] = score

    for task in TASKS:
        score = extract_task_score(other_info.get(task))
        if score is not None:
            scores[task] = score
    if setup.startswith("no_multi") and "tiny_imagenet" not in scores:
        score = extract_task_score(other_info)
        if score is not None:
            scores["tiny_imagenet"] = score
    return scores


def make_advanced_readable(phenotype: Any) -> str | None:
    if not phenotype:
        return None
    try:
        return advanced_readable_phenotype(str(phenotype))
    except Exception as error:
        return f"<advanced_readable_phenotype failed: {type(error).__name__}: {error}>"


def make_phenotype_ids(setup_to_keys: dict[str, set[str]]) -> dict[tuple[str, str], str]:
    pairs = [
        (setup, smart_phenotype)
        for setup in sorted(setup_to_keys)
        for smart_phenotype in sorted(setup_to_keys[setup])
    ]
    return {
        pair: f"{pair[0]}_phen_id_{index}"
        for index, pair in enumerate(pairs)
    }


def collect_smart_keys(files_by_setup: dict[str, list[Path]]) -> dict[str, set[str]]:
    setup_to_keys: dict[str, set[str]] = {}
    for setup, files in files_by_setup.items():
        keys: set[str] = set()
        for path in files:
            for individual in load_iteration(path):
                smart_phenotype = individual.get("smart_phenotype") or individual.get("key")
                if smart_phenotype:
                    keys.add(str(smart_phenotype))
        setup_to_keys[setup] = keys
    return setup_to_keys


def collect_records(root: Path) -> list[OptimizerRecord]:
    files_by_setup = discover_iteration_files(root)
    phenotype_ids = make_phenotype_ids(collect_smart_keys(files_by_setup))
    records: list[OptimizerRecord] = []

    for setup, files in files_by_setup.items():
        for path in files:
            run = run_number_from_path(path)
            generation = generation_from_path(path)
            for individual in load_iteration(path):
                smart_phenotype = individual.get("smart_phenotype") or individual.get("key")
                fitness = finite_float(individual.get("fitness"))
                if not smart_phenotype:
                    continue
                smart_phenotype = str(smart_phenotype)
                is_evaluated = fitness is not None and fitness != 0.0
                task_scores = extract_task_scores(setup, individual.get("other_info"))
                if not is_evaluated:
                    task_scores = {task: 0.0 for task in task_scores}
                records.append(
                    OptimizerRecord(
                        setup=setup,
                        run=run,
                        generation=generation,
                        genetic_id=individual.get("id"),
                        phen_id=phenotype_ids[(setup, smart_phenotype)],
                        smart_phenotype=smart_phenotype,
                        phenotype=individual.get("phenotype"),
                        advanced_readable_phenotype=make_advanced_readable(
                            individual.get("phenotype")
                        ),
                        fitness=fitness,
                        score=-fitness if fitness is not None else 0.0,
                        operation=individual.get("operation"),
                        source_file=str(path),
                        has_architecture=bool(ARCHITECTURAL_PATTERN.search(smart_phenotype)),
                        is_evaluated=is_evaluated,
                        task_scores=task_scores,
                    )
                )
    apply_authoritative_task_scores(records)
    return records


def archive_identity(record: OptimizerRecord) -> tuple[str, int, str]:
    return record.setup, record.run, record.smart_phenotype


def apply_authoritative_task_scores(records: list[OptimizerRecord]) -> None:
    """Avoid inherited parent scores by canonicalizing task scores.

    Offspring can inherit ``other_info`` through parent deepcopy before their
    archive fitness is assigned. Multi-task rows use the first evaluated
    occurrence of an archive key as the safest task-level source available in
    iteration files. Single-task no-multi rows use ``-fitness`` directly because
    archive fitness is the authoritative F-race mean for that task.
    """
    canonical_scores: dict[tuple[str, int, str], dict[str, float]] = {}
    ordered_records = sorted(records, key=lambda record: (record.generation, record.source_file))

    for record in ordered_records:
        if not record.is_evaluated:
            continue
        if record.task_scores:
            canonical_scores.setdefault(archive_identity(record), dict(record.task_scores))

    for record in records:
        if not record.is_evaluated:
            record.task_scores = {}
        elif record.setup.startswith("no_multi"):
            record.task_scores = {"tiny_imagenet": record.score}
        else:
            record.task_scores = dict(canonical_scores.get(archive_identity(record), {}))


def load_concatenated_json(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    decoder = json.JSONDecoder()
    objects = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        obj, index = decoder.raw_decode(text, index)
        if isinstance(obj, dict):
            objects.append(obj)
    return objects


def infer_cifar100_cumulative_gate(root: Path) -> tuple[float, dict[str, float], str]:
    for parameter_path in sorted(root.glob("multi*/run_*/_parameters.json")):
        try:
            objects = load_concatenated_json(parameter_path)
        except (OSError, json.JSONDecodeError):
            continue
        for parameters in reversed(objects):
            thresholds = {}
            for key in THRESHOLD_KEYS:
                value = parameters.get(key, parameters.get(key.lower()))
                numeric = finite_float(value)
                if numeric is not None:
                    thresholds[key] = numeric
            if set(THRESHOLD_KEYS) <= set(thresholds):
                return (
                    sum(thresholds[key] for key in THRESHOLD_KEYS),
                    thresholds,
                    str(parameter_path),
                )
    thresholds = {
        "FMNIST_THRESHOLD": 0.829154,
        "CIFAR10_THRESHOLD": 0.754643,
        "CIFAR100_THRESHOLD": 0.556333,
    }
    return DEFAULT_CIFAR100_CUMULATIVE_GATE, thresholds, "default final-experiment thresholds"


def best_by_key(records: list[OptimizerRecord], score_fn) -> list[OptimizerRecord]:
    best: dict[tuple[str, str], OptimizerRecord] = {}
    best_score: dict[tuple[str, str], float] = {}
    for record in records:
        score = score_fn(record)
        if score is None:
            continue
        key = (record.setup, record.smart_phenotype)
        if key not in best or score > best_score[key]:
            best[key] = record
            best_score[key] = score
    return sorted(best.values(), key=lambda record: score_fn(record), reverse=True)


def top_n(records: list[OptimizerRecord], n: int, score_fn) -> list[OptimizerRecord]:
    return best_by_key(records, score_fn)[:n]


def is_multi_setup(setup: str, records: list[OptimizerRecord]) -> bool:
    return setup.startswith("multi") or any(len(record.task_scores) > 1 for record in records)


def is_architectural_setup(setup: str) -> bool:
    return "arch" in setup and "no_arch" not in setup


def task_score_total(record: OptimizerRecord) -> float | None:
    if not record.is_evaluated:
        return None
    if record.setup.startswith("no_multi"):
        return None
    if record.task_scores:
        return sum(record.task_scores.values())
    return None


def task_score(record: OptimizerRecord, task: str) -> float | None:
    if not record.is_evaluated:
        return None
    return record.task_scores.get(task)


def adjusted_score(record: OptimizerRecord, non_multi_offset: float) -> float | None:
    if not record.is_evaluated:
        return None
    if record.setup.startswith("no_multi"):
        tiny_score = record.task_scores.get("tiny_imagenet")
        return None if tiny_score is None else non_multi_offset + tiny_score
    return task_score_total(record)


def record_to_json(
    record: OptimizerRecord,
    criterion: str,
    criterion_score: float | None,
    non_multi_offset: float,
) -> dict[str, Any]:
    return {
        "phen_id": record.phen_id,
        "criterion": criterion,
        "criterion_score": criterion_score,
        "adjusted_score": adjusted_score(record, non_multi_offset),
        "task_score_total": task_score_total(record),
        "overall_score": record.score,
        "fitness": record.fitness,
        "setup": record.setup,
        "run": record.run,
        "generation": record.generation,
        "genetic_id": record.genetic_id,
        "is_evaluated": record.is_evaluated,
        "has_architecture": record.has_architecture,
        "task_scores": record.task_scores,
        "operation": record.operation,
        "smart_phenotype": record.smart_phenotype,
        "phenotype": record.phenotype,
        "advanced_readable_phenotype": record.advanced_readable_phenotype,
        "source_file": record.source_file,
    }


def render_record(
    record: OptimizerRecord,
    seen: set[str],
    criterion: str,
    criterion_score: float | None,
    non_multi_offset: float,
) -> str:
    if record.phen_id in seen:
        return f"- `{record.phen_id}`"
    seen.add(record.phen_id)
    lines = [
        f"- `{record.phen_id}`",
        f"  - criterion: `{criterion}` = `{criterion_score}`",
        f"  - adjusted_score: `{adjusted_score(record, non_multi_offset)}`",
        f"  - task_score_total: `{task_score_total(record)}`",
        f"  - overall_score: `{record.score}`; fitness: `{record.fitness}`",
        f"  - setup/run/genetic/generation: `{record.setup}` / `run_{record.run}` / `{record.genetic_id}` / `{record.generation}`",
        f"  - is_evaluated: `{record.is_evaluated}`",
        f"  - has_architecture: `{record.has_architecture}`",
    ]
    if record.task_scores:
        task_text = ", ".join(f"{task}={score}" for task, score in sorted(record.task_scores.items()))
        lines.append(f"  - task_scores: `{task_text}`")
    if record.operation:
        lines.append(f"  - operation/source: `{record.operation}` / `{Path(record.source_file).name}`")
    lines.append(f"  - smart_phenotype: `{record.smart_phenotype}`")
    if record.advanced_readable_phenotype:
        lines.extend(
            [
                "  - advanced_readable_phenotype:",
                "",
                "```text",
                record.advanced_readable_phenotype,
                "```",
            ]
        )
    if record.phenotype:
        lines.extend(
            [
                "  - phenotype:",
                "",
                "```python",
                record.phenotype,
                "```",
            ]
        )
    return "\n".join(lines)


def build_report(
    records: list[OptimizerRecord],
    n: int,
    non_multi_offset: float,
    threshold_source: str,
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], str]:
    by_setup: dict[str, list[OptimizerRecord]] = {}
    for record in records:
        by_setup.setdefault(record.setup, []).append(record)

    json_report: dict[str, Any] = {
        "top_n": n,
        "architectural_variables": ARCHITECTURAL_VARIABLES,
        "adjustment": {
            "non_multi_adjusted_score": "tiny_imagenet_score + cifar100_cumulative_gate",
            "multi_adjusted_score": "sum of recorded per-task fitness scores",
            "task_score_total": "sum of available recorded task fitness scores",
            "cifar100_cumulative_gate": non_multi_offset,
            "thresholds": thresholds,
            "threshold_source": threshold_source,
        },
        "setups": {},
    }
    markdown_lines = [
        f"# Top Optimizers Report",
        "",
        f"Top N: `{n}`",
        "",
        "Adjustment:",
        f"- non-multi adjusted score: `tiny_imagenet_score + {non_multi_offset}`",
        "- multi adjusted score: sum of recorded per-task fitness scores",
        "- task score total: sum of available recorded task fitness scores",
        f"- threshold source: `{threshold_source}`",
        "",
    ]
    seen_in_markdown: set[str] = set()

    adjusted_overall = top_n(
        records,
        n,
        lambda record: adjusted_score(record, non_multi_offset),
    )
    json_report["cross_setup_adjusted_overall"] = [
        record_to_json(
            record,
            "adjusted_score",
            adjusted_score(record, non_multi_offset),
            non_multi_offset,
        )
        for record in adjusted_overall
    ]
    markdown_lines.extend(["## Cross-Setup Adjusted Overall", ""])
    markdown_lines.extend(
        render_record(
            record,
            seen_in_markdown,
            "adjusted_score",
            adjusted_score(record, non_multi_offset),
            non_multi_offset,
        )
        for record in adjusted_overall
    )
    markdown_lines.append("")

    all_tasks = sorted({task for record in records for task in record.task_scores})
    cross_setup_per_task: dict[str, list[dict[str, Any]]] = {}
    for task in all_tasks:
        ranked = top_n(records, n, lambda record, task=task: task_score(record, task))
        cross_setup_per_task[task] = [
            record_to_json(
                record,
                f"{task}_score",
                task_score(record, task),
                non_multi_offset,
            )
            for record in ranked
        ]
        markdown_lines.extend([f"## Cross-Setup Task: {task}", ""])
        markdown_lines.extend(
            render_record(
                record,
                seen_in_markdown,
                f"{task}_score",
                task_score(record, task),
                non_multi_offset,
            )
            for record in ranked
        )
        markdown_lines.append("")
    json_report["cross_setup_per_task"] = cross_setup_per_task

    for setup in sorted(by_setup):
        setup_records = by_setup[setup]
        setup_report: dict[str, Any] = {}
        markdown_lines.extend([f"## {setup}", ""])

        adjusted = top_n(
            setup_records,
            n,
            lambda record: adjusted_score(record, non_multi_offset),
        )
        setup_report["adjusted_overall"] = [
            record_to_json(
                record,
                "adjusted_score",
                adjusted_score(record, non_multi_offset),
                non_multi_offset,
            )
            for record in adjusted
        ]
        markdown_lines.extend(["### Adjusted Overall", ""])
        markdown_lines.extend(
            render_record(
                record,
                seen_in_markdown,
                "adjusted_score",
                adjusted_score(record, non_multi_offset),
                non_multi_offset,
            )
            for record in adjusted
        )
        markdown_lines.append("")

        overall = top_n(setup_records, n, task_score_total)
        setup_report["overall"] = [
            record_to_json(
                record,
                "task_score_total",
                task_score_total(record),
                non_multi_offset,
            )
            for record in overall
        ]
        markdown_lines.extend(["### Task-Score Overall", ""])
        markdown_lines.extend(
            render_record(
                record,
                seen_in_markdown,
                "task_score_total",
                task_score_total(record),
                non_multi_offset,
            )
            for record in overall
        )
        markdown_lines.append("")

        if is_architectural_setup(setup):
            for label, predicate in [
                ("with_architecture", lambda record: record.has_architecture),
                ("without_architecture", lambda record: not record.has_architecture),
            ]:
                split_records = [record for record in setup_records if predicate(record)]
                ranked = top_n(
                    split_records,
                    n,
                    lambda record: adjusted_score(record, non_multi_offset),
                )
                setup_report[label] = [
                    record_to_json(
                        record,
                        "adjusted_score",
                        adjusted_score(record, non_multi_offset),
                        non_multi_offset,
                    )
                    for record in ranked
                ]
                markdown_lines.extend([f"### {label.replace('_', ' ').title()}", ""])
                if ranked:
                    markdown_lines.extend(
                        render_record(
                            record,
                            seen_in_markdown,
                            "adjusted_score",
                            adjusted_score(record, non_multi_offset),
                            non_multi_offset,
                        )
                        for record in ranked
                    )
                else:
                    markdown_lines.append("- none")
                markdown_lines.append("")

        per_task: dict[str, list[dict[str, Any]]] = {}
        tasks = sorted({task for record in setup_records for task in record.task_scores})
        for task in tasks:
            ranked = top_n(
                setup_records,
                n,
                lambda record, task=task: task_score(record, task),
            )
            per_task[task] = [
                record_to_json(
                    record,
                    f"{task}_score",
                    task_score(record, task),
                    non_multi_offset,
                )
                for record in ranked
            ]
            markdown_lines.extend([f"### Task: {task}", ""])
            if ranked:
                markdown_lines.extend(
                    render_record(
                        record,
                        seen_in_markdown,
                        f"{task}_score",
                        task_score(record, task),
                        non_multi_offset,
                    )
                    for record in ranked
                )
            else:
                markdown_lines.append("- none")
            markdown_lines.append("")
        if per_task:
            setup_report["per_task"] = per_task

        json_report["setups"][setup] = setup_report

    return json_report, "\n".join(markdown_lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Dumps folder to scan.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of optimizers per list.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/top_optimizers"),
        help="Directory where report files are written.",
    )
    args = parser.parse_args()

    if args.top_n <= 0:
        raise SystemExit("--top-n must be positive")
    if not args.root.exists():
        raise SystemExit(f"Root folder does not exist: {args.root}")

    records = collect_records(args.root)
    if not records:
        raise SystemExit(f"No optimizer records found under {args.root}")

    non_multi_offset, thresholds, threshold_source = infer_cifar100_cumulative_gate(args.root)
    json_report, markdown_report = build_report(
        records,
        args.top_n,
        non_multi_offset,
        threshold_source,
        thresholds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "top_optimizers_report.json"
    markdown_path = args.output_dir / "top_optimizers_report.md"
    json_path.write_text(json.dumps(json_report, indent=2, allow_nan=False))
    markdown_path.write_text(markdown_report)

    print(f"Read {len(records)} optimizer appearances from {args.root}")
    print(f"Using non-multi offset {non_multi_offset} from {threshold_source}")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
