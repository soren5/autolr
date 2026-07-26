#!/usr/bin/env python3
"""Analyze Chapter 4 AFM/PSGE rerun dumps.

This script reads PSGE-style ``progress_report.csv`` files for the Quartic,
Pagie, and Boston Housing AFM reruns. It writes descriptive summaries,
Kruskal-Wallis tests, and pairwise Mann-Whitney U tests for the
final-generation results.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats


TASKS = {
    "quartic": "Quartic",
    "pagie": "Pagie",
    "bh": "Boston Housing",
}

METHODS = {
    "base": "SM+SG",
    "fgg": "SM+FGG",
    "afm": "AFM+SG",
    "afm+fgg": "AFM+FGG",
}

@dataclass(frozen=True)
class ProgressRow:
    generation: int
    best_fitness: float
    mean_population_fitness: float
    std_population_fitness: float
    test_error: float


def parse_progress_report(path: Path) -> list[ProgressRow]:
    rows: list[ProgressRow] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 5:
                raise ValueError(f"Malformed progress row in {path}: {line!r}")
            rows.append(
                ProgressRow(
                    generation=int(float(fields[0])),
                    best_fitness=float(fields[1]),
                    mean_population_fitness=float(fields[2]),
                    std_population_fitness=float(fields[3]),
                    test_error=float(fields[4]),
                )
            )
    if not rows:
        raise ValueError(f"No progress rows found in {path}")
    return rows


def iter_run_reports(root: Path, task: str, method: str) -> Iterable[tuple[int, Path]]:
    method_root = root / task / method / "1.0"
    for run in range(1, 31):
        path = method_root / f"run_{run}" / "progress_report.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        yield run, path


def load_runs(root: Path) -> dict[str, dict[str, dict[int, list[ProgressRow]]]]:
    data: dict[str, dict[str, dict[int, list[ProgressRow]]]] = {}
    for task in TASKS:
        data[task] = {}
        for method in METHODS:
            data[task][method] = {}
            for run, path in iter_run_reports(root, task, method):
                data[task][method][run] = parse_progress_report(path)
    return data


def summarize(values: list[float] | np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def final_values(
    data: dict[str, dict[str, dict[int, list[ProgressRow]]]],
    task: str,
    method: str,
    metric: str,
) -> list[float]:
    values = []
    for rows in data[task][method].values():
        final = rows[-1]
        if metric == "train":
            values.append(final.best_fitness)
        elif metric == "test":
            values.append(final.test_error)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return values


def write_summary_csv(data: dict, output_dir: Path) -> None:
    path = output_dir / "chapter4_afm_final_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "problem",
                "metric",
                "method",
                "n",
                "mean",
                "median",
                "std",
                "iqr",
                "min",
                "max",
            ],
        )
        writer.writeheader()
        for task, task_label in TASKS.items():
            metrics = ["train", "test"] if task == "bh" else ["train"]
            for metric in metrics:
                for method, method_label in METHODS.items():
                    row = {
                        "problem": task_label,
                        "metric": metric,
                        "method": method_label,
                        **summarize(final_values(data, task, method, metric)),
                    }
                    writer.writerow(row)


def write_kruskal_csv(data: dict, output_dir: Path) -> None:
    path = output_dir / "chapter4_afm_kruskal_tests.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["problem", "metric", "h_statistic", "p_value"],
        )
        writer.writeheader()
        for task, task_label in TASKS.items():
            metrics = ["train", "test"] if task == "bh" else ["train"]
            for metric in metrics:
                samples = [final_values(data, task, method, metric) for method in METHODS]
                h_statistic, p_value = stats.kruskal(*samples)
                writer.writerow(
                    {
                        "problem": task_label,
                        "metric": metric,
                        "h_statistic": h_statistic,
                        "p_value": p_value,
                    }
                )


def write_pairwise_csv(data: dict, output_dir: Path) -> None:
    path = output_dir / "chapter4_afm_pairwise_mannwhitney.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "problem",
                "metric",
                "method_a",
                "method_b",
                "u_statistic",
                "p_value",
                "mean_a",
                "mean_b",
                "median_a",
                "median_b",
                "lower_mean_method",
                "lower_median_method",
            ],
        )
        writer.writeheader()
        methods = list(METHODS.items())
        for task, task_label in TASKS.items():
            metrics = ["train", "test"] if task == "bh" else ["train"]
            for metric in metrics:
                for index_a in range(len(methods)):
                    for index_b in range(index_a + 1, len(methods)):
                        method_a, label_a = methods[index_a]
                        method_b, label_b = methods[index_b]
                        values_a = np.asarray(final_values(data, task, method_a, metric))
                        values_b = np.asarray(final_values(data, task, method_b, metric))
                        u_statistic, p_value = stats.mannwhitneyu(
                            values_a,
                            values_b,
                            alternative="two-sided",
                        )
                        mean_a = float(np.mean(values_a))
                        mean_b = float(np.mean(values_b))
                        median_a = float(np.median(values_a))
                        median_b = float(np.median(values_b))
                        writer.writerow(
                            {
                                "problem": task_label,
                                "metric": metric,
                                "method_a": label_a,
                                "method_b": label_b,
                                "u_statistic": u_statistic,
                                "p_value": p_value,
                                "mean_a": mean_a,
                                "mean_b": mean_b,
                                "median_a": median_a,
                                "median_b": median_b,
                                "lower_mean_method": label_a if mean_a < mean_b else label_b,
                                "lower_median_method": label_a if median_a < median_b else label_b,
                            }
                        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/Users/soren/Work/cenas_do_pedro/psge/sge/dumps"),
        help="Root containing task/setup/1.0/run_* PSGE dumps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/chapter4_afm_rerun_analysis"),
        help="Directory where CSV statistical outputs will be written.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_runs(args.root)
    write_summary_csv(data, output_dir)
    write_kruskal_csv(data, output_dir)
    write_pairwise_csv(data, output_dir)

    print(f"Wrote Chapter 4 AFM rerun statistical analysis to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
