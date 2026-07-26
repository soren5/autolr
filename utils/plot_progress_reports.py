#!/usr/bin/env python3
"""Plot best-fitness and population-fitness progress reports.

The script expects AutoLR dump folders shaped like:

    dumps/<experiment_name>/run_<n>/_progress_report.csv

Each progress row is parsed as:

    generation best_fitness mean_population_fitness std_population_fitness other_metric
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROGRESS_FILE = "_progress_report.csv"
PARAMETERS_FILE = "_parameters.json"

# Keep this palette in sync with notebooks/make_plots.py without importing that
# module, because make_plots.py executes plot-building code at import time.
ACADEMIC_COLORS = [
    "#007191",
    "#62C8D3",
    "#F47A00",
    "#EF2D56",
    "#6D597A",
    "#372248",
    "#495867",
]

MULTI_TASK_THRESHOLDS = [
    ("FMNIST", "FMNIST_THRESHOLD", "FMNIST_CONFIG"),
    ("CIFAR10", "CIFAR10_THRESHOLD", "CIFAR10_CONFIG"),
    ("CIFAR100", "CIFAR100_THRESHOLD", "CIFAR100_CONFIG"),
    ("TinyImageNet", "TINY_IMAGENET_THRESHOLD", "TINY_IMAGENET_CONFIG"),
    ("TinyImageNet Custom", "TINY_IMAGENET_CUSTOM_THRESHOLD", "TINY_IMAGENET_CUSTOM_CONFIG"),
]


def parse_progress_report(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 4:
                continue
            rows.append(
                (
                    int(float(fields[0])),
                    float(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                )
            )
    if not rows:
        raise ValueError(f"No progress rows found in {path}")
    data = np.asarray(rows, dtype=float)
    return {
        "generation": data[:, 0].astype(int),
        "best_fitness": -data[:, 1],
        "population_fitness": -data[:, 2],
        "population_std": data[:, 3],
    }


def discover_experiments(root: Path) -> dict[str, list[Path]]:
    experiments = {}
    for experiment_dir in sorted(root.iterdir()):
        if not experiment_dir.is_dir() or experiment_dir.name in {"benchmarks", "logs"}:
            continue
        reports = sorted(
            experiment_dir.glob(f"run_*/{PROGRESS_FILE}"),
            key=lambda path: int(path.parent.name.split("_")[-1]),
        )
        if reports:
            experiments[experiment_dir.name] = reports
    return experiments


def align_runs(run_data: list[dict[str, np.ndarray]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    generations = sorted(set.union(*(set(data["generation"]) for data in run_data)))
    values = []
    for data in run_data:
        by_generation = dict(zip(data["generation"], data[metric]))
        values.append([by_generation.get(generation, np.nan) for generation in generations])
    return np.asarray(generations), np.asarray(values, dtype=float)


def load_concatenated_json(path: Path) -> list[dict]:
    """Read one or more adjacent JSON objects from AutoLR parameter logs."""
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


def load_thresholds_from_run_parameters(reports: list[Path]) -> dict[str, float]:
    for report in reports:
        parameter_path = report.parent / PARAMETERS_FILE
        if not parameter_path.exists():
            continue
        try:
            objects = load_concatenated_json(parameter_path)
        except (json.JSONDecodeError, OSError):
            continue
        for parameters in reversed(objects):
            thresholds = {}
            for _, key, config_key in MULTI_TASK_THRESHOLDS:
                config_value = parameters.get(config_key, parameters.get(config_key.lower()))
                if config_value in (None, "", False):
                    continue
                value = parameters.get(key, parameters.get(key.lower()))
                if value is not None:
                    thresholds[key] = float(value)
            if thresholds:
                return thresholds
    return {}


def load_thresholds_from_parameter_file(experiment: str) -> dict[str, float]:
    candidates = [
        Path("parameters") / f"{experiment}.yml",
        Path("parameters") / f"{experiment}.yaml",
        Path("parameters/archived_parameters") / f"{experiment}.yml",
        Path("parameters/archived_parameters") / f"{experiment}.yaml",
    ]
    pattern = re.compile(r"^\s*([A-Z0-9_]+_THRESHOLD)\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
    for path in candidates:
        if not path.exists():
            continue
        thresholds = {}
        for line in path.read_text().splitlines():
            match = pattern.match(line)
            if match:
                thresholds[match.group(1)] = float(match.group(2))
        if thresholds:
            return thresholds
    return {}


def threshold_lines_for_experiment(experiment: str, reports: list[Path]) -> list[tuple[str, float]]:
    if not experiment.startswith("multi"):
        return []

    thresholds = load_thresholds_from_run_parameters(reports)
    if not thresholds:
        thresholds = load_thresholds_from_parameter_file(experiment)

    lines = []
    cumulative_threshold = 0.0
    for label, key, _ in MULTI_TASK_THRESHOLDS:
        if key not in thresholds:
            continue
        cumulative_threshold += thresholds[key]
        lines.append((f"{label} cumulative gate", cumulative_threshold))
    return lines


def plot_metric(
    experiment: str,
    reports: list[Path],
    metric: str,
    output_dir: Path,
    threshold_lines: list[tuple[str, float]] | None = None,
) -> Path:
    run_data = [parse_progress_report(path) for path in reports]
    generations, values = align_runs(run_data, metric)
    mean_values = np.nanmean(values, axis=0)

    title_metric = "Best Fitness" if metric == "best_fitness" else "Population Fitness"
    output_path = output_dir / f"{experiment}_{metric}.png"

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for index, (data, run_values) in enumerate(zip(run_data, values)):
        ax.plot(
            data["generation"],
            data[metric],
            color=ACADEMIC_COLORS[1],
            alpha=0.28,
            linewidth=1.0,
            label="Runs" if index == 0 else None,
        )
    ax.plot(
        generations,
        mean_values,
        color=ACADEMIC_COLORS[0],
        linewidth=2.4,
        label=f"Mean across {len(reports)} runs",
    )
    for index, (label, value) in enumerate(threshold_lines or []):
        ax.axhline(
            value,
            color=ACADEMIC_COLORS[(index + 2) % len(ACADEMIC_COLORS)],
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
            label=f"{label}: {value:.3f}",
        )
    ax.set_title(f"{experiment}: {title_metric}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Inverted fitness (higher is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def reference_threshold_lines(experiments: dict[str, list[Path]]) -> list[tuple[str, float]]:
    for experiment, reports in experiments.items():
        lines = threshold_lines_for_experiment(experiment, reports)
        if lines:
            return lines
    return []


def no_multi_offset(experiments: dict[str, list[Path]]) -> float:
    for label, value in reference_threshold_lines(experiments):
        if label.startswith("CIFAR100 "):
            return value
    return 0.0


def plot_combined_metric(
    experiments: dict[str, list[Path]],
    metric: str,
    output_dir: Path,
) -> Path:
    title_metric = "Best Fitness" if metric == "best_fitness" else "Population Fitness"
    output_path = output_dir / f"all_setups_{metric}.png"
    offset = no_multi_offset(experiments)
    threshold_lines = reference_threshold_lines(experiments)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for index, (experiment, reports) in enumerate(experiments.items()):
        run_data = [parse_progress_report(path) for path in reports]
        generations, values = align_runs(run_data, metric)
        plotted_values = values.copy()
        label = experiment
        if experiment.startswith("no_multi"):
            plotted_values += offset
            label = f"{experiment} (+CIFAR100 gate)"
        mean_values = np.nanmean(plotted_values, axis=0)
        color = ACADEMIC_COLORS[index % len(ACADEMIC_COLORS)]
        ax.plot(
            generations,
            mean_values,
            color=color,
            linewidth=2.2,
            label=label,
        )
        lower = np.nanmin(plotted_values, axis=0)
        upper = np.nanmax(plotted_values, axis=0)
        ax.fill_between(
            generations,
            lower,
            upper,
            color=color,
            alpha=0.12,
            linewidth=0,
        )

    for index, (label, value) in enumerate(threshold_lines):
        ax.axhline(
            value,
            color=ACADEMIC_COLORS[(index + 2) % len(ACADEMIC_COLORS)],
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            label=f"{label}: {value:.3f}",
        )

    ax.set_title(f"All setups: {title_metric}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Inverted fitness on multi-task scalar axis")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Dump root to scan.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/progress_report_plots"),
        help="Directory where plots will be written.",
    )
    args = parser.parse_args()

    experiments = discover_experiments(args.root)
    if not experiments:
        raise SystemExit(f"No {PROGRESS_FILE} files found under {args.root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for experiment, reports in experiments.items():
        threshold_lines = threshold_lines_for_experiment(experiment, reports)
        written.append(plot_metric(experiment, reports, "best_fitness", args.output_dir, threshold_lines))
        written.append(plot_metric(experiment, reports, "population_fitness", args.output_dir, threshold_lines))
    written.append(plot_combined_metric(experiments, "best_fitness", args.output_dir))
    written.append(plot_combined_metric(experiments, "population_fitness", args.output_dir))

    print(f"Wrote {len(written)} plots to {args.output_dir.resolve()}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
