#!/usr/bin/env python3
"""Create visualizations for F-race impact artifacts.

Example:
    python utils/visualize_f_race_impact.py --root dumps/my_experiment --output-dir analysis/f_race
"""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/autolr-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/autolr-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


F_RACE_SUMMARY = "_race_f_race_summary.csv"
SELECTION_SUMMARY = "_race_selection_audit_summary.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize F-race cost and decision impact.")
    parser.add_argument("--root", default="dumps", help="Dump root, experiment directory, or run directory to scan.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated CSV and PNG files.")
    parser.add_argument("--dpi", type=int, default=160, help="DPI for generated PNG files.")
    return parser.parse_args()


def find_run_dirs(root):
    root = Path(root)
    if (root / F_RACE_SUMMARY).is_file() or (root / SELECTION_SUMMARY).is_file():
        return [root]
    return sorted(
        path.parent
        for path in root.rglob(F_RACE_SUMMARY)
        if path.parent.is_dir()
    )


def run_metadata(run_dir, root):
    try:
        relative = run_dir.relative_to(root)
    except ValueError:
        relative = run_dir

    parts = relative.parts
    experiment_name = parts[-2] if len(parts) >= 2 else run_dir.parent.name
    run_number = parts[-1] if parts else run_dir.name
    return experiment_name, run_number, str(relative)


def load_csv(path):
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def load_generation_metrics(root):
    root = Path(root)
    rows = []
    for run_dir in find_run_dirs(root):
        experiment_name, run_number, scope = run_metadata(run_dir, root)
        race = load_csv(run_dir / F_RACE_SUMMARY)
        selection = load_csv(run_dir / SELECTION_SUMMARY)

        if race.empty and selection.empty:
            continue
        if race.empty:
            merged = selection
        elif selection.empty:
            merged = race
        else:
            merged = pd.merge(race, selection, on="generation", how="outer")

        merged["experiment_name"] = experiment_name
        merged["run_number"] = run_number
        merged["scope"] = scope
        rows.append(merged)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True, sort=False)
    return normalize_generation_metrics(df)


def normalize_generation_metrics(df):
    numeric_columns = [
        "generation",
        "eligible_count",
        "invalid_count",
        "extra_evaluations",
        "eliminated_count",
        "max_evals_hit_count",
        "winner_evals",
        "tournament_events",
        "tournament_changed",
        "tournament_changed_rate",
        "audit_unavailable_count",
    ]
    bool_columns = [
        "initial_best_changed",
        "elitism_changed",
        "elitism_order_changed",
    ]

    for column in numeric_columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    for column in bool_columns:
        if column in df:
            df[column] = df[column].apply(as_bool)

    if "generation" in df:
        df["generation"] = df["generation"].astype(int)
    if "extra_evaluations" not in df:
        df["extra_evaluations"] = 0
    if "tournament_changed_rate" not in df:
        df["tournament_changed_rate"] = 0
    if "initial_best_changed" not in df:
        df["initial_best_changed"] = False
    if "elitism_changed" not in df:
        df["elitism_changed"] = False
    if "stop_reason" not in df:
        df["stop_reason"] = "unknown"

    df["run_label"] = df["experiment_name"].astype(str) + "/" + df["run_number"].astype(str)
    return df.sort_values(["experiment_name", "run_number", "generation"])


def as_bool(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def summarize_runs(df):
    if df.empty:
        return pd.DataFrame()

    rows = []
    for (experiment_name, run_number), group in df.groupby(["experiment_name", "run_number"]):
        generations = len(group)
        tournament_events = group.get("tournament_events", pd.Series(dtype=float)).sum()
        tournament_changed = group.get("tournament_changed", pd.Series(dtype=float)).sum()
        best_changed = group["initial_best_changed"].sum()
        rows.append({
            "experiment_name": experiment_name,
            "run_number": run_number,
            "generations": generations,
            "extra_evaluations": group["extra_evaluations"].sum(),
            "best_changed_count": best_changed,
            "best_changed_rate": best_changed / generations if generations else 0,
            "tournament_events": tournament_events,
            "tournament_changed": tournament_changed,
            "tournament_changed_rate": tournament_changed / tournament_events if tournament_events else 0,
            "elitism_changed_count": group["elitism_changed"].sum(),
            "elitism_changed_rate": group["elitism_changed"].sum() / generations if generations else 0,
            "audit_unavailable_count": group.get("audit_unavailable_count", pd.Series(dtype=float)).sum(),
        })
    return pd.DataFrame(rows).sort_values(["experiment_name", "run_number"])


def save_outputs(df, output_dir, dpi):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_csv = output_dir / "_race_impact_generation_metrics.csv"
    summary_csv = output_dir / "_race_impact_run_summary.csv"
    df.to_csv(generation_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    summary = summarize_runs(df)
    summary.to_csv(summary_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    plot_extra_evaluations(df, output_dir / "_race_impact_extra_evaluations.png", dpi)
    plot_cumulative_extra_evaluations(df, output_dir / "_race_impact_cumulative_extra_evaluations.png", dpi)
    plot_best_changed(df, output_dir / "_race_impact_best_changed.png", dpi)
    plot_selection_changes(df, output_dir / "_race_impact_selection_changes.png", dpi)
    plot_stop_reasons(df, output_dir / "_race_impact_stop_reasons.png", dpi)
    plot_cost_vs_selection_change(df, output_dir / "_race_impact_cost_vs_selection_change.png", dpi)
    return summary


def setup_axes(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    return fig, ax


def save_fig(fig, path, dpi):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_extra_evaluations(df, path, dpi):
    fig, ax = setup_axes("F-race extra evaluations per generation", "Generation", "Extra evaluations")
    for label, group in df.groupby("run_label"):
        ax.plot(group["generation"], group["extra_evaluations"], marker="o", linewidth=1.4, markersize=3, label=label)
    add_legend(ax)
    save_fig(fig, path, dpi)


def plot_cumulative_extra_evaluations(df, path, dpi):
    fig, ax = setup_axes("Cumulative F-race extra evaluations", "Generation", "Cumulative extra evaluations")
    for label, group in df.groupby("run_label"):
        ordered = group.sort_values("generation")
        ax.plot(ordered["generation"], ordered["extra_evaluations"].cumsum(), linewidth=1.8, label=label)
    add_legend(ax)
    save_fig(fig, path, dpi)


def plot_best_changed(df, path, dpi):
    fig, ax = setup_axes("Generations where F-race changed the best candidate", "Generation", "Run")
    labels = sorted(df["run_label"].unique())
    label_to_y = {label: i for i, label in enumerate(labels)}
    changed = df[df["initial_best_changed"]]
    if changed.empty:
        ax.text(0.5, 0.5, "No initial-best changes recorded", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.scatter(changed["generation"], changed["run_label"].map(label_to_y), s=45)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    save_fig(fig, path, dpi)


def plot_selection_changes(df, path, dpi):
    fig, ax = setup_axes("Selection decisions changed by F-race", "Generation", "Tournament changed rate")
    for label, group in df.groupby("run_label"):
        ax.plot(group["generation"], group["tournament_changed_rate"], marker="o", linewidth=1.4, markersize=3, label=label)

    elitism_changed = df[df["elitism_changed"]]
    if not elitism_changed.empty:
        ax.scatter(
            elitism_changed["generation"],
            [1.02] * len(elitism_changed),
            marker="x",
            color="black",
            label="elitism changed",
        )
    ax.set_ylim(bottom=0, top=max(1.08, df["tournament_changed_rate"].max() * 1.1))
    add_legend(ax)
    save_fig(fig, path, dpi)


def plot_stop_reasons(df, path, dpi):
    counts = df["stop_reason"].fillna("unknown").value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(counts) + 2)))
    ax.barh(counts.index, counts.values)
    ax.set_title("F-race stop reasons")
    ax.set_xlabel("Generation count")
    ax.set_ylabel("Stop reason")
    ax.grid(True, axis="x", alpha=0.25)
    save_fig(fig, path, dpi)


def plot_cost_vs_selection_change(df, path, dpi):
    fig, ax = setup_axes("F-race cost vs parent-selection impact", "Extra evaluations", "Tournament changed rate")
    for label, group in df.groupby("run_label"):
        ax.scatter(group["extra_evaluations"], group["tournament_changed_rate"], s=35, alpha=0.75, label=label)
    add_legend(ax)
    save_fig(fig, path, dpi)


def add_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize="small")


def print_summary(summary, output_dir):
    if summary.empty:
        print("No F-race summary rows found.")
        return
    print(f"Wrote F-race impact outputs to: {output_dir}")
    columns = [
        "experiment_name",
        "run_number",
        "generations",
        "extra_evaluations",
        "best_changed_rate",
        "tournament_changed_rate",
        "elitism_changed_rate",
    ]
    print(summary[columns].to_string(index=False))


def main():
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root
    df = load_generation_metrics(root)
    if df.empty:
        print(f"No F-race summary files found under {root}")
        return 1

    summary = save_outputs(df, output_dir, args.dpi)
    print_summary(summary, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
