"""Reproduce the historical AutoLR speed-up mechanism plots.

This script restores the plotting logic from the historical notebook
``utils/evolutionary_optimization_visualization.ipynb`` at commit 569b7ad1.
It is intentionally read-only: recovered dump JSON files are never modified.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


CLASS_LABELS = ["strong optimizer", "invalid optimizer"]
DEFAULT_MECHANISMS = ["archive", "invalid detection", "degenerate detection"]


def extract_generation(path):
    match = re.search(r"iteration_(\d+)", str(path))
    if not match:
        return None
    return int(match.group(1))


def load_iteration_json(path):
    content = path.read_text()
    content = content.replace("NaN", "0.0")
    return json.loads(content)


def normalize_source(source):
    if source == "sanity check":
        return "degenerate detection"
    return source


def load_results_read_only(root_dir):
    rows = []
    root_dir = Path(root_dir)
    for experiment_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in experiment_dir.iterdir() if p.is_dir()):
            for json_path in sorted(run_dir.glob("iteration_*.json"), key=extract_generation):
                generation = extract_generation(json_path)
                try:
                    data = load_iteration_json(json_path)
                except Exception as exc:
                    print(f"Skipping {json_path}: {type(exc).__name__}: {exc}")
                    continue
                for individual in data:
                    if not isinstance(individual, dict):
                        continue
                    other_info = individual.get("other_info", {}) or {}
                    source = individual.get("source", other_info.get("source", "Unknown"))
                    duration = individual.get("duration", other_info.get("duration", 0.0))
                    rows.append(
                        {
                            "Experiment name": experiment_dir.name,
                            "Run number": run_dir.name,
                            "Individual number": individual.get("id"),
                            "Generation": generation,
                            "Phenotype": individual.get("phenotype"),
                            "Smart Phenotype": individual.get("smart_phenotype"),
                            "Fitness": individual.get("fitness"),
                            "Duration": duration,
                            "Source": normalize_source(source),
                        }
                    )
    return pd.DataFrame(rows)


def plot_sources_per_generation(df, output_dir):
    if df.empty:
        raise ValueError("No iteration rows were loaded from the dump root.")

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()
    plot_df["Generation"] = pd.Categorical(
        plot_df["Generation"],
        categories=sorted(plot_df["Generation"].dropna().unique()),
        ordered=True,
    )

    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.despine(fig)
    sns.histplot(
        data=plot_df,
        x="Generation",
        hue="Source",
        multiple="stack",
        edgecolor=".3",
        linewidth=.5,
        ax=ax,
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of Sources by Generation")
    fig.tight_layout()
    fig.savefig(output_dir / "methods.png", dpi=150)
    fig.savefig(output_dir / "mechanisms.pdf")
    plt.close(fig)


def load_full_evaluation_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "source" not in df.columns and "Source" in df.columns:
        df["source"] = df["Source"]
    if "source" not in df.columns:
        raise ValueError("Full-evaluation CSV must contain either 'source' or 'Source'.")

    required = ["Fitness", "Full Fitness", "Full Duration"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Full-evaluation CSV is missing required columns: {missing}")

    df = df[df["Full Duration"] != 0.0].copy()
    df["source"] = df["source"].apply(normalize_source)
    return df


def classify_optimizers(df, invalid_threshold):
    classified = df.copy()
    classified["Optimizer Class"] = classified["Full Fitness"].apply(
        lambda fitness: "invalid optimizer"
        if fitness <= invalid_threshold
        else "strong optimizer"
    )
    classified["Predicted Optimizer Class"] = classified["Fitness"].apply(
        lambda fitness: "invalid optimizer"
        if fitness <= invalid_threshold
        else "strong optimizer"
    )

    # Historical notebook correction. It happened after the predicted class was
    # assigned, so it is preserved here only for downstream error calculations.
    invalid_detection_mask = classified["source"] == "invalid detection"
    classified.loc[invalid_detection_mask, "Fitness"] = 0.1
    return classified


def mechanism_filename(mechanism):
    if mechanism is None:
        return "all_confusion_matrix.png"
    return f"{mechanism.replace(' ', '_')}_confusion_matrix.png"


def make_confusion_matrix_plot(df, output_dir, mechanism=None, labels=None):
    labels = labels or CLASS_LABELS
    if mechanism is None:
        working_df = df
        title_mechanism = "the"
    else:
        working_df = df[df["source"] == mechanism]
        title_mechanism = mechanism

    if working_df.empty:
        print(f"Skipping {mechanism or 'all'} confusion matrix: no rows.")
        return None

    counts = confusion_matrix(
        working_df["Optimizer Class"],
        working_df["Predicted Optimizer Class"],
        labels=labels,
    )
    observation_count = int(counts.sum())
    normalized = counts / observation_count if observation_count else counts

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    sns.heatmap(normalized, annot=True, fmt="f", ax=ax)
    ax.set_xlabel("Predicted Optimizer Class", fontsize=14, labelpad=20)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Actual Optimizer Class", fontsize=14, labelpad=20)
    ax.set_yticklabels(labels)
    if mechanism is None:
        ax.set_title(
            f"Confusion Matrix for the Optimizer Class Prediction "
            f"({observation_count} observations)",
            fontsize=14,
            pad=20,
        )
    else:
        ax.set_title(
            f"Confusion Matrix for {title_mechanism} Optimizer Class Prediction "
            f"({observation_count} observations)",
            fontsize=14,
            pad=20,
        )
    fig.tight_layout()
    output_path = output_dir / mechanism_filename(mechanism)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return {
        "mechanism": mechanism or "all",
        "observations": observation_count,
        "counts": counts.tolist(),
        "normalized": normalized.tolist(),
        "output": str(output_path),
    }


def write_summary(records, output_dir):
    summary_path = output_dir / "confusion_matrix_summary.json"
    summary_path.write_text(json.dumps(records, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce AutoLR speed-up mechanism source and confusion-matrix plots."
    )
    parser.add_argument(
        "--dump-root",
        type=Path,
        help="Root containing experiment/run/iteration_*.json files for methods.png.",
    )
    parser.add_argument(
        "--full-eval-csv",
        type=Path,
        help="Recovered dataframe3.csv with Fitness, Full Fitness, Full Duration, and source columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reproduced_speedup_mechanism_plots"),
        help="Directory where reproduced plots are written.",
    )
    parser.add_argument(
        "--invalid-threshold",
        type=float,
        default=0.15,
        help="Fitness threshold used by the historical notebook to classify invalid optimizers.",
    )
    parser.add_argument(
        "--mechanisms",
        nargs="*",
        default=DEFAULT_MECHANISMS,
        help="Mechanism-specific confusion matrices to create.",
    )
    parser.add_argument(
        "--skip-global-confusion",
        action="store_true",
        help="Only create mechanism-specific confusion matrices.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dump_root is None and args.full_eval_csv is None:
        raise SystemExit("Provide --dump-root, --full-eval-csv, or both.")

    if args.dump_root is not None:
        methods_df = load_results_read_only(args.dump_root)
        plot_sources_per_generation(methods_df, args.output_dir)
        print(f"Wrote source histogram to {args.output_dir / 'methods.png'}")

    records = []
    if args.full_eval_csv is not None:
        full_eval_df = load_full_evaluation_csv(args.full_eval_csv)
        classified_df = classify_optimizers(full_eval_df, args.invalid_threshold)
        if not args.skip_global_confusion:
            record = make_confusion_matrix_plot(classified_df, args.output_dir)
            if record is not None:
                records.append(record)
        for mechanism in args.mechanisms:
            record = make_confusion_matrix_plot(
                classified_df,
                args.output_dir,
                mechanism=mechanism,
            )
            if record is not None:
                records.append(record)
        write_summary(records, args.output_dir)
        print(f"Wrote confusion matrix summary to {args.output_dir / 'confusion_matrix_summary.json'}")


if __name__ == "__main__":
    main()
