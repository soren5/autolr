"""Compare facilitated-mutation and original AutoLR result dumps.

The script is intentionally read-only with respect to experiment dumps. It
parses iteration JSON files, derives per-run trajectories, writes summary
tables and plots, and produces a short Markdown report that separates observed
signals from confounded comparisons.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/autolr_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/autolr_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.smart_phenotype import smart_phenotype


DEFAULT_FM_ROOT = "/Users/soren/Work/autolr/dumps/facilitated_mutation_base"
DEFAULT_OM_ROOT = "/Users/soren/desktop_back_up/_Organized_Results/Original_AutoLR_experiments/adaptiveTest"
DEFAULT_OUTPUT_DIR = "analysis/fm_vs_om_result_analysis"

ARCHITECTURE_TERMS = [
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
STATE_TERMS = ["alpha", "beta", "sigma"]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FM vs OM/original AutoLR result dumps."
    )
    parser.add_argument(
        "--fm-root",
        action="append",
        default=None,
        help="FM result root. May be supplied multiple times.",
    )
    parser.add_argument(
        "--om-root",
        action="append",
        default=None,
        help="OM/original result root. May be supplied multiple times.",
    )
    parser.add_argument(
        "--fm-with-update-root",
        action="append",
        default=[],
        help="Optional FM-with-update result root. May be supplied multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where analysis artifacts are written.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[0.6, 0.7, 0.8],
        help="Quality thresholds. With the default minimized fitness convention, quality=-fitness.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        type=int,
        default=[25, 50, 100, 200],
        help="Early generations used for early-to-final predictiveness.",
    )
    parser.add_argument(
        "--fitness-direction",
        choices=["minimize", "maximize"],
        default="minimize",
        help="Evolutionary fitness direction. The framework normally minimizes, so quality=-fitness.",
    )
    parser.add_argument(
        "--max-phenotype-examples",
        type=int,
        default=8,
        help="Number of final-best phenotype examples to include in the Markdown report.",
    )
    parser.add_argument(
        "--individual-record-sample",
        type=int,
        default=10000,
        help="Maximum individual rows to write to individual_records_sample.csv.",
    )
    parser.add_argument(
        "--write-full-individual-records",
        action="store_true",
        help="Also write full_individual_records.csv. This can be large.",
    )
    args = parser.parse_args()

    condition_roots = build_condition_roots(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    individuals = load_all_individuals(condition_roots, args.fitness_direction)
    if individuals.empty:
        raise SystemExit("No iteration_*.json individual records were found.")

    generation_summary = summarize_generations(individuals)
    run_summary = summarize_runs(generation_summary, individuals)
    final_summary = summarize_conditions(run_summary, args.thresholds)
    budget_comparison = compare_by_budget(generation_summary, run_summary)
    threshold_hits = compute_threshold_hits(generation_summary, run_summary, args.thresholds)
    phenotype_summary, phenotype_examples = summarize_phenotypes(individuals, run_summary)
    predictiveness = compute_early_late_predictiveness(
        generation_summary, run_summary, args.checkpoints
    )
    f_race_diagnostics = collect_f_race_diagnostics(condition_roots)

    write_tables(
        output_dir,
        individuals,
        generation_summary,
        run_summary,
        final_summary,
        budget_comparison,
        threshold_hits,
        phenotype_summary,
        predictiveness,
        f_race_diagnostics,
        args.individual_record_sample,
        args.write_full_individual_records,
    )
    write_plots(output_dir, generation_summary, run_summary, predictiveness)
    write_report(
        output_dir,
        condition_roots,
        args,
        individuals,
        generation_summary,
        run_summary,
        final_summary,
        budget_comparison,
        threshold_hits,
        phenotype_summary,
        phenotype_examples,
        predictiveness,
        f_race_diagnostics,
    )

    print_console_summary(output_dir, final_summary, threshold_hits)


def build_condition_roots(args):
    condition_roots = {
        "FM": [Path(root) for root in (args.fm_root or [DEFAULT_FM_ROOT])],
        "OM": [Path(root) for root in (args.om_root or [DEFAULT_OM_ROOT])],
    }
    if args.fm_with_update_root:
        condition_roots["FM_WITH_UPDATE"] = [
            Path(root) for root in args.fm_with_update_root
        ]
    return condition_roots


def load_all_individuals(condition_roots, fitness_direction):
    records = []
    for condition, roots in condition_roots.items():
        for root in roots:
            for file_path in find_iteration_files(root):
                generation = extract_generation(file_path)
                experiment_name, run_id = infer_experiment_and_run(file_path)
                try:
                    data = load_iteration_json(file_path)
                except Exception as error:
                    print(f"Skipping {file_path}: {type(error).__name__}: {error}")
                    continue
                if not isinstance(data, list):
                    continue
                for index, individual in enumerate(data):
                    if not isinstance(individual, dict):
                        continue
                    fitness = to_float(individual.get("fitness"))
                    if fitness is None or not math.isfinite(fitness):
                        continue
                    phenotype = string_or_none(individual.get("phenotype"))
                    archive_key = extract_archive_key(individual)
                    quality = quality_from_fitness(fitness, fitness_direction)
                    records.append(
                        {
                            "condition": condition,
                            "root": str(root),
                            "experiment_name": experiment_name,
                            "run_id": run_id,
                            "generation": generation,
                            "individual_index": index,
                            "individual_id": individual.get("id", index),
                            "fitness": fitness,
                            "quality": quality,
                            "archive_key": archive_key or "",
                            "smart_phenotype": string_or_none(
                                individual.get("smart_phenotype")
                            )
                            or archive_key
                            or "",
                            "phenotype": phenotype or "",
                            "tree_depth": to_float(individual.get("tree_depth")),
                            "evaluation_count": extract_evaluation_count(individual),
                            "source_file": str(file_path),
                        }
                    )
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce").astype("Int64")
    return df


def find_iteration_files(root):
    if root.is_file():
        if root.name.startswith("iteration_") and root.suffix == ".json":
            return [root]
        return []
    return sorted(root.rglob("iteration_*.json"))


def extract_generation(file_path):
    match = re.search(r"iteration_(\d+)", file_path.name)
    return int(match.group(1)) if match else None


def infer_experiment_and_run(file_path):
    run_dir = file_path.parent
    experiment_dir = run_dir.parent
    return experiment_dir.name, run_dir.name


def load_iteration_json(file_path):
    content = file_path.read_text()
    return json.loads(content.replace("NaN", "0.0"))


def extract_archive_key(individual):
    for field in ["key", "smart_phenotype"]:
        value = string_or_none(individual.get(field))
        if value:
            return value
    phenotype = string_or_none(individual.get("phenotype"))
    if not phenotype:
        return None
    try:
        return smart_phenotype(phenotype)
    except Exception:
        return phenotype


def extract_evaluation_count(individual):
    trials = individual.get("trials")
    if isinstance(trials, list):
        return len(trials)
    evaluations = individual.get("evaluations")
    if isinstance(evaluations, list):
        return len(evaluations)
    return None


def quality_from_fitness(fitness, fitness_direction):
    return -fitness if fitness_direction == "minimize" else fitness


def to_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def string_or_none(value):
    if value is None:
        return None
    return str(value)


def summarize_generations(individuals):
    best_indexes = individuals.groupby(
        ["condition", "run_id", "generation"], observed=True
    )["quality"].idxmax()
    best = individuals.loc[best_indexes].copy()
    best = best.rename(
        columns={
            "quality": "best_quality",
            "fitness": "best_raw_fitness",
            "individual_id": "best_individual_id",
            "archive_key": "best_archive_key",
            "source_file": "best_source_file",
        }
    )
    best = best[
        [
            "condition",
            "run_id",
            "generation",
            "experiment_name",
            "root",
            "best_quality",
            "best_raw_fitness",
            "best_individual_id",
            "best_archive_key",
            "best_source_file",
        ]
    ]

    aggregates = (
        individuals.groupby(["condition", "run_id", "generation"], observed=True)
        .agg(
            population_mean_quality=("quality", "mean"),
            population_median_quality=("quality", "median"),
            population_std_quality=("quality", "std"),
            population_size=("quality", "count"),
            mean_evaluation_count=("evaluation_count", "mean"),
        )
        .reset_index()
    )
    summary = best.merge(
        aggregates, on=["condition", "run_id", "generation"], how="left"
    ).sort_values(["condition", "run_id", "generation"])
    summary["generation_budget_step"] = summary["population_size"]
    cumulative_created = summary.groupby(["condition", "run_id"], observed=True)[
        "generation_budget_step"
    ].cumsum()
    summary["budget_created_before"] = cumulative_created - summary["generation_budget_step"]
    return summary


def summarize_runs(generation_summary, individuals):
    final_rows = generation_summary.loc[
        generation_summary.groupby(["condition", "run_id"], observed=True)[
            "generation"
        ].idxmax()
    ].copy()
    max_generation_by_condition = generation_summary.groupby("condition", observed=True)[
        "generation"
    ].max()
    final_rows["condition_max_generation_seen"] = final_rows["condition"].map(
        max_generation_by_condition
    )
    final_rows["completed_by_observed_condition_max"] = (
        final_rows["generation"] >= final_rows["condition_max_generation_seen"]
    )
    final_rows = final_rows.rename(
        columns={
            "generation": "final_generation",
            "best_quality": "final_best_quality",
            "best_raw_fitness": "final_best_raw_fitness",
            "best_archive_key": "final_best_archive_key",
            "best_individual_id": "final_best_individual_id",
            "population_mean_quality": "final_population_mean_quality",
            "population_median_quality": "final_population_median_quality",
            "population_size": "final_population_size",
            "budget_created_before": "final_budget_created_before",
        }
    )

    observed = (
        individuals.groupby(["condition", "run_id"], observed=True)
        .agg(
            generations_seen=("generation", "nunique"),
            individuals_seen=("quality", "count"),
            max_observed_quality=("quality", "max"),
            median_population_size=("generation", lambda values: len(values)),
        )
        .reset_index()
    )
    return final_rows.merge(
        observed, on=["condition", "run_id"], how="left", suffixes=("", "_all")
    )


def summarize_conditions(run_summary, thresholds):
    rows = []
    for condition, group in run_summary.groupby("condition", observed=True):
        row = {
            "condition": condition,
            "runs": len(group),
            "completed_runs_by_observed_condition_max": int(
                group["completed_by_observed_condition_max"].sum()
            ),
            "final_generation_min": int(group["final_generation"].min()),
            "final_generation_median": float(group["final_generation"].median()),
            "final_generation_max": int(group["final_generation"].max()),
            "final_population_size_median": float(
                group["final_population_size"].median()
            ),
            "final_budget_created_before_median": float(
                group["final_budget_created_before"].median()
            ),
            "final_budget_created_before_max": float(
                group["final_budget_created_before"].max()
            ),
            "final_best_quality_mean": float(group["final_best_quality"].mean()),
            "final_best_quality_median": float(group["final_best_quality"].median()),
            "final_best_quality_max": float(group["final_best_quality"].max()),
            "final_best_quality_std": float(group["final_best_quality"].std(ddof=1))
            if len(group) > 1
            else 0.0,
            "final_population_mean_quality_mean": float(
                group["final_population_mean_quality"].mean()
            ),
            "final_population_median_quality_mean": float(
                group["final_population_median_quality"].mean()
            ),
        }
        for threshold in thresholds:
            row[f"runs_reaching_quality_{threshold:g}"] = int(
                (group["final_best_quality"] >= threshold).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("condition")


def compare_by_budget(generation_summary, run_summary):
    """Compare conditions on the number of prior solutions created.

    Budget is the number of solutions created before the optimizer's generation.
    For constant population size this is generation_index * population_size. The
    implementation uses cumulative logged population sizes from earlier
    generations so it remains correct if a run has a varying population size.
    """

    if generation_summary.empty or run_summary.empty:
        return pd.DataFrame()

    condition_max_budget = run_summary.groupby("condition", observed=True)[
        "final_budget_created_before"
    ].max()
    shared_budget = int(condition_max_budget.min())
    comparable_rows = []
    for (condition, run_id), group in generation_summary.groupby(
        ["condition", "run_id"], observed=True
    ):
        within_budget = group[
            group["budget_created_before"] <= shared_budget
        ]
        if within_budget.empty:
            selected = group.iloc[0]
        else:
            selected = within_budget.iloc[-1]
        final_row = run_summary[
            (run_summary["condition"] == condition) & (run_summary["run_id"] == run_id)
        ].iloc[0]
        reached_shared_budget = (
            final_row["final_budget_created_before"] >= shared_budget
        )
        comparable_rows.append(
            {
                "scope": "run_at_shared_budget",
                "condition": condition,
                "run_id": run_id,
                "shared_budget_created_before": shared_budget,
                "condition_max_budget_created_before": float(condition_max_budget[condition]),
                "selected_generation": int(selected["generation"]),
                "selected_budget_created_before": int(selected["budget_created_before"]),
                "selected_best_quality": float(selected["best_quality"]),
                "selected_population_median_quality": float(
                    selected["population_median_quality"]
                ),
                "final_generation": int(final_row["final_generation"]),
                "final_budget_created_before": int(
                    final_row["final_budget_created_before"]
                ),
                "final_best_quality": float(final_row["final_best_quality"]),
                "post_shared_budget_quality_gain": float(
                    final_row["final_best_quality"] - selected["best_quality"]
                ),
                "reached_shared_budget": bool(reached_shared_budget),
            }
        )

    run_level = pd.DataFrame(comparable_rows)
    condition_rows = []
    for condition, group in run_level.groupby("condition", observed=True):
        comparable_group = group[group["reached_shared_budget"]]
        if comparable_group.empty:
            comparable_group = group
        condition_rows.append(
            {
                "scope": "condition_summary_at_shared_budget",
                "condition": condition,
                "run_id": "",
                "shared_budget_created_before": shared_budget,
                "condition_max_budget_created_before": float(condition_max_budget[condition]),
                "runs_total": int(len(group)),
                "runs_reaching_shared_budget": int(group["reached_shared_budget"].sum()),
                "selected_generation": "",
                "selected_budget_created_before": float(
                    comparable_group["selected_budget_created_before"].median()
                ),
                "selected_best_quality": float(
                    comparable_group["selected_best_quality"].mean()
                ),
                "selected_best_quality_all_runs": float(group["selected_best_quality"].mean()),
                "selected_population_median_quality": float(
                    comparable_group["selected_population_median_quality"].mean()
                ),
                "final_generation": "",
                "final_budget_created_before": float(
                    comparable_group["final_budget_created_before"].median()
                ),
                "final_best_quality": float(comparable_group["final_best_quality"].mean()),
                "post_shared_budget_quality_gain": float(
                    comparable_group["post_shared_budget_quality_gain"].mean()
                ),
                "reached_shared_budget": bool(group["reached_shared_budget"].all()),
            }
        )
    return pd.concat([pd.DataFrame(condition_rows), run_level], ignore_index=True)


def compute_threshold_hits(generation_summary, run_summary, thresholds):
    rows = []
    run_counts = run_summary.groupby("condition", observed=True)["run_id"].nunique()
    for threshold in thresholds:
        hits = generation_summary[generation_summary["best_quality"] >= threshold]
        first_hits = (
            hits.groupby(["condition", "run_id"], observed=True)["generation"]
            .min()
            .reset_index(name="first_hit_generation")
        )
        for condition, run_count in run_counts.items():
            condition_hits = first_hits[first_hits["condition"] == condition]
            hit_count = len(condition_hits)
            rows.append(
                {
                    "condition": condition,
                    "threshold_quality": threshold,
                    "run_count": int(run_count),
                    "hit_count": int(hit_count),
                    "hit_rate": float(hit_count / run_count) if run_count else 0.0,
                    "median_generation_to_threshold": nullable_float(
                        condition_hits["first_hit_generation"].median()
                    ),
                    "earliest_generation_to_threshold": nullable_int(
                        condition_hits["first_hit_generation"].min()
                    ),
                    "latest_generation_to_threshold": nullable_int(
                        condition_hits["first_hit_generation"].max()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["condition", "threshold_quality"])


def nullable_float(value):
    if pd.isna(value):
        return ""
    return float(value)


def nullable_int(value):
    if pd.isna(value):
        return ""
    return int(value)


def summarize_phenotypes(individuals, run_summary):
    final_keys = run_summary[["condition", "run_id", "final_generation"]].rename(
        columns={"final_generation": "generation"}
    )
    candidates = individuals.merge(
        final_keys, on=["condition", "run_id", "generation"], how="inner"
    )
    if candidates.empty:
        return pd.DataFrame(), []
    best_indexes = candidates.groupby(["condition", "run_id"], observed=True)[
        "quality"
    ].idxmax()
    best = candidates.loc[best_indexes].copy()

    feature_rows = []
    examples = []
    for _, row in best.iterrows():
        expression = row["smart_phenotype"] or row["archive_key"] or row["phenotype"]
        features = phenotype_features(expression, row["phenotype"])
        features.update(
            {
                "condition": row["condition"],
                "run_id": row["run_id"],
                "generation": int(row["generation"]),
                "quality": row["quality"],
                "fitness": row["fitness"],
            }
        )
        feature_rows.append(features)
        examples.append(
            {
                "condition": row["condition"],
                "run_id": row["run_id"],
                "generation": int(row["generation"]),
                "quality": row["quality"],
                "archive_key": row["archive_key"],
                "phenotype": row["phenotype"],
            }
        )

    feature_df = pd.DataFrame(feature_rows)
    if feature_df.empty:
        return pd.DataFrame(), examples
    summary = (
        feature_df.groupby("condition", observed=True)
        .agg(
            best_individuals=("quality", "count"),
            token_count_mean=("token_count", "mean"),
            token_count_median=("token_count", "median"),
            char_length_mean=("char_length", "mean"),
            max_parenthesis_depth_mean=("max_parenthesis_depth", "mean"),
            grad_reference_mean=("grad_references", "mean"),
            state_reference_mean=("state_references", "mean"),
            constant_reference_mean=("constant_references", "mean"),
            numeric_literal_mean=("numeric_literals", "mean"),
            architecture_reference_mean=("architecture_references", "mean"),
            repeated_subexpression_proxy_mean=(
                "repeated_subexpression_proxy",
                "mean",
            ),
            state_update_self_reference_proxy_rate=(
                "state_update_self_reference_proxy",
                "mean",
            ),
        )
        .reset_index()
    )
    for term in ARCHITECTURE_TERMS:
        summary[f"{term}_presence_rate"] = (
            feature_df.groupby("condition", observed=True)[f"uses_{term}"].mean().values
        )
    return summary, examples


def phenotype_features(expression, full_phenotype):
    expression = expression or ""
    full_phenotype = full_phenotype or expression
    tokens = re.findall(r"[A-Za-z_]\w*|[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|\S", expression)
    feature = {
        "token_count": len(tokens),
        "char_length": len(expression),
        "max_parenthesis_depth": max_parenthesis_depth(expression),
        "grad_references": word_count(expression, "grad"),
        "state_references": sum(word_count(expression, term) for term in STATE_TERMS),
        "constant_references": len(re.findall(r"\b(?:tf\.)?constant\s*\(", expression)),
        "numeric_literals": len(
            re.findall(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", expression)
        ),
        "architecture_references": sum(
            word_count(expression, term) for term in ARCHITECTURE_TERMS
        ),
        "repeated_subexpression_proxy": repeated_subexpression_proxy(expression),
        "state_update_self_reference_proxy": int(
            has_state_update_self_reference_proxy(full_phenotype)
        ),
    }
    for term in ARCHITECTURE_TERMS:
        feature[f"uses_{term}"] = int(word_count(expression, term) > 0)
    return feature


def max_parenthesis_depth(text):
    depth = 0
    max_depth = 0
    for character in text:
        if character == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif character == ")":
            depth = max(0, depth - 1)
    return max_depth


def word_count(text, word):
    return len(re.findall(rf"\b{re.escape(word)}\b", text))


def repeated_subexpression_proxy(expression):
    calls = re.findall(r"\b[A-Za-z_]\w*\([^()]{3,80}\)", expression)
    counts = Counter(calls)
    return sum(count - 1 for count in counts.values() if count > 1)


def has_state_update_self_reference_proxy(phenotype):
    for state in STATE_TERMS:
        pattern = rf"{state}_func.*?lambda.*?:.*?\b{state}\b"
        if re.search(pattern, phenotype, flags=re.DOTALL):
            return True
    return False


def compute_early_late_predictiveness(generation_summary, run_summary, checkpoints):
    rows = []
    final = run_summary[["condition", "run_id", "final_best_quality"]]
    for checkpoint in checkpoints:
        checkpoint_rows = generation_summary[
            generation_summary["generation"] == checkpoint
        ][["condition", "run_id", "best_quality"]].rename(
            columns={"best_quality": "checkpoint_best_quality"}
        )
        merged = checkpoint_rows.merge(final, on=["condition", "run_id"], how="inner")
        for condition, group in merged.groupby("condition", observed=True):
            rows.append(
                {
                    "condition": condition,
                    "checkpoint_generation": checkpoint,
                    "paired_runs": len(group),
                    "pearson_correlation": correlation(
                        group["checkpoint_best_quality"], group["final_best_quality"]
                    ),
                    "spearman_correlation": correlation(
                        group["checkpoint_best_quality"].rank(method="average"),
                        group["final_best_quality"].rank(method="average"),
                    ),
                    "checkpoint_quality_mean": float(
                        group["checkpoint_best_quality"].mean()
                    )
                    if len(group)
                    else "",
                    "final_quality_mean": float(group["final_best_quality"].mean())
                    if len(group)
                    else "",
                }
            )
    return pd.DataFrame(rows)


def correlation(left, right):
    if len(left) < 2 or left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return ""
    value = left.corr(right)
    return "" if pd.isna(value) else float(value)


def collect_f_race_diagnostics(condition_roots):
    rows = []
    for condition, roots in condition_roots.items():
        for root in roots:
            summary_files = sorted(root.rglob("_race_f_race_summary.csv"))
            selection_files = sorted(root.rglob("_race_selection_audit_summary.csv"))
            if not summary_files and not selection_files:
                rows.append(
                    {
                        "condition": condition,
                        "root": str(root),
                        "run_id": "",
                        "f_race_summary_files": 0,
                        "selection_audit_summary_files": 0,
                        "generations_logged": 0,
                        "extra_evaluations": 0,
                        "eliminated_count": 0,
                        "tournament_events": 0,
                        "tournament_changed": 0,
                        "audit_unavailable_count": 0,
                    }
                )
                continue
            by_run = {}
            for file_path in summary_files:
                by_run.setdefault(file_path.parent.name, {"race": [], "audit": []})[
                    "race"
                ].append(file_path)
            for file_path in selection_files:
                by_run.setdefault(file_path.parent.name, {"race": [], "audit": []})[
                    "audit"
                ].append(file_path)
            for run_id, files in sorted(by_run.items()):
                race_frames = safe_read_csvs(files["race"])
                audit_frames = safe_read_csvs(files["audit"])
                race = pd.concat(race_frames, ignore_index=True) if race_frames else pd.DataFrame()
                audit = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()
                rows.append(
                    {
                        "condition": condition,
                        "root": str(root),
                        "run_id": run_id,
                        "f_race_summary_files": len(files["race"]),
                        "selection_audit_summary_files": len(files["audit"]),
                        "generations_logged": int(
                            race["generation"].nunique()
                            if "generation" in race.columns
                            else 0
                        ),
                        "extra_evaluations": numeric_column_sum(
                            race, "extra_evaluations"
                        ),
                        "eliminated_count": numeric_column_sum(
                            race, "eliminated_count"
                        ),
                        "tournament_events": numeric_column_sum(
                            audit, "tournament_events"
                        ),
                        "tournament_changed": numeric_column_sum(
                            audit, "tournament_changed"
                        ),
                        "audit_unavailable_count": numeric_column_sum(
                            audit, "audit_unavailable_count"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def safe_read_csvs(paths):
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path))
        except Exception as error:
            print(f"Skipping F-race CSV {path}: {type(error).__name__}: {error}")
    return frames


def numeric_column_sum(frame, column):
    if frame.empty or column not in frame.columns:
        return 0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def write_tables(
    output_dir,
    individuals,
    generation_summary,
    run_summary,
    final_summary,
    budget_comparison,
    threshold_hits,
    phenotype_summary,
    predictiveness,
    f_race_diagnostics,
    individual_record_sample,
    write_full_individual_records,
):
    generation_summary.to_csv(output_dir / "generation_summary.csv", index=False)
    run_summary.to_csv(output_dir / "run_summary.csv", index=False)
    final_summary.to_csv(output_dir / "final_fitness_summary.csv", index=False)
    budget_comparison.to_csv(output_dir / "budget_comparison.csv", index=False)
    threshold_hits.to_csv(output_dir / "threshold_hits.csv", index=False)
    phenotype_summary.to_csv(output_dir / "phenotype_structure_summary.csv", index=False)
    predictiveness.to_csv(output_dir / "early_late_predictiveness.csv", index=False)
    f_race_diagnostics.to_csv(output_dir / "f_race_diagnostics.csv", index=False)

    sample_columns = [
        "condition",
        "run_id",
        "generation",
        "individual_id",
        "fitness",
        "quality",
        "archive_key",
        "source_file",
    ]
    individuals[sample_columns].head(individual_record_sample).to_csv(
        output_dir / "individual_records_sample.csv", index=False
    )
    if write_full_individual_records:
        individuals[sample_columns].to_csv(
            output_dir / "full_individual_records.csv", index=False
        )


def write_plots(output_dir, generation_summary, run_summary, predictiveness):
    plot_trajectory(
        generation_summary,
        output_dir / "best_fitness_trajectory.png",
        x_column="generation",
        x_label="Generation",
        value_column="best_quality",
        ylabel="Best quality (-fitness when minimizing)",
        title="Best Fitness Trajectory",
        include_runs=True,
    )
    plot_trajectory(
        generation_summary,
        output_dir / "best_fitness_by_budget.png",
        x_column="budget_created_before",
        x_label="Budget: solutions created before current generation",
        value_column="best_quality",
        ylabel="Best quality (-fitness when minimizing)",
        title="Best Fitness By Budget",
        include_runs=True,
    )
    plot_trajectory(
        generation_summary,
        output_dir / "population_fitness_trajectory.png",
        x_column="generation",
        x_label="Generation",
        value_column="population_median_quality",
        ylabel="Median population quality (-fitness when minimizing)",
        title="Population Fitness Trajectory",
        include_runs=False,
    )
    plot_trajectory(
        generation_summary,
        output_dir / "population_fitness_by_budget.png",
        x_column="budget_created_before",
        x_label="Budget: solutions created before current generation",
        value_column="population_median_quality",
        ylabel="Median population quality (-fitness when minimizing)",
        title="Population Fitness By Budget",
        include_runs=False,
    )
    plot_checkpoint_scatter(output_dir, generation_summary, run_summary, predictiveness)


def plot_trajectory(
    generation_summary,
    path,
    x_column,
    x_label,
    value_column,
    ylabel,
    title,
    include_runs=False,
):
    fig, axis = plt.subplots(figsize=(11, 6))
    for condition, condition_frame in generation_summary.groupby("condition", observed=True):
        if include_runs:
            for _, run_frame in condition_frame.groupby("run_id", observed=True):
                axis.plot(
                    run_frame[x_column],
                    run_frame[value_column],
                    color=condition_color(condition),
                    alpha=0.12,
                    linewidth=0.8,
                )

        grouped_values = (
            condition_frame.groupby("generation", observed=True)[value_column]
            if x_column == "generation"
            else condition_frame.groupby(x_column, observed=True)[value_column]
        )
        aggregate = (
            grouped_values
            .agg(["median", percentile_25, percentile_75])
            .reset_index()
        )
        axis.plot(
            aggregate[x_column],
            aggregate["median"],
            label=f"{condition} median",
            color=condition_color(condition),
            linewidth=2.2,
        )
        axis.fill_between(
            aggregate[x_column].astype(float).to_numpy(),
            aggregate["percentile_25"].astype(float).to_numpy(),
            aggregate["percentile_75"].astype(float).to_numpy(),
            color=condition_color(condition),
            alpha=0.16,
            linewidth=0,
        )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def percentile_25(values):
    return values.quantile(0.25)


def percentile_75(values):
    return values.quantile(0.75)


def condition_color(condition):
    return {
        "FM": "#2F80ED",
        "OM": "#D35400",
        "FM_WITH_UPDATE": "#219653",
    }.get(condition, "#4F4F4F")


def plot_checkpoint_scatter(output_dir, generation_summary, run_summary, predictiveness):
    if predictiveness.empty:
        return
    final = run_summary[["condition", "run_id", "final_best_quality"]]
    for checkpoint in sorted(predictiveness["checkpoint_generation"].unique()):
        checkpoint_rows = generation_summary[
            generation_summary["generation"] == checkpoint
        ][["condition", "run_id", "best_quality"]].rename(
            columns={"best_quality": "checkpoint_best_quality"}
        )
        merged = checkpoint_rows.merge(final, on=["condition", "run_id"], how="inner")
        if merged.empty:
            continue
        fig, axis = plt.subplots(figsize=(7, 6))
        for condition, group in merged.groupby("condition", observed=True):
            axis.scatter(
                group["checkpoint_best_quality"],
                group["final_best_quality"],
                label=condition,
                color=condition_color(condition),
                alpha=0.75,
            )
        axis.set_title(f"Generation {checkpoint} vs Final Quality")
        axis.set_xlabel(f"Best quality at generation {checkpoint}")
        axis.set_ylabel("Final best quality")
        axis.grid(True, alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"checkpoint_vs_final_gen_{checkpoint}.png", dpi=180)
        plt.close(fig)


def write_report(
    output_dir,
    condition_roots,
    args,
    individuals,
    generation_summary,
    run_summary,
    final_summary,
    budget_comparison,
    threshold_hits,
    phenotype_summary,
    phenotype_examples,
    predictiveness,
    f_race_diagnostics,
):
    comparability_notes = build_comparability_notes(run_summary, f_race_diagnostics)
    top_summary = dataframe_to_markdown(final_summary)
    threshold_summary = dataframe_to_markdown(threshold_hits)
    budget_comparison_table = (
        dataframe_to_markdown(
            budget_comparison[
                budget_comparison["scope"] == "condition_summary_at_shared_budget"
            ]
        )
        if not budget_comparison.empty
        else "No budget comparison could be computed."
    )
    phenotype_table = (
        dataframe_to_markdown(phenotype_summary)
        if not phenotype_summary.empty
        else "No phenotype features could be summarized."
    )
    predictiveness_table = (
        dataframe_to_markdown(predictiveness)
        if not predictiveness.empty
        else "No checkpoint rows were available for the requested checkpoints."
    )
    f_race_table = (
        dataframe_to_markdown(f_race_diagnostics)
        if not f_race_diagnostics.empty
        else "No F-race diagnostic files were found."
    )
    examples = format_phenotype_examples(phenotype_examples, args.max_phenotype_examples)
    automated_recommendation = build_automated_recommendation(
        final_summary, budget_comparison, comparability_notes
    )

    lines = [
        "# FM vs OM Result Analysis",
        "",
        "## Inputs",
    ]
    for condition, roots in condition_roots.items():
        for root in roots:
            lines.append(f"- {condition}: `{root}`")
    lines.extend(
        [
            "",
            "## Fitness Convention",
            "",
            f"- Raw evolutionary fitness was interpreted as `{args.fitness_direction}`.",
            "- Reported `quality` is higher-is-better. With the default framework convention, `quality = -fitness`.",
            "- `budget_created_before` is the number of solutions logged in earlier generations of the same run. With constant population size, this is `generation_index * population_size`.",
            "- This is a creation-budget axis, not an F-race evaluation-call axis.",
            "",
            "## Comparability Notes",
            "",
        ]
    )
    if comparability_notes:
        lines.extend(f"- {note}" for note in comparability_notes)
    else:
        lines.append("- No obvious run-count, generation-count, population-size, or F-race-log mismatch was detected from the parsed files.")

    lines.extend(
        [
            "",
            "## Directly Supported Findings",
            "",
            "- The tables and plots below are directly computed from parsed `iteration_*.json` files without modifying the dumps.",
            "- Threshold hits use final/run trajectory quality and the thresholds supplied to the script.",
            "",
            "### Final Fitness Summary",
            "",
            top_summary,
            "",
            "### Budget Comparison",
            "",
            budget_comparison_table,
            "",
            "### Threshold Hits",
            "",
            threshold_summary,
            "",
            "### Phenotype Structure Summary",
            "",
            phenotype_table,
            "",
            "### Early-To-Late Predictiveness",
            "",
            predictiveness_table,
            "",
            "### F-Race Diagnostics",
            "",
            f_race_table,
            "",
            "## Suggestive But Confounded Findings",
            "",
            "- Differences between FM and OM should be treated as suggestive unless dataset/task sequence, grammar, and racing configuration are confirmed comparable.",
            "- If one condition has a higher final budget, compare both the shared-budget table and the by-budget trajectory plots before interpreting final summaries.",
            "- The phenotype structure features are proxies. They are useful for finding grammar/search symptoms such as bloat or architecture-variable usage, but they are not a semantic equivalence proof.",
            "",
            "## Recommendation",
            "",
            automated_recommendation,
            "",
            "- Prefer current FM if its median trajectory and threshold hit rate are close to OM under comparable observed budgets.",
            "- Prefer longer FM runs if the FM median best-quality trajectory is still climbing at the cutoff.",
            "- Prefer an OM-like update-rule experiment if OM substantially outperforms FM and OM final phenotypes show more compact stateful structure.",
            "- Prefer reducing architecture/noise first if architecture-variable usage is common in weak final best phenotypes.",
            "- Prefer softer racing/thresholding if early-to-final correlations are weak and F-race logs show substantial early elimination.",
            "",
            "## Final-Best Phenotype Examples",
            "",
            examples,
            "",
            "## Artifacts",
            "",
            "- `final_fitness_summary.csv`",
            "- `budget_comparison.csv`",
            "- `generation_summary.csv`",
            "- `run_summary.csv`",
            "- `threshold_hits.csv`",
            "- `phenotype_structure_summary.csv`",
            "- `early_late_predictiveness.csv`",
            "- `f_race_diagnostics.csv`",
            "- `individual_records_sample.csv` bounded by `--individual-record-sample`",
            "- `full_individual_records.csv` only when `--write-full-individual-records` is used",
            "- `best_fitness_trajectory.png`",
            "- `best_fitness_by_budget.png`",
            "- `population_fitness_trajectory.png`",
            "- `population_fitness_by_budget.png`",
            "- `checkpoint_vs_final_gen_<N>.png` when checkpoint data exists",
        ]
    )
    (output_dir / "fm_vs_om_report.md").write_text("\n".join(lines))


def build_automated_recommendation(final_summary, budget_comparison, comparability_notes):
    if {"FM", "OM"} - set(final_summary["condition"]):
        return (
            "Automated recommendation: insufficient FM and OM rows were available "
            "for a direct recommendation."
        )
    fm = final_summary[final_summary["condition"] == "FM"].iloc[0]
    om = final_summary[final_summary["condition"] == "OM"].iloc[0]
    fm_median = fm["final_best_quality_median"]
    om_median = om["final_best_quality_median"]
    fm_max = fm["final_best_quality_max"]
    om_max = om["final_best_quality_max"]
    confounded = bool(comparability_notes)
    budget_summary = (
        budget_comparison[
            budget_comparison["scope"] == "condition_summary_at_shared_budget"
        ]
        if not budget_comparison.empty
        else pd.DataFrame()
    )

    if {"FM", "OM"} <= set(budget_summary.get("condition", [])):
        fm_budget = budget_summary[budget_summary["condition"] == "FM"].iloc[0]
        om_budget = budget_summary[budget_summary["condition"] == "OM"].iloc[0]
        fm_shared = fm_budget["selected_best_quality"]
        om_shared = om_budget["selected_best_quality"]
        om_extra_gain = om_budget["post_shared_budget_quality_gain"]
        shared_budget = int(fm_budget["shared_budget_created_before"])
        if fm_shared >= om_shared * 0.98:
            recommendation = (
                f"At the shared budget of {shared_budget} prior solutions, FM is "
                "not clearly worse than OM on mean selected best quality."
            )
        else:
            recommendation = (
                f"At the shared budget of {shared_budget} prior solutions, OM has "
                "the stronger mean selected best-quality signal."
            )
        recommendation += (
            f" OM's mean gain after the shared budget is {om_extra_gain:.4f}, "
            "which estimates the practical value of its larger search budget."
        )
    elif fm_median >= om_median * 0.98:
        recommendation = (
            "FM is not clearly worse on final median quality in these parsed results."
        )
    elif om_median > fm_median:
        recommendation = (
            "OM has the stronger final median quality signal in these parsed results."
        )
    else:
        recommendation = (
            "The final median quality comparison is inconclusive from these parsed results."
        )

    if om_max > fm_max:
        recommendation += " OM also has the higher observed maximum quality."
    elif fm_max > om_max:
        recommendation += " FM has the higher observed maximum quality."

    if confounded:
        recommendation += (
            " Because the comparison is confounded by observed setup differences, "
            "treat this as a prioritization signal rather than a causal result."
        )
    return f"Automated recommendation: {recommendation}"


def dataframe_to_markdown(frame):
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in frame.iterrows():
        rows.append(
            "| "
            + " | ".join(markdown_cell(row[column]) for column in columns)
            + " |"
        )
    return "\n".join(rows)


def markdown_cell(value):
    if pd.isna(value):
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def build_comparability_notes(run_summary, f_race_diagnostics):
    notes = []
    if run_summary.empty:
        return ["No run summaries were produced."]
    by_condition = run_summary.groupby("condition", observed=True)
    run_counts = by_condition["run_id"].nunique()
    if run_counts.nunique() > 1:
        notes.append(f"Run counts differ by condition: {run_counts.to_dict()}.")
    generation_ranges = by_condition["final_generation"].agg(["min", "max"]).to_dict(
        orient="index"
    )
    if len({(v["min"], v["max"]) for v in generation_ranges.values()}) > 1:
        notes.append(f"Observed final-generation ranges differ: {generation_ranges}.")
    pop_sizes = by_condition["final_population_size"].median().to_dict()
    if len({round(value, 6) for value in pop_sizes.values()}) > 1:
        notes.append(f"Median final population sizes differ: {pop_sizes}.")
    if not f_race_diagnostics.empty:
        f_race_presence = (
            f_race_diagnostics.groupby("condition", observed=True)[
                "f_race_summary_files"
            ]
            .sum()
            .to_dict()
        )
        if len(set(value > 0 for value in f_race_presence.values())) > 1:
            notes.append(f"F-race log availability differs: {f_race_presence}.")
    return notes


def format_phenotype_examples(examples, limit):
    if not examples:
        return "No phenotype examples were available."
    lines = []
    for example in sorted(
        examples, key=lambda item: (item["condition"], item["run_id"])
    )[:limit]:
        key = example["archive_key"].replace("\n", " ")
        if len(key) > 500:
            key = key[:497] + "..."
        lines.extend(
            [
                f"### {example['condition']} {example['run_id']}",
                "",
                f"- generation: {example['generation']}",
                f"- quality: {example['quality']}",
                f"- archive key: `{key}`",
                "",
            ]
        )
    return "\n".join(lines)


def print_console_summary(output_dir, final_summary, threshold_hits):
    print("FM vs OM analysis complete")
    print(f"Output directory: {output_dir}")
    print("\nFinal fitness summary:")
    print(final_summary.to_string(index=False))
    print("\nThreshold hits:")
    print(threshold_hits.to_string(index=False))


if __name__ == "__main__":
    main()
