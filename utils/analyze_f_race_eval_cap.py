#!/usr/bin/env python3
"""Estimate how many F-race evaluations were needed in high-max runs.

Example:
    python utils/analyze_f_race_eval_cap.py --root dumps/my_experiment --output-dir analysis/f_race_caps
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


F_RACE_REPORT = "_race_f_race_report.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Replay observed F-race trial prefixes to estimate useful eval caps.")
    parser.add_argument("--root", default="dumps", help="Dump root, experiment directory, or run directory to scan.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated CSV files.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Mann-Whitney significance threshold.")
    parser.add_argument("--min-evals", type=int, default=2, help="Minimum trials per side before testing separation.")
    parser.add_argument("--max-cap", type=int, default=None, help="Optional highest prefix cap to test.")
    return parser.parse_args()


def find_run_dirs(root):
    root = Path(root)
    if (root / F_RACE_REPORT).is_file():
        return [root]
    return sorted(path.parent for path in root.rglob(F_RACE_REPORT) if path.parent.is_dir())


def run_metadata(run_dir, root):
    try:
        relative = run_dir.relative_to(root)
    except ValueError:
        relative = run_dir
    parts = relative.parts
    experiment_name = parts[-2] if len(parts) >= 2 else run_dir.parent.name
    run_number = parts[-1] if parts else run_dir.name
    return experiment_name, run_number, str(relative)


def read_jsonl(path):
    events = []
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def load_iteration_trials(run_dir):
    trials_by_generation = {}
    for path in sorted(run_dir.glob("iteration_*.json"), key=iteration_number):
        generation = iteration_number(path)
        if generation is None:
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        key_to_trials = {}
        for individual in data:
            if not isinstance(individual, dict):
                continue
            key = individual.get("key") or individual.get("smart_phenotype")
            trials = individual.get("trials")
            if key is None or not isinstance(trials, list):
                continue
            numeric_trials = [float(value) for value in trials if is_number(value)]
            if len(numeric_trials) > len(key_to_trials.get(key, [])):
                key_to_trials[key] = numeric_trials
        if key_to_trials:
            trials_by_generation[generation] = key_to_trials
    return trials_by_generation


def iteration_number(path):
    stem = Path(path).stem
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return None


def is_number(value):
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(float(value))


def group_events_by_generation(events):
    grouped = defaultdict(list)
    for event in events:
        if "generation" in event:
            grouped[int(event["generation"])].append(event)
    return grouped


def reconstruct_generation_trials(events, iteration_trials):
    snapshots = {}
    reevaluations = defaultdict(list)
    race_stop = None

    for event in events:
        event_name = event.get("event")
        if event_name == "candidate_snapshot" and event.get("valid", True):
            key = event.get("key")
            if key is not None:
                snapshots[key] = event
        elif event_name == "reevaluation":
            reevaluations[event.get("key")].append(event)
        elif event_name == "race_stop":
            race_stop = event

    trials = {}
    sources = {}
    missing_pre_race_trials = {}
    for key, snapshot in snapshots.items():
        iteration_key_trials = iteration_trials.get(key)
        if iteration_key_trials:
            trials[key] = iteration_key_trials
            sources[key] = "iteration_trials"
            missing_pre_race_trials[key] = 0
            continue

        reconstructed = []
        first_fitness = snapshot.get("first_fitness")
        if is_number(first_fitness):
            reconstructed.append(float(first_fitness))
        for event in sorted(reevaluations.get(key, []), key=lambda row: row.get("eval_number", 0)):
            new_fitness = event.get("new_fitness")
            if is_number(new_fitness):
                reconstructed.append(float(new_fitness))
        trials[key] = reconstructed
        sources[key] = "race_report_partial"
        missing_pre_race_trials[key] = max(0, int(snapshot.get("n_evals_before") or 0) - 1)

    return snapshots, trials, sources, missing_pre_race_trials, race_stop


def prefix(values, cap):
    return values[: min(cap, len(values))]


def mean(values):
    return sum(values) / len(values) if values else math.inf


def clearly_worse(best_trials, candidate_trials, alpha, min_evals):
    if len(best_trials) < min_evals or len(candidate_trials) < min_evals:
        return False, None
    if mean(candidate_trials) <= mean(best_trials):
        return False, None
    try:
        _, p_value = stats.mannwhitneyu(best_trials, candidate_trials)
    except ValueError:
        p_value = 1.0
    return p_value < alpha, p_value


def analyze_generation(trials, race_stop, alpha, min_evals, max_cap):
    if not race_stop or not trials:
        return None, []

    final_best_key = race_stop.get("final_best_key") or race_stop.get("winner_key")
    if final_best_key not in trials:
        return None, []

    observed_max = max((len(values) for values in trials.values()), default=0)
    tested_max = min(observed_max, max_cap) if max_cap else observed_max
    candidate_keys = sorted(trials.keys())
    cap_to_match = None
    cap_to_separate = None

    pairwise = []
    for candidate_key in candidate_keys:
        if candidate_key == final_best_key:
            continue
        separation_cap = None
        separation_p = None
        for cap in range(min_evals, tested_max + 1):
            is_worse, p_value = clearly_worse(
                prefix(trials[final_best_key], cap),
                prefix(trials[candidate_key], cap),
                alpha,
                min_evals,
            )
            if is_worse:
                separation_cap = cap
                separation_p = p_value
                break
        pairwise.append({
            "candidate_key": candidate_key,
            "candidate_observed_evals": len(trials[candidate_key]),
            "winner_observed_evals": len(trials[final_best_key]),
            "first_separation_cap": separation_cap,
            "separation_p_value": separation_p,
            "separated_at_observed_max": separation_cap is not None,
            "candidate_final_mean": mean(trials[candidate_key]),
            "winner_final_mean": mean(trials[final_best_key]),
        })

    for cap in range(1, tested_max + 1):
        prefix_means = {
            key: mean(prefix(values, cap))
            for key, values in trials.items()
            if prefix(values, cap)
        }
        if prefix_means and min(prefix_means, key=prefix_means.get) == final_best_key:
            cap_to_match = cap
            break

    for cap in range(min_evals, tested_max + 1):
        prefix_means = {
            key: mean(prefix(values, cap))
            for key, values in trials.items()
            if prefix(values, cap)
        }
        if not prefix_means or min(prefix_means, key=prefix_means.get) != final_best_key:
            continue
        all_losers_separated = True
        for candidate_key in candidate_keys:
            if candidate_key == final_best_key:
                continue
            is_worse, _ = clearly_worse(
                prefix(trials[final_best_key], cap),
                prefix(trials[candidate_key], cap),
                alpha,
                min_evals,
            )
            if not is_worse:
                all_losers_separated = False
                break
        if all_losers_separated:
            cap_to_separate = cap
            break

    generation_row = {
        "observed_final_best_key": final_best_key,
        "candidate_count": len(candidate_keys),
        "max_observed_evals": observed_max,
        "tested_max_cap": tested_max,
        "cap_to_match_final_best": cap_to_match,
        "cap_to_separate_final_best": cap_to_separate,
        "final_best_separated_at_observed_max": cap_to_separate is not None,
        "pairwise_comparisons": len(pairwise),
        "pairwise_separations_found": sum(1 for row in pairwise if row["separated_at_observed_max"]),
    }
    return generation_row, pairwise


def analyze_run(run_dir, root, alpha, min_evals, max_cap):
    experiment_name, run_number, scope = run_metadata(run_dir, root)
    events_by_generation = group_events_by_generation(read_jsonl(run_dir / F_RACE_REPORT))
    iteration_trials_by_generation = load_iteration_trials(run_dir)
    generation_rows = []
    pairwise_rows = []

    for generation, events in sorted(events_by_generation.items()):
        snapshots, trials, sources, missing, race_stop = reconstruct_generation_trials(
            events,
            iteration_trials_by_generation.get(generation, {}),
        )
        generation_result, pairwise = analyze_generation(trials, race_stop, alpha, min_evals, max_cap)
        if generation_result is None:
            continue

        shared = {
            "scope": scope,
            "experiment_name": experiment_name,
            "run_number": run_number,
            "generation": generation,
        }
        source_counts = pd.Series(list(sources.values())).value_counts().to_dict() if sources else {}
        generation_result.update(shared)
        generation_result.update({
            "snapshot_candidate_count": len(snapshots),
            "candidates_with_trials": sum(1 for values in trials.values() if values),
            "missing_pre_race_trial_count": sum(missing.values()),
            "iteration_trial_candidates": source_counts.get("iteration_trials", 0),
            "partial_report_candidates": source_counts.get("race_report_partial", 0),
        })
        generation_rows.append(generation_result)

        for row in pairwise:
            row.update(shared)
            row["winner_key"] = generation_result["observed_final_best_key"]
            pairwise_rows.append(row)

    return generation_rows, pairwise_rows


def summarize_runs(generation_df, pairwise_df):
    if generation_df.empty:
        return pd.DataFrame()
    rows = []
    for (experiment_name, run_number), group in generation_df.groupby(["experiment_name", "run_number"]):
        pair_group = pairwise_df[
            (pairwise_df["experiment_name"] == experiment_name)
            & (pairwise_df["run_number"] == run_number)
        ]
        separation_caps = pair_group["first_separation_cap"].dropna() if not pair_group.empty else pd.Series(dtype=float)
        generation_caps = group["cap_to_separate_final_best"].dropna()
        rows.append({
            "experiment_name": experiment_name,
            "run_number": run_number,
            "generations_analyzed": len(group),
            "generations_with_full_final_separation": int(group["final_best_separated_at_observed_max"].sum()),
            "generation_full_separation_rate": float(group["final_best_separated_at_observed_max"].mean()),
            "median_cap_to_separate_generation": percentile_or_none(generation_caps, 50),
            "p90_cap_to_separate_generation": percentile_or_none(generation_caps, 90),
            "pairwise_comparisons": len(pair_group),
            "pairwise_separations_found": int(pair_group["separated_at_observed_max"].sum()) if not pair_group.empty else 0,
            "pairwise_separation_rate": float(pair_group["separated_at_observed_max"].mean()) if not pair_group.empty else 0,
            "median_pairwise_separation_cap": percentile_or_none(separation_caps, 50),
            "p90_pairwise_separation_cap": percentile_or_none(separation_caps, 90),
            "median_cap_to_match_final_best": percentile_or_none(group["cap_to_match_final_best"].dropna(), 50),
            "missing_pre_race_trial_count": int(group["missing_pre_race_trial_count"].sum()),
        })
    return pd.DataFrame(rows).sort_values(["experiment_name", "run_number"])


def percentile_or_none(values, percentile):
    if len(values) == 0:
        return None
    return float(np.percentile(values, percentile))


def write_outputs(generation_rows, pairwise_rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_df = pd.DataFrame(generation_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)
    summary_df = summarize_runs(generation_df, pairwise_df)

    generation_df.to_csv(output_dir / "_race_eval_cap_generation_analysis.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pairwise_df.to_csv(output_dir / "_race_eval_cap_pairwise_analysis.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    summary_df.to_csv(output_dir / "_race_eval_cap_run_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    return summary_df


def print_summary(summary_df, output_dir):
    if summary_df.empty:
        print("No analyzable F-race generations found.")
        return
    print(f"Wrote F-race eval-cap analysis to: {output_dir}")
    columns = [
        "experiment_name",
        "run_number",
        "generations_analyzed",
        "generation_full_separation_rate",
        "median_cap_to_separate_generation",
        "p90_cap_to_separate_generation",
        "pairwise_separation_rate",
        "median_pairwise_separation_cap",
        "missing_pre_race_trial_count",
    ]
    print(summary_df[columns].to_string(index=False))


def main():
    args = parse_args()
    root = Path(args.root)
    run_dirs = find_run_dirs(root)
    if not run_dirs:
        print(f"No {F_RACE_REPORT} files found under {root}")
        return 1

    generation_rows = []
    pairwise_rows = []
    for run_dir in run_dirs:
        run_generation_rows, run_pairwise_rows = analyze_run(
            run_dir,
            root,
            alpha=args.alpha,
            min_evals=args.min_evals,
            max_cap=args.max_cap,
        )
        generation_rows.extend(run_generation_rows)
        pairwise_rows.extend(run_pairwise_rows)

    output_dir = Path(args.output_dir) if args.output_dir else root
    summary_df = write_outputs(generation_rows, pairwise_rows, output_dir)
    print_summary(summary_df, output_dir)
    return 0 if generation_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
