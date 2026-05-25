import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.race_behavior_key import extract_numeric_constants, parse_expression, race_behavior_key, refined_race_behavior_key
from utils.smart_phenotype import smart_phenotype


SUMMARY_FIELDS = [
    "scope",
    "method",
    "experiment_name",
    "run_number",
    "generations_seen",
    "individuals_seen",
    "archive_keys",
    "race_behavior_keys",
    "collapsed_behavior_groups",
    "collapsed_archive_keys",
    "collapsed_individual_occurrences",
]


def main():
    parser = argparse.ArgumentParser(description="Analyze conservative race behavior key grouping in existing results.")
    parser.add_argument("--root", default="dumps", help="Dumps root, experiment/run directory, or iteration JSON file.")
    parser.add_argument("--output-dir", default=None, help="Directory for analysis artifacts. Defaults to the analyzed root.")
    parser.add_argument("--examples", type=int, default=10, help="Maximum example individuals stored per collapsing group.")
    parser.add_argument(
        "--method",
        default="safe",
        choices=["safe", "constants", "fingerprint", "fingerprint_refined"],
        help="Race behavior key method. 'constants' simplifies identities; fingerprint methods hash probe outputs.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir is not None else default_output_dir(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_iteration_records(root, method=args.method)
    summary_rows, groups = analyze_records(records, examples=args.examples, method=args.method)
    write_summary_csv(output_dir / "_race_behavior_key_summary.csv", summary_rows)
    write_groups_json(output_dir / "_race_behavior_key_groups.json", summary_rows[0], groups)
    print_console_summary(summary_rows[0], groups, args.examples)


def default_output_dir(root):
    if root.is_file():
        return root.parent
    return root


def read_iteration_records(root, method="safe"):
    records = []
    for file_path in find_iteration_files(root):
        experiment_name, run_number = infer_experiment_and_run(file_path)
        generation = extract_iteration_number(str(file_path))
        try:
            data = load_json_without_mutating(file_path)
        except Exception as error:
            print(f"Skipping {file_path}: {type(error).__name__}: {error}")
            continue
        if not isinstance(data, list):
            continue
        for indiv in data:
            if not isinstance(indiv, dict):
                continue
            archive_key = extract_archive_key(indiv)
            if archive_key is None:
                continue
            records.append({
                "experiment_name": experiment_name,
                "run_number": run_number,
                "generation": generation,
                "genetic_id": indiv.get("id"),
                "fitness": indiv.get("fitness"),
                "archive_key": archive_key,
                "race_behavior_key": initial_race_behavior_key(archive_key, method),
                "source_file": str(file_path),
            })
    return records


def find_iteration_files(root):
    if root.is_file():
        if root.name.startswith("iteration_") and root.suffix == ".json":
            return [root]
        return []
    return sorted(root.rglob("iteration_*.json"))


def infer_experiment_and_run(file_path):
    run_dir = file_path.parent
    experiment_dir = run_dir.parent
    run_number = run_dir.name
    experiment_name = experiment_dir.name
    return experiment_name, run_number


def load_json_without_mutating(file_path):
    content = file_path.read_text()
    return json.loads(content.replace("NaN", "0.0"))


def extract_iteration_number(file_path):
    match = re.search(r"iteration_(\d+)", file_path)
    if match:
        return int(match.group(1))
    return None


def extract_archive_key(indiv):
    if indiv.get("key") is not None:
        return indiv["key"]
    if indiv.get("smart_phenotype") is not None:
        return indiv["smart_phenotype"]
    if indiv.get("phenotype") is not None:
        try:
            return smart_phenotype(indiv["phenotype"])
        except Exception:
            return None
    return None


def initial_race_behavior_key(archive_key, method):
    if method == "fingerprint_refined":
        return race_behavior_key(archive_key, method="fingerprint")
    return race_behavior_key(archive_key, method=method)


def analyze_records(records, examples=10, method="safe"):
    if method == "fingerprint_refined":
        records = refine_fingerprint_records(records)
    summary_rows = [summarize_scope("global", None, None, records, method)]
    grouped_by_run = defaultdict(list)
    for record in records:
        grouped_by_run[(record["experiment_name"], record["run_number"])].append(record)
    for (experiment_name, run_number), run_records in sorted(grouped_by_run.items()):
        summary_rows.append(summarize_scope("run", experiment_name, run_number, run_records, method))
    return summary_rows, build_collapsing_groups(records, examples)


def refine_fingerprint_records(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["race_behavior_key"]].append(record)

    refined_records = []
    for coarse_key, group_records in grouped.items():
        if len({record["archive_key"] for record in group_records}) <= 1:
            refined_records.extend(group_records)
            continue
        constants = set()
        for record in group_records:
            constants.update(extract_numeric_constants(parse_expression(record["archive_key"])))
        for record in group_records:
            refined = dict(record)
            refined["race_behavior_key"] = refined_race_behavior_key(record["archive_key"], constants)
            refined_records.append(refined)
    return refined_records


def summarize_scope(scope, experiment_name, run_number, records, method="safe"):
    archive_keys = {record["archive_key"] for record in records}
    race_behavior_keys = {record["race_behavior_key"] for record in records}
    generations = {
        (record["experiment_name"], record["run_number"], record["generation"])
        for record in records
    }
    grouped = group_archive_keys_by_behavior(records)
    collapsed_groups = {
        behavior_key: keys
        for behavior_key, keys in grouped.items()
        if len(keys) > 1
    }
    collapsed_archive_keys = set()
    for keys in collapsed_groups.values():
        collapsed_archive_keys.update(keys)
    collapsed_individual_occurrences = sum(
        1 for record in records
        if record["race_behavior_key"] in collapsed_groups
    )
    return {
        "scope": scope,
        "method": method,
        "experiment_name": experiment_name or "",
        "run_number": run_number or "",
        "generations_seen": len(generations),
        "individuals_seen": len(records),
        "archive_keys": len(archive_keys),
        "race_behavior_keys": len(race_behavior_keys),
        "collapsed_behavior_groups": len(collapsed_groups),
        "collapsed_archive_keys": len(collapsed_archive_keys),
        "collapsed_individual_occurrences": collapsed_individual_occurrences,
    }


def group_archive_keys_by_behavior(records):
    grouped = defaultdict(set)
    for record in records:
        grouped[record["race_behavior_key"]].add(record["archive_key"])
    return grouped


def build_collapsing_groups(records, examples):
    grouped_records = defaultdict(list)
    for record in records:
        grouped_records[record["race_behavior_key"]].append(record)

    groups = []
    for behavior_key, behavior_records in grouped_records.items():
        archive_keys = sorted({record["archive_key"] for record in behavior_records})
        if len(archive_keys) <= 1:
            continue
        groups.append({
            "race_behavior_key": behavior_key,
            "archive_keys": archive_keys,
            "archive_key_count": len(archive_keys),
            "occurrence_count": len(behavior_records),
            "experiments": sorted({record["experiment_name"] for record in behavior_records}),
            "runs": sorted({record["run_number"] for record in behavior_records}),
            "generations": sorted({record["generation"] for record in behavior_records}),
            "examples": [
                {
                    "experiment_name": record["experiment_name"],
                    "run_number": record["run_number"],
                    "generation": record["generation"],
                    "genetic_id": record["genetic_id"],
                    "fitness": record["fitness"],
                    "archive_key": record["archive_key"],
                    "source_file": record["source_file"],
                }
                for record in behavior_records[:examples]
            ],
        })
    groups.sort(key=lambda group: (group["archive_key_count"], group["occurrence_count"]), reverse=True)
    return groups


def write_summary_csv(path, summary_rows):
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)


def write_groups_json(path, global_summary, groups):
    with path.open("w") as output_file:
        json.dump({
            "summary": global_summary,
            "groups": groups,
        }, output_file, indent=2)


def print_console_summary(global_summary, groups, examples):
    print("Race behavior key analysis")
    for field in SUMMARY_FIELDS:
        print(f"{field}: {global_summary[field]}")
    print(f"\nTop {examples} collapsing groups:")
    for group in groups[:examples]:
        print(
            f"- archive_keys={group['archive_key_count']} "
            f"occurrences={group['occurrence_count']} "
            f"race_behavior_key={group['race_behavior_key']}"
        )


if __name__ == "__main__":
    main()
