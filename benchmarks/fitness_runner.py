"""Run a supplied optimizer or phenotype through an evolution fitness split."""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.runner_utils import (
    TASK_CONFIG_NAMES,
    _json_safe,
    create_prebuilt_optimizer,
    create_task_evaluator,
    evaluate_optimizer,
    evaluate_phenotype,
    load_task_parameters,
    prepare_runner_parameters,
    read_phenotype_argument,
    resolve_runner_output_dir,
    serialize_optimizer,
    write_json,
)


DEFAULT_FITNESS_REPEATS = 1
DATASET_PARAMETERS_DIR = REPOSITORY_ROOT / "parameters" / "dataset_parameters"
REQUIRED_DATASET_PARAMETER_FIELDS = {
    "VALIDATION_SIZE",
    "FITNESS_SIZE",
    "BATCH_SIZE",
    "EPOCHS",
    "PATIENCE",
    "MODEL",
    "NORMALIZE",
    "SUBTRACT_MEAN",
}


def load_fitness_parameters(task_name, parameter_file=None):
    """Load SGE defaults plus an optional dataset-parameter JSON overlay."""

    parameters = load_task_parameters(task_name, use_test_data=False)
    if parameter_file is None:
        return parameters

    config_path = _resolve_dataset_parameter_file(parameter_file)
    with config_path.open() as input_file:
        loaded = json.load(input_file)
    if not isinstance(loaded, dict):
        raise ValueError(f"Dataset parameter file must contain a JSON object: {config_path}")
    _validate_dataset_parameter_file(config_path, loaded)
    parameters.update(loaded)
    parameters["PARAMETERS"] = str(config_path)
    return parameters


def _validate_dataset_parameter_file(config_path, loaded):
    missing_fields = REQUIRED_DATASET_PARAMETER_FIELDS - set(loaded)
    if missing_fields:
        raise ValueError(
            "Dataset parameter file is incomplete and would silently fall back "
            "to SGE defaults for actual evaluation settings. Add the missing "
            f"fields to {config_path}: {', '.join(sorted(missing_fields))}"
        )


def _resolve_dataset_parameter_file(parameter_file):
    requested_path = Path(parameter_file)
    if requested_path.is_absolute():
        config_path = requested_path
    elif len(requested_path.parts) == 1:
        config_path = DATASET_PARAMETERS_DIR / requested_path
    else:
        config_path = (REPOSITORY_ROOT / requested_path).resolve()

    config_path = config_path.resolve()
    parameters_dir = DATASET_PARAMETERS_DIR.resolve()
    if config_path.suffix.lower() != ".json":
        raise ValueError("fitness_runner dataset parameter files must be JSON files")
    if parameters_dir not in config_path.parents:
        raise ValueError(
            "fitness_runner --parameters only accepts files under "
            f"{DATASET_PARAMETERS_DIR}"
        )
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset parameter file not found: {config_path}")
    return config_path


def _prepare_fitness_parameters(parameters, output_dir):
    return prepare_runner_parameters(parameters, output_dir, "fitness_runner")


def _read_jsonl(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def _validate_or_write_manifest(path, manifest):
    if path.exists():
        with path.open() as manifest_file:
            existing_manifest = json.load(manifest_file)
        if existing_manifest != manifest:
            raise ValueError(
                "Existing fitness runs belong to different inputs. Use another "
                "output directory or remove the old fitness artifacts."
            )
    else:
        write_json(path, manifest)


def _write_fitness_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["run", "score"])
        writer.writeheader()
        writer.writerows(
            {"run": record["run"], "score": record["score"]} for record in records
        )
    temporary_path.replace(path)


def _write_fitness_artifacts(output_dir, records, requested_runs, summary_fields):
    scores = [float(record["score"]) for record in records]
    summary = dict(summary_fields)
    summary.update(
        {
            "requested_runs": requested_runs,
            "runs": len(scores),
            "complete": len(scores) >= requested_runs,
            "mean_score": sum(scores) / len(scores) if scores else None,
            "median_score": statistics.median(scores) if scores else None,
            "min_score": min(scores) if scores else None,
            "max_score": max(scores) if scores else None,
            "scores": scores,
        }
    )
    if len(scores) >= 2:
        summary["std_score"] = statistics.stdev(scores)
    else:
        summary["std_score"] = None
    write_json(Path(output_dir) / "fitness_summary.json", summary)
    _write_fitness_csv(Path(output_dir) / "fitness_runs.csv", records)
    return summary


def run_fitness_subject(
    task_name,
    parameters,
    output_dir,
    repeats=DEFAULT_FITNESS_REPEATS,
    optimizer_name=None,
    phenotype=None,
    evaluator=None,
):
    """Evaluate one supplied optimizer or phenotype on the evolution fitness split."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if bool(optimizer_name) == bool(phenotype):
        raise ValueError("Provide exactly one of optimizer_name or phenotype")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = _prepare_fitness_parameters(parameters, output_dir)
    results_path = output_dir / "fitness_runs.jsonl"
    manifest_path = output_dir / "fitness_manifest.json"

    if optimizer_name:
        base_optimizer = create_prebuilt_optimizer(optimizer_name)
        optimizer_spec = serialize_optimizer(base_optimizer)
        manifest = {
            "task_name": task_name.lower().replace("-", "_"),
            "assessment_split": "fitness",
            "subject_type": "optimizer",
            "optimizer_name": optimizer_name,
            "optimizer": optimizer_spec,
        }
        summary_fields = dict(manifest)
    else:
        manifest = {
            "task_name": task_name.lower().replace("-", "_"),
            "assessment_split": "fitness",
            "subject_type": "phenotype",
            "phenotype": phenotype,
        }
        summary_fields = dict(manifest)

    _validate_or_write_manifest(manifest_path, manifest)
    existing_results = _read_jsonl(results_path)
    if len(existing_results) > repeats:
        raise ValueError(
            f"{results_path} already contains {len(existing_results)} runs, "
            f"more than the requested {repeats}"
        )
    _write_fitness_artifacts(
        output_dir, existing_results[:repeats], repeats, summary_fields
    )

    evaluator = evaluator or create_task_evaluator(
        task_name,
        parameters,
        use_validation_data=False,
        use_test_data=False,
    )
    with results_path.open("a") as results_file:
        for run_index in range(len(existing_results), repeats):
            if optimizer_name:
                optimizer = create_prebuilt_optimizer(optimizer_name)
                score, details = evaluate_optimizer(evaluator, optimizer)
            else:
                score, details = evaluate_phenotype(
                    evaluator,
                    phenotype,
                    task_name,
                    parameters,
                )
            record = {
                "run": run_index + 1,
                "score": score,
                "assessment_split": "fitness",
                "details": _json_safe(details),
            }
            results_file.write(json.dumps(record, sort_keys=True) + "\n")
            results_file.flush()
            existing_results.append(record)
            _write_fitness_artifacts(
                output_dir,
                existing_results[:repeats],
                repeats,
                summary_fields,
            )

    return _write_fitness_artifacts(
        output_dir,
        existing_results[:repeats],
        repeats,
        summary_fields,
    )


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a supplied optimizer or phenotype on the fitness split."
    )
    subject_group = parser.add_mutually_exclusive_group(required=True)
    subject_group.add_argument("--phenotype", help="Full optimizer phenotype")
    subject_group.add_argument("--phenotype-file", help="File containing a phenotype")
    subject_group.add_argument(
        "--optimizer",
        choices=["adam"],
        help="Supported prebuilt TensorFlow optimizer",
    )
    parser.add_argument("--task", required=True, choices=sorted(TASK_CONFIG_NAMES))
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Experiment output folder name under dumps/benchmarks, or an "
            "absolute path to write elsewhere."
        ),
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_FITNESS_REPEATS)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--parameters",
        help=(
            "JSON dataset parameter file from parameters/dataset_parameters, "
            "for example FMNIST_CONFIG.json"
        ),
    )
    return parser.parse_args(arguments)


def main(arguments=None):
    args = parse_args(arguments)
    output_dir = resolve_runner_output_dir(args.output_dir)
    if args.seed is not None:
        import random
        import numpy as np
        import tensorflow as tf

        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    parameters = load_fitness_parameters(args.task, args.parameters)
    phenotype = None if args.optimizer else read_phenotype_argument(args)
    summary = run_fitness_subject(
        task_name=args.task,
        parameters=parameters,
        output_dir=output_dir,
        repeats=args.repeats,
        optimizer_name=args.optimizer,
        phenotype=phenotype,
    )
    print(f"Fitness mean over {summary['runs']} runs: {summary['mean_score']}")


if __name__ == "__main__":
    main()
