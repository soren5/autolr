"""Tune an evolved phenotype or prebuilt optimizer, then benchmark it repeatedly.

The Optuna study and benchmark artifacts are deliberately separate. Tuning
selects an optimizer using benchmark-layout validation data, while benchmarking
measures the selected optimizer on held-out test data without feeding those
results back into tuning.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.runner_utils import (
    DEFAULT_SEARCH_HIGH,
    DEFAULT_SEARCH_LOW,
    TASK_CONFIG_NAMES,
    _json_safe,
    create_prebuilt_optimizer,
    create_task_evaluator,
    default_optimizer_search_space,
    evaluate_optimizer,
    evaluate_phenotype,
    load_task_parameters,
    materialize_optimizer,
    prepare_evaluator_for_test_assessment,
    prepare_evaluator_for_validation_assessment,
    prepare_runner_parameters,
    read_phenotype_argument,
    resolve_runner_output_dir,
    serialize_optimizer,
    write_dataframe_csv,
    write_json,
)
from utils.smart_phenotype import abstract_active_constants, materialize_constants


DEFAULT_BENCHMARK_REPEATS = 30
OPTUNA_HEARTBEAT_INTERVAL = 60
OPTUNA_HEARTBEAT_GRACE_PERIOD = 300


def _suggest_optimizer_parameters(trial, search_space):
    values = {}
    for name, specification in search_space.items():
        parameter_type = specification.get("type", "float")
        if parameter_type == "float":
            values[name] = trial.suggest_float(
                name,
                specification["low"],
                specification["high"],
                log=specification.get("log", False),
                step=specification.get("step"),
            )
        elif parameter_type == "int":
            values[name] = trial.suggest_int(
                name,
                specification["low"],
                specification["high"],
                step=specification.get("step", 1),
                log=specification.get("log", False),
            )
        elif parameter_type == "categorical":
            values[name] = trial.suggest_categorical(name, specification["choices"])
        else:
            raise ValueError(f"Unsupported search-space type {parameter_type!r} for {name}")
    return values


def _validate_probe_values(probe_values, search_space):
    """Validate that a default-parameter probe belongs to its search space."""

    for name, value in probe_values.items():
        specification = search_space[name]
        parameter_type = specification.get("type", "float")
        if parameter_type in {"float", "int"}:
            if not specification["low"] <= value <= specification["high"]:
                raise ValueError(
                    f"Default probe value {name}={value!r} is outside its "
                    f"search space [{specification['low']}, {specification['high']}]"
                )
        elif parameter_type == "categorical":
            if value not in specification["choices"]:
                raise ValueError(
                    f"Default probe value {name}={value!r} is absent from its "
                    f"categorical choices {specification['choices']!r}"
                )


def _enqueue_default_probe(study, probe_values, search_space, remaining_trials):
    """Queue one resume-safe default-parameter trial before sampler trials."""

    if remaining_trials <= 0:
        return
    _validate_probe_values(probe_values, search_space)
    study.enqueue_trial(
        probe_values,
        user_attrs={"source": "default_parameter_probe"},
        skip_if_exists=True,
    )


def _prepare_benchmark_parameters(parameters, output_dir):
    return prepare_runner_parameters(parameters, output_dir, "benchmark_runner")


def _create_optuna_storage(optuna, output_dir):
    storage_path = (Path(output_dir) / "optuna_study.sqlite3").resolve()
    return optuna.storages.RDBStorage(
        url=f"sqlite:///{storage_path}",
        heartbeat_interval=OPTUNA_HEARTBEAT_INTERVAL,
        grace_period=OPTUNA_HEARTBEAT_GRACE_PERIOD,
    )


def _write_tuning_status(study, output_dir, requested_completed_trials):
    state_counts = {}
    for trial in study.trials:
        state_counts[trial.state.name.lower()] = (
            state_counts.get(trial.state.name.lower(), 0) + 1
        )
    completed_trials = state_counts.get("complete", 0)
    status = {
        "study_name": study.study_name,
        "requested_completed_trials": requested_completed_trials,
        "completed_trials": completed_trials,
        "remaining_completed_trials": max(
            0, requested_completed_trials - completed_trials
        ),
        "trial_state_counts": state_counts,
        "total_trials": len(study.trials),
    }
    write_json(Path(output_dir) / "tuning_status.json", status)


def _write_study_artifacts(
    study,
    output_dir,
    phenotype_template,
    tunable_parameters,
    requested_completed_trials=None,
):
    trials_path = output_dir / "tuning_trials.csv"
    write_dataframe_csv(trials_path, study.trials_dataframe())
    if requested_completed_trials is not None:
        _write_tuning_status(study, output_dir, requested_completed_trials)
    completed_trials = [
        trial for trial in study.trials if trial.state.name == "COMPLETE"
    ]
    if not completed_trials:
        return
    best = {
        "study_name": study.study_name,
        "completed_trials": len(completed_trials),
        "best_score": study.best_value,
        "best_parameters": study.best_params,
        "tunable_parameters": tunable_parameters,
        "phenotype": materialize_constants(phenotype_template, study.best_params),
    }
    write_json(output_dir / "best_tuned_phenotype.json", best)


def _write_optimizer_study_artifacts(
    study,
    output_dir,
    optimizer_spec,
    search_space,
    requested_completed_trials=None,
):
    write_dataframe_csv(output_dir / "tuning_trials.csv", study.trials_dataframe())
    if requested_completed_trials is not None:
        _write_tuning_status(study, output_dir, requested_completed_trials)
    completed_trials = [
        trial for trial in study.trials if trial.state.name == "COMPLETE"
    ]
    if not completed_trials:
        return
    tuned_optimizer = materialize_optimizer(optimizer_spec, study.best_params)
    best = {
        "study_name": study.study_name,
        "completed_trials": len(completed_trials),
        "best_score": study.best_value,
        "best_parameters": study.best_params,
        "search_space": search_space,
        "optimizer": serialize_optimizer(tuned_optimizer),
    }
    write_json(output_dir / "best_tuned_optimizer.json", best)


def _validate_or_record_study_inputs(study, expected_inputs):
    saved_inputs = study.user_attrs.get("benchmark_inputs")
    if saved_inputs is not None and saved_inputs != expected_inputs:
        raise ValueError(
            "The existing Optuna study was created for different benchmark "
            "inputs. Use another study name or output directory."
        )
    study.set_user_attr("benchmark_inputs", expected_inputs)


def tune_phenotype(
    phenotype,
    task_name,
    parameters,
    n_trials,
    output_dir,
    study_name="autolr_optimizer_tuning",
    search_low=DEFAULT_SEARCH_LOW,
    search_high=DEFAULT_SEARCH_HIGH,
    timeout=None,
    evaluator=None,
    seed=None,
):
    """Tune active phenotype constants in a resumable SQLite Optuna study."""

    try:
        import optuna
    except ImportError as error:
        raise ImportError(
            "Optuna is required for optimizer tuning. Install it in the active "
            "environment before running this benchmark."
        ) from error

    if n_trials < 0:
        raise ValueError("n_trials must be non-negative")
    if search_low <= 0 or search_high <= search_low:
        raise ValueError("Log-scale search bounds require 0 < low < high")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = _prepare_benchmark_parameters(parameters, output_dir)
    phenotype_template, tunable_parameters = abstract_active_constants(phenotype)
    evaluator = evaluator or create_task_evaluator(
        task_name, parameters, use_validation_data=True
    )
    storage = _create_optuna_storage(optuna, output_dir)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )
    sampler_seed = None if seed is None else seed + len(study.trials)
    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
    )
    benchmark_inputs = {
        "task_name": task_name.lower().replace("-", "_"),
        "phenotype_template": phenotype_template,
        "tunable_parameters": tunable_parameters,
        "search_low": search_low,
        "search_high": search_high,
    }
    _validate_or_record_study_inputs(study, benchmark_inputs)
    study.set_user_attr("task_name", task_name)
    study.set_user_attr("phenotype_template", phenotype_template)
    study.set_user_attr("tunable_parameters", tunable_parameters)

    if not tunable_parameters:
        raise ValueError("The phenotype has no active tf.constant values to tune")

    def objective(trial):
        values = {}
        for parameter_name in tunable_parameters:
            values[parameter_name] = trial.suggest_float(
                parameter_name,
                search_low,
                search_high,
                log=True,
            )
        candidate = materialize_constants(phenotype_template, values)
        score, details = evaluate_phenotype(evaluator, candidate, task_name, parameters)
        trial.set_user_attr("phenotype", candidate)
        trial.set_user_attr("details", _json_safe(details))
        return score

    completed_trials = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining_trials = max(0, n_trials - completed_trials)
    phenotype_search_space = {
        parameter_name: {
            "type": "float",
            "low": search_low,
            "high": search_high,
            "log": True,
        }
        for parameter_name in tunable_parameters
    }
    _enqueue_default_probe(
        study,
        probe_values=tunable_parameters,
        search_space=phenotype_search_space,
        remaining_trials=remaining_trials,
    )
    artifact_callback = lambda current_study, _: _write_study_artifacts(
        current_study,
        output_dir,
        phenotype_template,
        tunable_parameters,
        requested_completed_trials=n_trials,
    )
    _write_study_artifacts(
        study,
        output_dir,
        phenotype_template,
        tunable_parameters,
        requested_completed_trials=n_trials,
    )
    if remaining_trials:
        try:
            study.optimize(
                objective,
                n_trials=remaining_trials,
                timeout=timeout,
                callbacks=[artifact_callback],
            )
        finally:
            _write_study_artifacts(
                study,
                output_dir,
                phenotype_template,
                tunable_parameters,
                requested_completed_trials=n_trials,
            )
    else:
        _write_study_artifacts(
            study,
            output_dir,
            phenotype_template,
            tunable_parameters,
            requested_completed_trials=n_trials,
        )
    return study


def tune_optimizer(
    optimizer,
    task_name,
    parameters,
    n_trials,
    output_dir,
    search_space=None,
    study_name="autolr_optimizer_tuning",
    timeout=None,
    evaluator=None,
    seed=None,
):
    """Tune a prebuilt TensorFlow optimizer in a resumable Optuna study."""

    try:
        import optuna
    except ImportError as error:
        raise ImportError(
            "Optuna is required for optimizer tuning. Install it in the active "
            "environment before running this benchmark."
        ) from error

    if n_trials < 0:
        raise ValueError("n_trials must be non-negative")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = _prepare_benchmark_parameters(parameters, output_dir)
    optimizer_spec = serialize_optimizer(optimizer)
    search_space = _json_safe(search_space or default_optimizer_search_space(optimizer))
    if not search_space:
        raise ValueError("The optimizer search space cannot be empty")
    unknown_parameters = set(search_space) - set(optimizer_spec["config"])
    if unknown_parameters:
        raise ValueError(
            "Search-space parameters are absent from the optimizer config: "
            + ", ".join(sorted(unknown_parameters))
        )

    evaluator = evaluator or create_task_evaluator(
        task_name, parameters, use_validation_data=True
    )
    storage = _create_optuna_storage(optuna, output_dir)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )
    sampler_seed = None if seed is None else seed + len(study.trials)
    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
    )
    benchmark_inputs = {
        "task_name": task_name.lower().replace("-", "_"),
        "optimizer": optimizer_spec,
        "search_space": search_space,
    }
    _validate_or_record_study_inputs(study, benchmark_inputs)
    study.set_user_attr("task_name", task_name)
    study.set_user_attr("optimizer_spec", optimizer_spec)
    study.set_user_attr("optimizer_search_space", search_space)

    def objective(trial):
        values = _suggest_optimizer_parameters(trial, search_space)
        candidate = materialize_optimizer(optimizer_spec, values)
        score, details = evaluate_optimizer(evaluator, candidate)
        trial.set_user_attr("optimizer", serialize_optimizer(candidate))
        trial.set_user_attr("details", _json_safe(details))
        return score

    completed_trials = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining_trials = max(0, n_trials - completed_trials)
    optimizer_defaults = {
        parameter_name: optimizer_spec["config"][parameter_name]
        for parameter_name in search_space
    }
    _enqueue_default_probe(
        study,
        probe_values=optimizer_defaults,
        search_space=search_space,
        remaining_trials=remaining_trials,
    )
    artifact_callback = lambda current_study, _: _write_optimizer_study_artifacts(
        current_study,
        output_dir,
        optimizer_spec,
        search_space,
        requested_completed_trials=n_trials,
    )
    _write_optimizer_study_artifacts(
        study,
        output_dir,
        optimizer_spec,
        search_space,
        requested_completed_trials=n_trials,
    )
    if remaining_trials:
        try:
            study.optimize(
                objective,
                n_trials=remaining_trials,
                timeout=timeout,
                callbacks=[artifact_callback],
            )
        finally:
            _write_optimizer_study_artifacts(
                study,
                output_dir,
                optimizer_spec,
                search_space,
                requested_completed_trials=n_trials,
            )
    else:
        _write_optimizer_study_artifacts(
            study,
            output_dir,
            optimizer_spec,
            search_space,
            requested_completed_trials=n_trials,
        )
    return study


def benchmark_best_phenotype(
    study,
    task_name,
    parameters,
    output_dir,
    repeats=DEFAULT_BENCHMARK_REPEATS,
    evaluator=None,
):
    """Evaluate the study's best phenotype repeatedly and save the results."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = _prepare_benchmark_parameters(parameters, output_dir)
    results_path = output_dir / "benchmark_runs.jsonl"
    manifest_path = output_dir / "benchmark_manifest.json"
    phenotype_template = study.user_attrs["phenotype_template"]
    phenotype = materialize_constants(phenotype_template, study.best_params)
    manifest = {
        "task_name": task_name.lower().replace("-", "_"),
        "assessment_split": "test",
        "study_name": study.study_name,
        "best_parameters": study.best_params,
        "phenotype": phenotype,
    }
    _validate_or_write_manifest(
        manifest_path,
        manifest,
        "Existing benchmark runs belong to a different best phenotype. "
        "Use another output directory or remove the old benchmark artifacts.",
    )

    existing_results = _read_jsonl(results_path)
    if len(existing_results) > repeats:
        raise ValueError(
            f"{results_path} already contains {len(existing_results)} runs, "
            f"more than the requested {repeats}"
        )
    summary_fields = {
        "task_name": task_name,
        "assessment_split": "test",
        "study_name": study.study_name,
        "best_tuning_score": study.best_value,
        "best_parameters": study.best_params,
        "phenotype": phenotype,
    }
    _write_benchmark_artifacts(
        output_dir, existing_results[:repeats], repeats, summary_fields
    )

    if evaluator is None:
        evaluator = create_task_evaluator(task_name, parameters, use_test_data=True)
    elif getattr(evaluator, "assessment_split", None) != "test":
        prepare_evaluator_for_test_assessment(
            evaluator,
            expected_test_size=parameters.get("TEST_SIZE"),
        )

    with results_path.open("a") as results_file:
        for run_index in range(len(existing_results), repeats):
            score, details = evaluate_phenotype(evaluator, phenotype, task_name, parameters)
            record = {
                "run": run_index + 1,
                "score": score,
                "assessment_split": "test",
                "details": _json_safe(details),
            }
            results_file.write(json.dumps(record, sort_keys=True) + "\n")
            results_file.flush()
            existing_results.append(record)
            _write_benchmark_artifacts(
                output_dir,
                existing_results[:repeats],
                repeats,
                summary_fields,
            )

    return _write_benchmark_artifacts(
        output_dir,
        existing_results[:repeats],
        repeats,
        summary_fields,
    )


def benchmark_best_optimizer(
    study,
    task_name,
    parameters,
    output_dir,
    repeats=DEFAULT_BENCHMARK_REPEATS,
    evaluator=None,
):
    """Repeatedly benchmark the best tuned prebuilt optimizer."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = _prepare_benchmark_parameters(parameters, output_dir)
    results_path = output_dir / "benchmark_runs.jsonl"
    manifest_path = output_dir / "benchmark_manifest.json"
    optimizer_spec = study.user_attrs["optimizer_spec"]
    tuned_spec = serialize_optimizer(
        materialize_optimizer(optimizer_spec, study.best_params)
    )
    manifest = {
        "task_name": task_name.lower().replace("-", "_"),
        "assessment_split": "test",
        "study_name": study.study_name,
        "best_parameters": study.best_params,
        "optimizer": tuned_spec,
    }
    _validate_or_write_manifest(
        manifest_path,
        manifest,
        "Existing benchmark runs belong to a different best optimizer. "
        "Use another output directory or remove the old benchmark artifacts.",
    )

    existing_results = _read_jsonl(results_path)
    if len(existing_results) > repeats:
        raise ValueError(
            f"{results_path} already contains {len(existing_results)} runs, "
            f"more than the requested {repeats}"
        )
    summary_fields = {
        "task_name": task_name,
        "assessment_split": "test",
        "study_name": study.study_name,
        "best_tuning_score": study.best_value,
        "best_parameters": study.best_params,
        "optimizer": tuned_spec,
    }
    _write_benchmark_artifacts(
        output_dir, existing_results[:repeats], repeats, summary_fields
    )

    if evaluator is None:
        evaluator = create_task_evaluator(task_name, parameters, use_test_data=True)
    elif getattr(evaluator, "assessment_split", None) != "test":
        prepare_evaluator_for_test_assessment(
            evaluator,
            expected_test_size=parameters.get("TEST_SIZE"),
        )
    with results_path.open("a") as results_file:
        for run_index in range(len(existing_results), repeats):
            candidate = materialize_optimizer(optimizer_spec, study.best_params)
            score, details = evaluate_optimizer(evaluator, candidate)
            record = {
                "run": run_index + 1,
                "score": score,
                "assessment_split": "test",
                "details": _json_safe(details),
            }
            results_file.write(json.dumps(record, sort_keys=True) + "\n")
            results_file.flush()
            existing_results.append(record)
            _write_benchmark_artifacts(
                output_dir,
                existing_results[:repeats],
                repeats,
                summary_fields,
            )

    return _write_benchmark_artifacts(
        output_dir,
        existing_results[:repeats],
        repeats,
        summary_fields,
    )


def _read_jsonl(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def _validate_or_write_manifest(path, manifest, mismatch_message):
    if path.exists():
        with path.open() as manifest_file:
            existing_manifest = json.load(manifest_file)
        if existing_manifest != manifest:
            raise ValueError(mismatch_message)
    else:
        write_json(path, manifest)


def _write_benchmark_artifacts(output_dir, records, requested_runs, summary_fields):
    scores = [float(record["score"]) for record in records]
    summary = dict(summary_fields)
    summary.update(
        {
            "requested_runs": requested_runs,
            "runs": len(scores),
            "complete": len(scores) >= requested_runs,
            "mean_score": sum(scores) / len(scores) if scores else None,
            "min_score": min(scores) if scores else None,
            "max_score": max(scores) if scores else None,
            "scores": scores,
        }
    )
    write_json(Path(output_dir) / "benchmark_summary.json", summary)
    _write_benchmark_csv(Path(output_dir) / "benchmark_runs.csv", records)
    return summary


def _write_benchmark_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["run", "score"])
        writer.writeheader()
        writer.writerows(
            {"run": record["run"], "score": record["score"]} for record in records
        )
    temporary_path.replace(path)


def _read_phenotype(args):
    return read_phenotype_argument(args)


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(
        description="Tune a phenotype or prebuilt optimizer and benchmark it."
    )
    subject_group = parser.add_mutually_exclusive_group(required=True)
    subject_group.add_argument("--phenotype", help="Full optimizer phenotype")
    subject_group.add_argument("--phenotype-file", help="File containing a phenotype")
    subject_group.add_argument(
        "--optimizer",
        choices=["adam"],
        help="Supported prebuilt TensorFlow optimizer with its default search space",
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
    parser.add_argument("--study-name", default="autolr_optimizer_tuning")
    parser.add_argument("--trials", type=int, default=100, help="Desired total trial count")
    parser.add_argument("--benchmark-repeats", type=int, default=DEFAULT_BENCHMARK_REPEATS)
    parser.add_argument(
        "--search-low",
        type=float,
        default=DEFAULT_SEARCH_LOW,
        help="Lower bound for phenotype constants; ignored for prebuilt optimizers",
    )
    parser.add_argument(
        "--search-high",
        type=float,
        default=DEFAULT_SEARCH_HIGH,
        help="Upper bound for phenotype constants; ignored for prebuilt optimizers",
    )
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--tune-only",
        action="store_true",
        help="Tune without running the repeated final benchmark",
    )
    return parser.parse_args(arguments)


def main(arguments=None):
    args = parse_args(arguments)
    output_dir = resolve_runner_output_dir(args.output_dir)
    parameters = load_task_parameters(args.task, use_test_data=True)
    if args.optimizer:
        optimizer = create_prebuilt_optimizer(args.optimizer)
        study = tune_optimizer(
            optimizer=optimizer,
            task_name=args.task,
            parameters=parameters,
            n_trials=args.trials,
            output_dir=output_dir,
            study_name=args.study_name,
            timeout=args.timeout,
            seed=args.seed,
        )
    else:
        phenotype = _read_phenotype(args)
        study = tune_phenotype(
            phenotype=phenotype,
            task_name=args.task,
            parameters=parameters,
            n_trials=args.trials,
            output_dir=output_dir,
            study_name=args.study_name,
            search_low=args.search_low,
            search_high=args.search_high,
            timeout=args.timeout,
            seed=args.seed,
        )
    print(f"Best tuning score: {study.best_value}")
    print(f"Best parameters: {study.best_params}")
    if not args.tune_only:
        benchmark_function = (
            benchmark_best_optimizer if args.optimizer else benchmark_best_phenotype
        )
        summary = benchmark_function(
            study=study,
            task_name=args.task,
            parameters=parameters,
            output_dir=output_dir,
            repeats=args.benchmark_repeats,
        )
        print(f"Benchmark mean over {summary['runs']} runs: {summary['mean_score']}")


if __name__ == "__main__":
    main()
