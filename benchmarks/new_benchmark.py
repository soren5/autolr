"""Tune an evolved phenotype or prebuilt optimizer, then benchmark it repeatedly.

The Optuna study and benchmark artifacts are deliberately separate. Tuning
selects an optimizer using benchmark-layout validation data, while benchmarking
measures the selected optimizer on held-out test data without feeding those
results back into tuning.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.smart_phenotype import abstract_active_constants, materialize_constants


DEFAULT_BENCHMARK_REPEATS = 30
DEFAULT_SEARCH_LOW = 1e-8
DEFAULT_SEARCH_HIGH = 1.0
BENCHMARK_CONFIG_DIR = Path(__file__).resolve().parent / "benchmark_dataset_configs"
TASK_CONFIG_NAMES = {
    "fmnist": "FMNIST",
    "mnist": "MNIST",
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "tiny_imagenet": "TINY_IMAGENET",
}
ADAM_SEARCH_SPACE = {
    "learning_rate": {"type": "float", "low": DEFAULT_SEARCH_LOW, "high": DEFAULT_SEARCH_HIGH},
    "beta_1": {"type": "float", "low": DEFAULT_SEARCH_LOW, "high": DEFAULT_SEARCH_HIGH},
    "beta_2": {"type": "float", "low": DEFAULT_SEARCH_LOW, "high": DEFAULT_SEARCH_HIGH},
    "epsilon": {"type": "float", "low": DEFAULT_SEARCH_LOW, "high": DEFAULT_SEARCH_HIGH},
}


def create_task_evaluator(
    task_name, parameters, use_validation_data=False, use_test_data=False
):
    """Create the current framework evaluator associated with ``task_name``.

    Benchmark tuning assesses validation accuracy. Final benchmarking assesses
    the held-out test set. Both paths use the existing evaluator assessment
    machinery by selecting which dataset split is exposed as ``x_fit/y_fit``.
    """

    if use_validation_data and use_test_data:
        raise ValueError("An evaluator cannot assess validation and test data together")

    task = task_name.lower().replace("-", "_")
    if task == "fmnist":
        from evaluators.evaluate_fmnist import FMNIST_Evaluator

        evaluator = FMNIST_Evaluator(
            parameters,
            task_name="fmnist",
            benchmark_data=use_validation_data or use_test_data,
        )
    elif task == "mnist":
        from evaluators.evaluate_mnist import MNIST_Evaluator

        evaluator = MNIST_Evaluator(
            parameters,
            task_name="mnist",
            benchmark_data=use_validation_data or use_test_data,
        )
    elif task == "cifar10":
        from evaluators.evaluate_cifar10 import CIFAR10_Evaluator

        evaluator = CIFAR10_Evaluator(
            parameters,
            task_name="cifar10",
            benchmark_data=use_validation_data or use_test_data,
        )
    elif task == "cifar100":
        from evaluators.evaluate_cifar100 import CIFAR100_Evaluator

        evaluator = CIFAR100_Evaluator(
            parameters,
            task_name="cifar100",
            benchmark_data=use_validation_data or use_test_data,
        )
    elif task in {"tiny_imagenet", "tinyimagenet"}:
        from evaluators.evaluate_tiny_imagenet import TINY_IMAGENET_Evaluator

        evaluator = TINY_IMAGENET_Evaluator(
            parameters,
            task_name="tiny_imagenet",
            benchmark_data=use_validation_data or use_test_data,
        )
    else:
        raise ValueError(
            f"Unknown task {task_name!r}. Supported tasks: fmnist, mnist, cifar10, "
            "cifar100, tiny_imagenet."
        )
    evaluator.assessment_split = "fitness"
    if use_validation_data:
        prepare_evaluator_for_validation_assessment(
            evaluator,
            expected_test_size=parameters.get("TEST_SIZE"),
        )
    elif use_test_data:
        prepare_evaluator_for_test_assessment(
            evaluator,
            expected_test_size=parameters.get("TEST_SIZE"),
        )
    return evaluator


def _load_benchmark_dataset(evaluator, expected_test_size=None):
    """Load and validate the benchmark-layout dataset for an evaluator."""

    dataset = getattr(evaluator, "dataset", None)
    if dataset is None or not hasattr(dataset, "load_data_for_benchmark"):
        raise ValueError(
            "Benchmark assessment requires an evaluator whose dataset implements "
            "load_data_for_benchmark()."
        )
    if not getattr(dataset, "_benchmark_data_loaded", False):
        if expected_test_size is not None:
            dataset.test_size = expected_test_size
        dataset.load_data_for_benchmark()
        dataset._benchmark_data_loaded = True
    if not all(hasattr(dataset, name) for name in ("x_test", "y_test")):
        raise ValueError("Benchmark dataset did not expose x_test and y_test")
    return dataset


def prepare_evaluator_for_validation_assessment(evaluator, expected_test_size=None):
    """Load benchmark-layout data and assess Optuna trials on validation data."""

    dataset = _load_benchmark_dataset(evaluator, expected_test_size)
    if not all(hasattr(dataset, name) for name in ("x_val", "y_val")):
        raise ValueError("Validation assessment requires dataset x_val and y_val")

    dataset.x_fit = dataset.x_val
    dataset.y_fit = dataset.y_val
    evaluator.assessment_split = "validation"
    return evaluator


def prepare_evaluator_for_test_assessment(evaluator, expected_test_size=None):
    """Reload an evaluator's dataset for held-out test-set assessment."""

    dataset = _load_benchmark_dataset(evaluator, expected_test_size)

    # Evaluator.train_model assesses on x_fit/y_fit. Point that existing,
    # well-tested path at the held-out benchmark data for final assessment.
    dataset.x_fit = dataset.x_test
    dataset.y_fit = dataset.y_test
    evaluator.assessment_split = "test"
    return evaluator


def evaluate_phenotype(evaluator, phenotype, task_name, parameters):
    """Return a maximized score and evaluator details for one phenotype run."""

    score, details = evaluator.evaluate(phenotype)
    details = dict(details)
    assessment_split = getattr(evaluator, "assessment_split", "unknown")
    details["assessment_split"] = assessment_split
    details[f"{assessment_split}_score"] = float(score)
    return float(score), details


def evaluate_optimizer(evaluator, optimizer):
    """Return a maximized score and evaluator details for one prebuilt optimizer."""

    score, details = evaluator.evaluate_optimizer(optimizer)
    details = dict(details)
    assessment_split = getattr(evaluator, "assessment_split", "unknown")
    details["assessment_split"] = assessment_split
    details[f"{assessment_split}_score"] = float(score)
    return float(score), details


def create_prebuilt_optimizer(name):
    """Create a supported standard TensorFlow optimizer by CLI name."""

    if name.lower() == "adam":
        from tensorflow.keras.optimizers import Adam
        opt = Adam()
        opt.name = "Adam" 
        return opt
    raise ValueError(f"Unknown prebuilt optimizer {name!r}. Supported optimizers: adam.")


def default_optimizer_search_space(optimizer):
    """Return the default parameter search space for a supported optimizer."""

    if optimizer.__class__.__name__.lower() == "adam":
        return ADAM_SEARCH_SPACE
    raise ValueError(
        f"No default search space is defined for {optimizer.__class__.__name__}. "
        "Supply search_space explicitly."
    )


def serialize_optimizer(optimizer):
    """Return a JSON-safe Keras optimizer specification."""

    from tensorflow.keras.optimizers import serialize

    return _json_safe(serialize(optimizer))


def materialize_optimizer(optimizer_spec, parameter_values):
    """Create a fresh optimizer from a serialized base and tuned values."""

    from tensorflow.keras.optimizers import deserialize

    specification = {
        "class_name": optimizer_spec["class_name"],
        "config": dict(optimizer_spec["config"]),
    }
    specification["config"].update(parameter_values)
    return deserialize(specification)


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


def _json_safe(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output_file:
        json.dump(_json_safe(value), output_file, indent=2, sort_keys=True)


def _write_study_artifacts(study, output_dir, phenotype_template, tunable_parameters):
    trials_path = output_dir / "tuning_trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)
    best = {
        "study_name": study.study_name,
        "completed_trials": sum(
            trial.state.name == "COMPLETE" for trial in study.trials
        ),
        "best_score": study.best_value,
        "best_parameters": study.best_params,
        "tunable_parameters": tunable_parameters,
        "phenotype": materialize_constants(phenotype_template, study.best_params),
    }
    _write_json(output_dir / "best_tuned_phenotype.json", best)


def _write_optimizer_study_artifacts(study, output_dir, optimizer_spec, search_space):
    study.trials_dataframe().to_csv(output_dir / "tuning_trials.csv", index=False)
    tuned_optimizer = materialize_optimizer(optimizer_spec, study.best_params)
    best = {
        "study_name": study.study_name,
        "completed_trials": sum(
            trial.state.name == "COMPLETE" for trial in study.trials
        ),
        "best_score": study.best_value,
        "best_parameters": study.best_params,
        "search_space": search_space,
        "optimizer": serialize_optimizer(tuned_optimizer),
    }
    _write_json(output_dir / "best_tuned_optimizer.json", best)


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
    phenotype_template, tunable_parameters = abstract_active_constants(phenotype)
    evaluator = evaluator or create_task_evaluator(
        task_name, parameters, use_validation_data=True
    )
    storage_path = (output_dir / "optuna_study.sqlite3").resolve()
    storage = f"sqlite:///{storage_path}"
    # Load once before constructing the seeded sampler. Optuna persists trials
    # but not sampler RNG state; offsetting by the saved count avoids replaying
    # the sampler's first suggestions after a resumed seeded run.
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

    # n_trials is the desired number of completed trials. Failed or interrupted
    # attempts remain visible in Optuna but do not consume the requested cap.
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
    if remaining_trials:
        study.optimize(objective, n_trials=remaining_trials, timeout=timeout)
    _write_study_artifacts(study, output_dir, phenotype_template, tunable_parameters)
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
    """Tune a prebuilt TensorFlow optimizer in a resumable Optuna study.

    The supplied optimizer is used only as a configuration template. Every
    evaluation receives a freshly deserialized optimizer so training state
    cannot leak between trials.
    """

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
    storage_path = (output_dir / "optuna_study.sqlite3").resolve()
    storage = f"sqlite:///{storage_path}"
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
    if remaining_trials:
        study.optimize(objective, n_trials=remaining_trials, timeout=timeout)
    _write_optimizer_study_artifacts(study, output_dir, optimizer_spec, search_space)
    return study


def benchmark_best_phenotype(
    study,
    task_name,
    parameters,
    output_dir,
    repeats=DEFAULT_BENCHMARK_REPEATS,
    evaluator=None,
):
    """Evaluate the study's best phenotype repeatedly and save the results.

    Existing JSONL rows are retained, so calling this function after an
    interruption evaluates only the remaining runs.
    """

    if repeats <= 0:
        raise ValueError("repeats must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    if manifest_path.exists():
        with manifest_path.open() as manifest_file:
            existing_manifest = json.load(manifest_file)
        if existing_manifest != manifest:
            raise ValueError(
                "Existing benchmark runs belong to a different best phenotype. "
                "Use another output directory or remove the old benchmark artifacts."
            )
    else:
        _write_json(manifest_path, manifest)

    existing_results = []
    if results_path.exists():
        with results_path.open() as results_file:
            existing_results = [json.loads(line) for line in results_file if line.strip()]
    if len(existing_results) > repeats:
        raise ValueError(
            f"{results_path} already contains {len(existing_results)} runs, "
            f"more than the requested {repeats}"
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

    scores = [float(record["score"]) for record in existing_results[:repeats]]
    summary = {
        "task_name": task_name,
        "assessment_split": "test",
        "study_name": study.study_name,
        "best_tuning_score": study.best_value,
        "best_parameters": study.best_params,
        "phenotype": phenotype,
        "runs": len(scores),
        "mean_score": sum(scores) / len(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "scores": scores,
    }
    _write_json(output_dir / "benchmark_summary.json", summary)
    _write_benchmark_csv(output_dir / "benchmark_runs.csv", existing_results[:repeats])
    return summary


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
    if manifest_path.exists():
        with manifest_path.open() as manifest_file:
            existing_manifest = json.load(manifest_file)
        if existing_manifest != manifest:
            raise ValueError(
                "Existing benchmark runs belong to a different best optimizer. "
                "Use another output directory or remove the old benchmark artifacts."
            )
    else:
        _write_json(manifest_path, manifest)

    existing_results = []
    if results_path.exists():
        with results_path.open() as results_file:
            existing_results = [json.loads(line) for line in results_file if line.strip()]
    if len(existing_results) > repeats:
        raise ValueError(
            f"{results_path} already contains {len(existing_results)} runs, "
            f"more than the requested {repeats}"
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

    scores = [float(record["score"]) for record in existing_results[:repeats]]
    summary = {
        "task_name": task_name,
        "assessment_split": "test",
        "study_name": study.study_name,
        "best_tuning_score": study.best_value,
        "best_parameters": study.best_params,
        "optimizer": tuned_spec,
        "runs": len(scores),
        "mean_score": sum(scores) / len(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "scores": scores,
    }
    _write_json(output_dir / "benchmark_summary.json", summary)
    _write_benchmark_csv(output_dir / "benchmark_runs.csv", existing_results[:repeats])
    return summary


def _write_benchmark_csv(path, records):
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["run", "score"])
        writer.writeheader()
        writer.writerows(
            {"run": record["run"], "score": record["score"]} for record in records
        )


def load_task_parameters(task_name, use_test_data=False, config_dir=None):
    """Load the tuning or final-assessment parameters for one benchmark task."""

    from sge.parameters import default_params

    task = task_name.lower().replace("-", "_")
    if task == "tinyimagenet":
        task = "tiny_imagenet"
    if task not in TASK_CONFIG_NAMES:
        supported = ", ".join(sorted(TASK_CONFIG_NAMES))
        raise ValueError(
            f"No benchmark dataset configuration is defined for {task_name!r}. "
            f"Configured tasks: {supported}."
        )

    config_dir = Path(config_dir or BENCHMARK_CONFIG_DIR)
    config_name = TASK_CONFIG_NAMES[task]
    config_paths = (
        [config_dir / f"{config_name}_CONFIG_TEST.json"] if use_test_data else []
    )

    parameters = default_params.copy()
    for config_path in config_paths:
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Missing benchmark configuration for task {task!r}: {config_path}"
            )
        with config_path.open() as input_file:
            loaded = json.load(input_file)
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Benchmark configuration must contain a JSON object: {config_path}"
            )
        parameters.update(loaded)
    return parameters


def _read_phenotype(args):
    if args.phenotype is not None:
        return args.phenotype
    return Path(args.phenotype_file).read_text().strip()


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
    parser.add_argument("--output-dir", required=True)
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
    parameters = load_task_parameters(args.task, use_test_data=True)
    if args.optimizer:
        optimizer = create_prebuilt_optimizer(args.optimizer)
        study = tune_optimizer(
            optimizer=optimizer,
            task_name=args.task,
            parameters=parameters,
            n_trials=args.trials,
            output_dir=args.output_dir,
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
            output_dir=args.output_dir,
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
            output_dir=args.output_dir,
            repeats=args.benchmark_repeats,
        )
        print(f"Benchmark mean over {summary['runs']} runs: {summary['mean_score']}")


if __name__ == "__main__":
    main()
