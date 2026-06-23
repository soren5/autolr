"""Shared helpers for benchmark and fitness runner entry points."""

import json
import math
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


DEFAULT_SEARCH_LOW = 1e-8
DEFAULT_SEARCH_HIGH = 1.0
BENCHMARK_CONFIG_DIR = Path(__file__).resolve().parent / "benchmark_dataset_configs"
DEFAULT_RUNNER_OUTPUT_ROOT = REPOSITORY_ROOT / "dumps" / "benchmarks"
TASK_CONFIG_NAMES = {
    "fmnist": "FMNIST",
    "mnist": "MNIST",
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "tiny_imagenet": "TINY_IMAGENET",
    "tiny_imagenet_custom": "TINY_IMAGENET_CUSTOM",
}
ADAM_SEARCH_SPACE = {
    "learning_rate": {
        "type": "float",
        "low": DEFAULT_SEARCH_LOW,
        "high": DEFAULT_SEARCH_HIGH,
    },
    "beta_1": {"type": "float", "low": DEFAULT_SEARCH_LOW, "high": DEFAULT_SEARCH_HIGH},
    "beta_2": {"type": "float", "low": DEFAULT_SEARCH_LOW, "high": DEFAULT_SEARCH_HIGH},
    "epsilon": {"type": "float", "low": DEFAULT_SEARCH_LOW, "high": DEFAULT_SEARCH_HIGH},
}


def normalize_task_name(task_name):
    task = task_name.lower().replace("-", "_")
    if task == "tinyimagenet":
        task = "tiny_imagenet"
    if task == "tinyimagenet_custom":
        task = "tiny_imagenet_custom"
    return task


def resolve_runner_output_dir(output_dir):
    """Resolve CLI output names under dumps/benchmarks by default."""

    output_path = Path(output_dir).expanduser()
    if output_path.is_absolute():
        return output_path
    return DEFAULT_RUNNER_OUTPUT_ROOT / output_path


def create_task_evaluator(
    task_name, parameters, use_validation_data=False, use_test_data=False
):
    """Create the framework evaluator associated with ``task_name``."""

    if use_validation_data and use_test_data:
        raise ValueError("An evaluator cannot assess validation and test data together")

    task = normalize_task_name(task_name)
    benchmark_data = use_validation_data or use_test_data
    if task == "fmnist":
        from evaluators.evaluate_fmnist import FMNIST_Evaluator

        evaluator = FMNIST_Evaluator(
            parameters,
            task_name="fmnist",
            benchmark_data=benchmark_data,
        )
    elif task == "mnist":
        from evaluators.evaluate_mnist import MNIST_Evaluator

        evaluator = MNIST_Evaluator(
            parameters,
            task_name="mnist",
            benchmark_data=benchmark_data,
        )
    elif task == "cifar10":
        from evaluators.evaluate_cifar10 import CIFAR10_Evaluator

        evaluator = CIFAR10_Evaluator(
            parameters,
            task_name="cifar10",
            benchmark_data=benchmark_data,
        )
    elif task == "cifar100":
        from evaluators.evaluate_cifar100 import CIFAR100_Evaluator

        evaluator = CIFAR100_Evaluator(
            parameters,
            task_name="cifar100",
            benchmark_data=benchmark_data,
        )
    elif task == "tiny_imagenet":
        from evaluators.evaluate_tiny_imagenet import TINY_IMAGENET_Evaluator

        evaluator = TINY_IMAGENET_Evaluator(
            parameters,
            task_name="tiny_imagenet",
            benchmark_data=benchmark_data,
        )
    elif task == "tiny_imagenet_custom":
        from evaluators.evaluate_tiny_imagenet_custom import TINY_IMAGENET_CUSTOM_Evaluator

        evaluator = TINY_IMAGENET_CUSTOM_Evaluator(
            parameters,
            task_name="tiny_imagenet_custom",
            benchmark_data=benchmark_data,
        )
    else:
        raise ValueError(
            f"Unknown task {task_name!r}. Supported tasks: fmnist, mnist, cifar10, "
            "cifar100, tiny_imagenet, tiny_imagenet_custom."
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
    has_array_test = all(hasattr(dataset, name) for name in ("x_test", "y_test"))
    has_streaming_test = hasattr(dataset, "test_data")
    if not has_array_test and not has_streaming_test:
        raise ValueError(
            "Benchmark dataset did not expose x_test/y_test or test_data"
        )
    return dataset


def prepare_evaluator_for_validation_assessment(evaluator, expected_test_size=None):
    """Load benchmark-layout data and assess Optuna trials on validation data."""

    dataset = _load_benchmark_dataset(evaluator, expected_test_size)
    has_array_validation = all(hasattr(dataset, name) for name in ("x_val", "y_val"))
    has_streaming_validation = hasattr(dataset, "validation_data")
    if not has_array_validation and not has_streaming_validation:
        raise ValueError(
            "Validation assessment requires x_val/y_val or validation_data"
        )

    if has_streaming_validation:
        dataset.fitness_data = dataset.validation_data
        dataset.fitness_steps = dataset.validation_steps
    else:
        dataset.x_fit = dataset.x_val
        dataset.y_fit = dataset.y_val
    evaluator.assessment_split = "validation"
    return evaluator


def prepare_evaluator_for_test_assessment(evaluator, expected_test_size=None):
    """Reload an evaluator's dataset for held-out test-set assessment."""

    dataset = _load_benchmark_dataset(evaluator, expected_test_size)
    if hasattr(dataset, "test_data"):
        dataset.fitness_data = dataset.test_data
        dataset.fitness_steps = dataset.test_steps
    else:
        dataset.x_fit = dataset.x_test
        dataset.y_fit = dataset.y_test
    evaluator.assessment_split = "test"
    return evaluator


def prepare_runner_parameters(parameters, output_dir, experiment_name):
    prepared = dict(parameters)
    prepared["LOGS_DIR"] = str(Path(output_dir) / "logs")
    prepared["EXPERIMENT_NAME"] = experiment_name
    return prepared


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

        optimizer = Adam()
        optimizer.name = "Adam"
        return optimizer
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


def materialize_optimizer(optimizer_spec, parameter_values=None):
    """Create a fresh optimizer from a serialized base and optional overrides."""

    from tensorflow.keras.optimizers import deserialize

    specification = {
        "class_name": optimizer_spec["class_name"],
        "config": dict(optimizer_spec["config"]),
    }
    if parameter_values:
        specification["config"].update(parameter_values)
    return deserialize(specification)


def load_task_parameters(task_name, use_test_data=False, config_dir=None):
    """Load default SGE parameters, optionally overlaid by benchmark TEST config."""

    from sge.parameters import default_params

    task = normalize_task_name(task_name)
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


def read_phenotype_argument(args):
    if getattr(args, "phenotype", None) is not None:
        return args.phenotype
    return Path(args.phenotype_file).read_text().strip()


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


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w") as output_file:
        json.dump(_json_safe(value), output_file, indent=2, sort_keys=True)
    temporary_path.replace(path)


def write_dataframe_csv(path, dataframe):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    dataframe.to_csv(temporary_path, index=False)
    temporary_path.replace(path)
