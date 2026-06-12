import json
from types import SimpleNamespace

import pytest


BASIC_PHENOTYPE = (
    "alpha_func, beta_func, sigma_func, grad_func = "
    "lambda shape, alpha, grad: tf.constant(0.25, dtype=tf.float32), "
    "lambda shape, alpha, beta, grad: tf.constant(0.75, dtype=tf.float32), "
    "lambda shape, alpha, beta, sigma, grad: grad, "
    "lambda shape, alpha, beta, sigma, grad: "
    "tf.math.multiply(tf.constant(0.25, dtype=tf.float32), grad)"
)


class FixedBenchmarkDataset:
    def __init__(self):
        self.x_fit = "fitness-data"
        self.y_fit = "fitness-labels"
        self.x_val = ["validation-example-1", "validation-example-2"]
        self.y_val = ["validation-label-1", "validation-label-2"]
        self.x_test = None
        self.y_test = None
        self.test_size = 2
        self.benchmark_loads = 0

    def load_data_for_benchmark(self):
        from dataset_loaders.dataset_utils import validate_benchmark_test_size

        self.benchmark_loads += 1
        self.x_test = ["test-example-1", "test-example-2"]
        self.y_test = ["test-label-1", "test-label-2"]
        validate_benchmark_test_size(
            self.x_test,
            self.y_test,
            expected_test_size=getattr(self, "test_size", None),
        )


class FixedEvaluator:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.phenotypes = []
        self.dataset = FixedBenchmarkDataset()
        self.assessment_split = "fitness"

    def evaluate(self, phenotype):
        self.phenotypes.append(phenotype)
        score = next(self.scores)
        return score, {"test_score": score}


class FixedOptimizerEvaluator:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.optimizers = []
        self.dataset = FixedBenchmarkDataset()
        self.assessment_split = "fitness"

    def evaluate_optimizer(self, optimizer):
        self.optimizers.append(optimizer)
        score = next(self.scores)
        return score, {"test_score": score}


class NamelessOptimizer:
    pass


def test_abstract_constants_only_tunes_active_values():
    from utils.smart_phenotype import abstract_active_constants

    template, parameters = abstract_active_constants(BASIC_PHENOTYPE)

    assert parameters == {"CONST_0": 0.25}
    assert template.count("CONST_0") == 1
    assert "lambda shape, alpha, grad: tf.constant(0.25, dtype=tf.float32)" in template
    assert "tf.constant(0.75, dtype=tf.float32)" in template


def test_materialize_phenotype_replaces_active_constant():
    from utils.smart_phenotype import abstract_active_constants, materialize_constants

    template, _ = abstract_active_constants(BASIC_PHENOTYPE)
    phenotype = materialize_constants(template, {"CONST_0": 0.125})

    assert "CONST_" not in phenotype
    assert phenotype.count("tf.constant(0.125, dtype=tf.float32)") == 1


def test_benchmark_best_phenotype_resumes_and_writes_summary(tmp_path):
    from benchmarks.new_benchmark import benchmark_best_phenotype
    from utils.smart_phenotype import abstract_active_constants, materialize_constants

    template, _ = abstract_active_constants(BASIC_PHENOTYPE)
    study = SimpleNamespace(
        study_name="test-study",
        best_value=0.8,
        best_params={"CONST_0": 0.1},
        user_attrs={"phenotype_template": template},
    )
    (tmp_path / "benchmark_manifest.json").write_text(
        json.dumps(
            {
                "task_name": "fmnist",
                "assessment_split": "test",
                "study_name": "test-study",
                "best_parameters": {"CONST_0": 0.1},
                "phenotype": materialize_constants(template, {"CONST_0": 0.1}),
            },
            sort_keys=True,
        )
    )
    result_path = tmp_path / "benchmark_runs.jsonl"
    result_path.write_text(json.dumps({"run": 1, "score": 0.5, "details": {}}) + "\n")
    evaluator = FixedEvaluator([0.7, 0.9])

    summary = benchmark_best_phenotype(
        study,
        "fmnist",
        {},
        tmp_path,
        repeats=3,
        evaluator=evaluator,
    )

    assert summary["scores"] == [0.5, 0.7, 0.9]
    assert summary["mean_score"] == pytest.approx(0.7)
    assert summary["assessment_split"] == "test"
    assert len(evaluator.phenotypes) == 2
    assert evaluator.dataset.x_fit == ["test-example-1", "test-example-2"]
    assert evaluator.dataset.y_fit == ["test-label-1", "test-label-2"]
    assert evaluator.dataset.benchmark_loads == 1
    assert (tmp_path / "benchmark_summary.json").is_file()
    assert (tmp_path / "benchmark_runs.csv").is_file()


def test_benchmark_rejects_results_from_a_different_best_phenotype(tmp_path):
    from benchmarks.new_benchmark import benchmark_best_phenotype
    from utils.smart_phenotype import abstract_active_constants

    template, _ = abstract_active_constants(BASIC_PHENOTYPE)
    study = SimpleNamespace(
        study_name="test-study",
        best_value=0.8,
        best_params={"CONST_0": 0.1},
        user_attrs={"phenotype_template": template},
    )
    (tmp_path / "benchmark_manifest.json").write_text(
        json.dumps({"phenotype": "a different optimizer"})
    )

    with pytest.raises(ValueError, match="different best phenotype"):
        benchmark_best_phenotype(
            study,
            "fmnist",
            {},
            tmp_path,
            repeats=1,
            evaluator=FixedEvaluator([0.5]),
        )


def test_tuning_resumes_sqlite_study_to_requested_total_trials(tmp_path):
    from benchmarks.new_benchmark import tune_phenotype

    first_evaluator = FixedEvaluator([0.4, 0.6])
    first_study = tune_phenotype(
        BASIC_PHENOTYPE,
        "fmnist",
        {},
        n_trials=2,
        output_dir=tmp_path,
        study_name="resume-test",
        evaluator=first_evaluator,
        seed=1,
    )
    first_trial_count = len(first_study.trials)
    first_suggestions = [trial.params["CONST_0"] for trial in first_study.trials]
    assert first_study.trials[0].params == {"CONST_0": 0.25}
    assert first_study.trials[0].user_attrs["source"] == "default_parameter_probe"
    second_evaluator = FixedEvaluator([0.8])
    resumed_study = tune_phenotype(
        BASIC_PHENOTYPE,
        "fmnist",
        {},
        n_trials=3,
        output_dir=tmp_path,
        study_name="resume-test",
        evaluator=second_evaluator,
        seed=1,
    )

    assert first_trial_count == 2
    assert len(resumed_study.trials) == 3
    assert resumed_study.best_value == pytest.approx(0.8)
    assert resumed_study.trials[-1].params["CONST_0"] not in first_suggestions
    assert sum(
        trial.user_attrs.get("source") == "default_parameter_probe"
        for trial in resumed_study.trials
    ) == 1
    assert len(second_evaluator.phenotypes) == 1
    assert (tmp_path / "optuna_study.sqlite3").is_file()
    assert (tmp_path / "tuning_trials.csv").is_file()
    assert (tmp_path / "best_tuned_phenotype.json").is_file()


def test_materialize_optimizer_creates_fresh_adam_with_overrides():
    from tensorflow.keras.optimizers import Adam

    from benchmarks.new_benchmark import materialize_optimizer, serialize_optimizer

    specification = serialize_optimizer(Adam())
    first = materialize_optimizer(specification, {"learning_rate": 0.01})
    second = materialize_optimizer(specification, {"learning_rate": 0.02})

    assert first is not second
    assert float(first.learning_rate.numpy()) == pytest.approx(0.01)
    assert float(second.learning_rate.numpy()) == pytest.approx(0.02)


def test_optimizer_evaluator_logging_accepts_optimizer_without_name(tmp_path):
    from evaluators.evaluator_utils import Evaluator

    evaluator = object.__new__(Evaluator)
    evaluator.task_name = "test"
    evaluator.run = 0
    evaluator.log_path = str(tmp_path)
    evaluator.fake_fitness = True
    evaluator.train_model = lambda *args, **kwargs: (0.5, {})
    (tmp_path / "logs").mkdir()
    (tmp_path / "csv").mkdir()

    score, _ = evaluator.evaluate_optimizer(NamelessOptimizer())

    assert score == pytest.approx(0.5)
    assert "NamelessOptimizer" in (
        tmp_path / "logs" / "run_0_test_log.log"
    ).read_text()


def test_tune_prebuilt_adam_uses_fresh_optimizer_per_trial(tmp_path):
    from tensorflow.keras.optimizers import Adam

    from benchmarks.new_benchmark import tune_optimizer

    evaluator = FixedOptimizerEvaluator([0.4, 0.8])
    study = tune_optimizer(
        optimizer=Adam(),
        task_name="fmnist",
        parameters={},
        n_trials=2,
        output_dir=tmp_path,
        search_space={
            "learning_rate": {
                "type": "float",
                "low": 1e-4,
                "high": 1e-2,
                "log": True,
            }
        },
        study_name="adam-test",
        evaluator=evaluator,
        seed=2,
    )

    assert len(evaluator.optimizers) == 2
    assert evaluator.optimizers[0] is not evaluator.optimizers[1]
    assert all(optimizer.__class__.__name__ == "Adam" for optimizer in evaluator.optimizers)
    assert float(evaluator.optimizers[0].learning_rate.numpy()) == pytest.approx(0.001)
    assert study.trials[0].user_attrs["source"] == "default_parameter_probe"
    assert study.best_value == pytest.approx(0.8)
    assert (tmp_path / "best_tuned_optimizer.json").is_file()


def test_default_probe_must_be_inside_search_space(tmp_path):
    from tensorflow.keras.optimizers import Adam

    from benchmarks.new_benchmark import tune_optimizer

    with pytest.raises(ValueError, match="outside its search space"):
        tune_optimizer(
            optimizer=Adam(learning_rate=0.001),
            task_name="fmnist",
            parameters={},
            n_trials=1,
            output_dir=tmp_path,
            search_space={
                "learning_rate": {
                    "type": "float",
                    "low": 0.01,
                    "high": 0.1,
                }
            },
            study_name="invalid-default-probe",
            evaluator=FixedOptimizerEvaluator([0.5]),
        )


def test_benchmark_best_optimizer_resumes_with_fresh_instances(tmp_path):
    from tensorflow.keras.optimizers import Adam

    from benchmarks.new_benchmark import (
        benchmark_best_optimizer,
        serialize_optimizer,
    )

    optimizer_spec = serialize_optimizer(Adam())
    study = SimpleNamespace(
        study_name="adam-test",
        best_value=0.8,
        best_params={"learning_rate": 0.01},
        user_attrs={"optimizer_spec": optimizer_spec},
    )
    evaluator = FixedOptimizerEvaluator([0.6, 0.7])

    summary = benchmark_best_optimizer(
        study,
        "fmnist",
        {},
        tmp_path,
        repeats=2,
        evaluator=evaluator,
    )

    assert summary["scores"] == [0.6, 0.7]
    assert summary["assessment_split"] == "test"
    assert len(evaluator.optimizers) == 2
    assert evaluator.optimizers[0] is not evaluator.optimizers[1]
    assert all(
        float(optimizer.learning_rate.numpy()) == pytest.approx(0.01)
        for optimizer in evaluator.optimizers
    )
    assert summary["optimizer"]["class_name"] == "Adam"
    assert evaluator.dataset.x_fit == ["test-example-1", "test-example-2"]
    assert evaluator.dataset.y_fit == ["test-label-1", "test-label-2"]
    assert evaluator.dataset.benchmark_loads == 1


def test_prepare_evaluator_for_test_assessment_replaces_fitness_split():
    from benchmarks.new_benchmark import prepare_evaluator_for_test_assessment

    evaluator = FixedEvaluator([0.5])

    prepared = prepare_evaluator_for_test_assessment(evaluator)

    assert prepared is evaluator
    assert evaluator.assessment_split == "test"
    assert evaluator.dataset.x_fit == evaluator.dataset.x_test
    assert evaluator.dataset.y_fit == evaluator.dataset.y_test


def test_prepare_evaluator_for_validation_assessment_replaces_fitness_split():
    from benchmarks.new_benchmark import (
        evaluate_phenotype,
        prepare_evaluator_for_validation_assessment,
    )

    evaluator = FixedEvaluator([0.5])

    prepared = prepare_evaluator_for_validation_assessment(evaluator)

    assert prepared is evaluator
    assert evaluator.assessment_split == "validation"
    assert evaluator.dataset.x_fit == evaluator.dataset.x_val
    assert evaluator.dataset.y_fit == evaluator.dataset.y_val
    assert evaluator.dataset.benchmark_loads == 1

    score, details = evaluate_phenotype(evaluator, BASIC_PHENOTYPE, "fmnist", {})

    assert score == pytest.approx(0.5)
    assert details["assessment_split"] == "validation"
    assert details["validation_score"] == pytest.approx(0.5)


def test_task_evaluator_rejects_simultaneous_validation_and_test_assessment():
    from benchmarks.new_benchmark import create_task_evaluator

    with pytest.raises(ValueError, match="cannot assess validation and test"):
        create_task_evaluator(
            "fmnist",
            {},
            use_validation_data=True,
            use_test_data=True,
        )


def test_test_size_parameter_matches_loaded_benchmark_dataset():
    from benchmarks.new_benchmark import prepare_evaluator_for_test_assessment

    evaluator = FixedEvaluator([0.5])

    prepare_evaluator_for_test_assessment(evaluator, expected_test_size=2)

    assert len(evaluator.dataset.x_test) == 2
    assert len(evaluator.dataset.y_test) == 2


def test_test_size_parameter_mismatch_is_rejected():
    from benchmarks.new_benchmark import prepare_evaluator_for_test_assessment

    evaluator = FixedEvaluator([0.5])

    with pytest.raises(ValueError, match="TEST_SIZE=3"):
        prepare_evaluator_for_test_assessment(evaluator, expected_test_size=3)


def test_evolutionary_parameter_defaults_exclude_benchmark_test_size():
    from sge.parameters import default_params

    assert "TEST_SIZE" not in default_params


def test_cli_accepts_prebuilt_adam():
    from benchmarks.new_benchmark import parse_args

    arguments = parse_args(
        [
            "--optimizer",
            "adam",
            "--task",
            "fmnist",
            "--output-dir",
            "results/adam",
        ]
    )

    assert arguments.optimizer == "adam"
    assert arguments.phenotype is None
    assert arguments.phenotype_file is None
    assert not hasattr(arguments, "parameters")


def test_load_task_parameters_without_test_data_uses_sge_defaults():
    from benchmarks.new_benchmark import load_task_parameters
    from sge.parameters import default_params

    parameters = load_task_parameters("fmnist")

    assert parameters == default_params
    assert "TEST_SIZE" not in parameters


def test_load_task_parameters_uses_test_configuration_without_base_overlay():
    from benchmarks.new_benchmark import load_task_parameters
    from sge.parameters import default_params

    parameters = load_task_parameters("tinyimagenet", use_test_data=True)

    assert parameters["VALIDATION_SIZE"] == 7000
    assert parameters["FITNESS_SIZE"] == default_params["FITNESS_SIZE"]
    assert parameters["TEST_SIZE"] == 10000
    assert parameters["BATCH_SIZE"] == 64


def test_optuna_parameter_loading_uses_test_configuration_overlay():
    from benchmarks.new_benchmark import load_task_parameters

    parameters = load_task_parameters("fmnist", use_test_data=True)

    assert parameters["VALIDATION_SIZE"] == 7000
    assert parameters["TEST_SIZE"] == 10000


def test_main_uses_test_configuration_parameters_for_optuna(monkeypatch, tmp_path):
    import benchmarks.new_benchmark as benchmark

    calls = []
    parameters = {"TEST_SIZE": 2, "source": "test-config"}
    study = SimpleNamespace(best_value=0.5, best_params={})

    def fake_load_task_parameters(task_name, use_test_data=False, config_dir=None):
        calls.append((task_name, use_test_data))
        return parameters

    def fake_tune_optimizer(**kwargs):
        assert kwargs["parameters"] is parameters
        return study

    monkeypatch.setattr(benchmark, "load_task_parameters", fake_load_task_parameters)
    monkeypatch.setattr(benchmark, "create_prebuilt_optimizer", lambda name: object())
    monkeypatch.setattr(benchmark, "tune_optimizer", fake_tune_optimizer)

    benchmark.main(
        [
            "--optimizer",
            "adam",
            "--task",
            "fmnist",
            "--output-dir",
            str(tmp_path),
            "--tune-only",
        ]
    )

    assert calls == [("fmnist", True)]


def test_load_task_parameters_supports_mnist_configurations():
    from benchmarks.new_benchmark import load_task_parameters
    from sge.parameters import default_params

    tuning_parameters = load_task_parameters("mnist")
    benchmark_parameters = load_task_parameters("mnist", use_test_data=True)

    assert tuning_parameters == default_params
    assert "TEST_SIZE" not in tuning_parameters
    assert benchmark_parameters["FITNESS_SIZE"] == default_params["FITNESS_SIZE"]
    assert benchmark_parameters["TEST_SIZE"] == 10000


def test_load_task_parameters_rejects_task_without_configuration():
    from benchmarks.new_benchmark import load_task_parameters

    with pytest.raises(ValueError, match="No benchmark dataset configuration"):
        load_task_parameters("imagenet")


def test_load_task_parameters_reports_missing_test_configuration(tmp_path):
    from benchmarks.new_benchmark import load_task_parameters

    (tmp_path / "FMNIST_CONFIG.json").write_text("{}")

    with pytest.raises(FileNotFoundError, match="FMNIST_CONFIG_TEST.json"):
        load_task_parameters("fmnist", use_test_data=True, config_dir=tmp_path)


def test_load_task_parameters_test_mode_does_not_require_base_configuration(tmp_path):
    from benchmarks.new_benchmark import load_task_parameters

    (tmp_path / "FMNIST_CONFIG_TEST.json").write_text(
        json.dumps({"TEST_SIZE": 2, "VALIDATION_SIZE": 3})
    )

    parameters = load_task_parameters("fmnist", use_test_data=True, config_dir=tmp_path)

    assert parameters["TEST_SIZE"] == 2
    assert parameters["VALIDATION_SIZE"] == 3


def test_evaluator_fitness_size_is_optional_only_for_benchmark_data():
    from evaluators.evaluator_utils import Evaluator

    evaluator = object.__new__(Evaluator)
    evaluator.task_name = "test"
    parameters = {
        "VALIDATION_SIZE": 3,
        "BATCH_SIZE": 2,
        "EPOCHS": 1,
        "PATIENCE": 1,
        "MODEL": "unused.keras",
        "NORMALIZE": True,
        "SUBTRACT_MEAN": False,
    }

    parsed = evaluator.find_params(
        None, parameters, fitness_size_required=False
    )

    assert parsed[1] is None
    with pytest.raises(KeyError, match="FITNESS_SIZE"):
        evaluator.find_params(None, parameters)
