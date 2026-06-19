import json

import pytest


BASIC_PHENOTYPE = (
    "alpha_func, beta_func, sigma_func, grad_func = "
    "lambda shape, alpha, grad: tf.constant(0.25, dtype=tf.float32), "
    "lambda shape, alpha, beta, grad: tf.constant(0.75, dtype=tf.float32), "
    "lambda shape, alpha, beta, sigma, grad: grad, "
    "lambda shape, alpha, beta, sigma, grad: "
    "tf.math.multiply(tf.constant(0.25, dtype=tf.float32), grad)"
)


class FixedFitnessEvaluator:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.phenotypes = []
        self.optimizers = []
        self.assessment_split = "fitness"

    def evaluate(self, phenotype):
        self.phenotypes.append(phenotype)
        score = next(self.scores)
        return score, {"raw_score": score}

    def evaluate_optimizer(self, optimizer):
        self.optimizers.append(optimizer)
        score = next(self.scores)
        return score, {"raw_score": score}


def test_fitness_runner_writes_phenotype_artifacts(tmp_path):
    from benchmarks.fitness_runner import run_fitness_subject

    evaluator = FixedFitnessEvaluator([0.3, 0.5])

    summary = run_fitness_subject(
        task_name="fmnist",
        parameters={},
        output_dir=tmp_path,
        repeats=2,
        phenotype=BASIC_PHENOTYPE,
        evaluator=evaluator,
    )

    assert summary["assessment_split"] == "fitness"
    assert summary["subject_type"] == "phenotype"
    assert summary["scores"] == [0.3, 0.5]
    assert summary["mean_score"] == pytest.approx(0.4)
    assert summary["median_score"] == pytest.approx(0.4)
    assert len(evaluator.phenotypes) == 2
    assert (tmp_path / "fitness_runs.csv").is_file()
    assert json.loads((tmp_path / "fitness_summary.json").read_text())["complete"]
    assert json.loads((tmp_path / "fitness_manifest.json").read_text())[
        "assessment_split"
    ] == "fitness"
    lines = (tmp_path / "fitness_runs.jsonl").read_text().splitlines()
    assert [json.loads(line)["assessment_split"] for line in lines] == [
        "fitness",
        "fitness",
    ]


def test_fitness_runner_uses_fresh_prebuilt_optimizer_per_repeat(monkeypatch, tmp_path):
    import benchmarks.fitness_runner as fitness_runner

    created = []

    def fake_create_prebuilt_optimizer(name):
        optimizer = {"name": name, "index": len(created)}
        created.append(optimizer)
        return optimizer

    monkeypatch.setattr(
        fitness_runner,
        "create_prebuilt_optimizer",
        fake_create_prebuilt_optimizer,
    )
    monkeypatch.setattr(
        fitness_runner,
        "serialize_optimizer",
        lambda optimizer: {"class_name": "FakeAdam", "config": dict(optimizer)},
    )
    evaluator = FixedFitnessEvaluator([0.4, 0.6])

    summary = fitness_runner.run_fitness_subject(
        task_name="fmnist",
        parameters={},
        output_dir=tmp_path,
        repeats=2,
        optimizer_name="adam",
        evaluator=evaluator,
    )

    assert summary["subject_type"] == "optimizer"
    assert summary["optimizer_name"] == "adam"
    assert len(evaluator.optimizers) == 2
    assert evaluator.optimizers[0] is not evaluator.optimizers[1]
    assert len(created) == 3


def test_fitness_runner_resumes_existing_runs(tmp_path):
    from benchmarks.fitness_runner import run_fitness_subject

    first = run_fitness_subject(
        task_name="fmnist",
        parameters={},
        output_dir=tmp_path,
        repeats=1,
        phenotype=BASIC_PHENOTYPE,
        evaluator=FixedFitnessEvaluator([0.2]),
    )
    second_evaluator = FixedFitnessEvaluator([0.8])
    second = run_fitness_subject(
        task_name="fmnist",
        parameters={},
        output_dir=tmp_path,
        repeats=2,
        phenotype=BASIC_PHENOTYPE,
        evaluator=second_evaluator,
    )

    assert first["scores"] == [0.2]
    assert second["scores"] == [0.2, 0.8]
    assert len(second_evaluator.phenotypes) == 1


def test_fitness_runner_rejects_manifest_mismatch(tmp_path):
    from benchmarks.fitness_runner import run_fitness_subject

    run_fitness_subject(
        task_name="fmnist",
        parameters={},
        output_dir=tmp_path,
        repeats=1,
        phenotype=BASIC_PHENOTYPE,
        evaluator=FixedFitnessEvaluator([0.2]),
    )

    with pytest.raises(ValueError, match="different inputs"):
        run_fitness_subject(
            task_name="mnist",
            parameters={},
            output_dir=tmp_path,
            repeats=1,
            phenotype=BASIC_PHENOTYPE,
            evaluator=FixedFitnessEvaluator([0.2]),
        )


def test_fitness_runner_creates_evolution_split_evaluator(monkeypatch, tmp_path):
    import benchmarks.fitness_runner as fitness_runner

    calls = []
    evaluator = FixedFitnessEvaluator([0.7])

    def fake_create_task_evaluator(
        task_name, parameters, use_validation_data=False, use_test_data=False
    ):
        calls.append((task_name, use_validation_data, use_test_data))
        return evaluator

    monkeypatch.setattr(
        fitness_runner,
        "create_task_evaluator",
        fake_create_task_evaluator,
    )

    fitness_runner.run_fitness_subject(
        task_name="fmnist",
        parameters={},
        output_dir=tmp_path,
        repeats=1,
        phenotype=BASIC_PHENOTYPE,
    )

    assert calls == [("fmnist", False, False)]


def test_fitness_main_loads_default_parameters_not_test_configs(monkeypatch, tmp_path):
    import benchmarks.fitness_runner as fitness_runner

    calls = []

    def fake_load_fitness_parameters(task_name, parameter_file=None):
        calls.append((task_name, parameter_file))
        return {"source": "sge-defaults"}

    monkeypatch.setattr(
        fitness_runner,
        "load_fitness_parameters",
        fake_load_fitness_parameters,
    )
    monkeypatch.setattr(
        fitness_runner,
        "run_fitness_subject",
        lambda **kwargs: {"runs": 1, "mean_score": 0.5},
    )

    fitness_runner.main(
        [
            "--phenotype",
            BASIC_PHENOTYPE,
            "--task",
            "fmnist",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert calls == [("fmnist", None)]


def test_fitness_runner_cli_accepts_prebuilt_adam():
    from benchmarks.fitness_runner import parse_args

    args = parse_args(
        [
            "--optimizer",
            "adam",
            "--task",
            "fmnist",
            "--output-dir",
            "results/fitness-adam",
        ]
    )

    assert args.optimizer == "adam"
    assert args.repeats == 1


def test_fitness_runner_loads_dataset_parameter_overlay(tmp_path, monkeypatch):
    import benchmarks.fitness_runner as fitness_runner

    parameter_dir = tmp_path / "parameters" / "dataset_parameters"
    parameter_dir.mkdir(parents=True)
    config_path = parameter_dir / "FMNIST_CONFIG.json"
    config_path.write_text(
        json.dumps(
            {
                "VALIDATION_SIZE": 10,
                "FITNESS_SIZE": 12,
                "BATCH_SIZE": 4,
                "EPOCHS": 2,
                "PATIENCE": 1,
                "MODEL": "small_mnist_model.h5",
                "NORMALIZE": True,
                "SUBTRACT_MEAN": False,
            }
        )
    )
    monkeypatch.setattr(fitness_runner, "DATASET_PARAMETERS_DIR", parameter_dir)

    parameters = fitness_runner.load_fitness_parameters(
        "fmnist",
        "FMNIST_CONFIG.json",
    )

    assert parameters["FITNESS_SIZE"] == 12
    assert parameters["BATCH_SIZE"] == 4
    assert parameters["MODEL"] == "small_mnist_model.h5"
    assert parameters["SUBTRACT_MEAN"] is False
    assert parameters["PARAMETERS"] == str(config_path.resolve())
    assert "TEST_SIZE" not in parameters


def test_fitness_runner_rejects_incomplete_dataset_parameter_overlay(
    tmp_path, monkeypatch
):
    import benchmarks.fitness_runner as fitness_runner

    parameter_dir = tmp_path / "parameters" / "dataset_parameters"
    parameter_dir.mkdir(parents=True)
    config_path = parameter_dir / "FMNIST_CONFIG.json"
    config_path.write_text(json.dumps({"FITNESS_SIZE": 12}))
    monkeypatch.setattr(fitness_runner, "DATASET_PARAMETERS_DIR", parameter_dir)

    with pytest.raises(ValueError, match="silently fall back to SGE defaults"):
        fitness_runner.load_fitness_parameters("fmnist", "FMNIST_CONFIG.json")


def test_fitness_runner_rejects_parameter_files_outside_dataset_directory(tmp_path):
    from benchmarks.fitness_runner import load_fitness_parameters

    outside_config = tmp_path / "outside.json"
    outside_config.write_text("{}")

    with pytest.raises(ValueError, match="parameters/dataset_parameters"):
        load_fitness_parameters("fmnist", outside_config)
