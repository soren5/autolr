import pytest


@pytest.fixture
def racing_parameters(tmp_path):
    from sge.parameters import manual_load_parameters

    parameters = {
        "CURRENT_GEN": 0,
        "FITNESS_FLOOR": 0,
        "RACING": True,
        "RACING_ALPHA": 0.05,
        "RACING_MIN_EVALS": 2,
        "RACING_MAX_EVALS": 5,
        "RACING_STAT_TEST": "mannwhitney",
        "DUMPS_DIR": str(tmp_path / "dumps"),
        "LOGS_DIR": str(tmp_path / "logs"),
        "DATA_DIR": str(tmp_path / "data"),
        "MODELS_DIR": str(tmp_path / "models"),
    }
    manual_load_parameters(parameters)
    return parameters


class FixedTaskEvaluator:
    def __init__(self, score):
        self.score = score

    def evaluate(self, phen):
        return self.score, {}


def test_multi_task_evaluator_emits_structured_cascade_record():
    from fitness_functions.fitness_functions import Optimizer_Evaluator_Multi_Task

    evaluator = object.__new__(Optimizer_Evaluator_Multi_Task)
    evaluator.fmnist_evaluator = FixedTaskEvaluator(0.91)
    evaluator.cifar10_evaluator = FixedTaskEvaluator(0.60)
    evaluator.cifar100_evaluator = None
    evaluator.tiny_imagenet_evaluator = None
    evaluator.tiny_imagenet_custom_evaluator = None
    params = {
        "FMNIST_THRESHOLD": 0.8,
        "CIFAR10_THRESHOLD": 0.7,
    }

    fitness, other_info = evaluator.evaluate("phenotype", params)

    assert fitness == pytest.approx(-1.51)
    assert other_info["multi_task"] == {
        "task_order": ["fmnist", "cifar10"],
        "scores": {"fmnist": 0.91, "cifar10": 0.60},
        "thresholds": {"fmnist": 0.8, "cifar10": 0.7},
        "passed": {"fmnist": True, "cifar10": False},
        "reached_depth": 2,
        "failed_task": "cifar10",
    }


def test_custom_tiny_imagenet_is_recorded_as_separate_multi_task():
    from fitness_functions.fitness_functions import Optimizer_Evaluator_Multi_Task

    evaluator = object.__new__(Optimizer_Evaluator_Multi_Task)
    evaluator.fmnist_evaluator = None
    evaluator.cifar10_evaluator = None
    evaluator.cifar100_evaluator = None
    evaluator.tiny_imagenet_evaluator = None
    evaluator.tiny_imagenet_custom_evaluator = FixedTaskEvaluator(0.42)
    params = {"TINY_IMAGENET_CUSTOM_THRESHOLD": 0.3}

    fitness, other_info = evaluator.evaluate("phenotype", params)

    assert fitness == pytest.approx(-0.42)
    assert "tiny_imagenet" not in other_info
    assert "tiny_imagenet_custom" in other_info
    assert other_info["source"] == "tiny_imagenet_custom_evaluation"
    assert other_info["multi_task"]["task_order"] == ["tiny_imagenet_custom"]
    assert other_info["multi_task"]["scores"] == {"tiny_imagenet_custom": 0.42}


def test_canonical_and_custom_tiny_imagenet_can_coexist_as_separate_tasks():
    from fitness_functions.fitness_functions import Optimizer_Evaluator_Multi_Task

    evaluator = object.__new__(Optimizer_Evaluator_Multi_Task)
    evaluator.fmnist_evaluator = None
    evaluator.cifar10_evaluator = None
    evaluator.cifar100_evaluator = None
    evaluator.tiny_imagenet_evaluator = FixedTaskEvaluator(0.41)
    evaluator.tiny_imagenet_custom_evaluator = FixedTaskEvaluator(0.43)
    params = {
        "TINY_IMAGENET_THRESHOLD": 0.3,
        "TINY_IMAGENET_CUSTOM_THRESHOLD": 0.3,
    }

    fitness, other_info = evaluator.evaluate("phenotype", params)

    assert fitness == pytest.approx(-0.84)
    assert other_info["multi_task"]["task_order"] == [
        "tiny_imagenet",
        "tiny_imagenet_custom",
    ]
    assert other_info["multi_task"]["scores"] == {
        "tiny_imagenet": 0.41,
        "tiny_imagenet_custom": 0.43,
    }


def test_multi_task_threshold_uses_current_task_not_cumulative_score():
    from fitness_functions.fitness_functions import Optimizer_Evaluator_Multi_Task

    evaluator = object.__new__(Optimizer_Evaluator_Multi_Task)
    evaluator.fmnist_evaluator = FixedTaskEvaluator(0.99)
    evaluator.cifar10_evaluator = FixedTaskEvaluator(0.10)
    evaluator.cifar100_evaluator = None
    evaluator.tiny_imagenet_evaluator = None
    evaluator.tiny_imagenet_custom_evaluator = None
    params = {
        "FMNIST_THRESHOLD": 0.8,
        "CIFAR10_THRESHOLD": 0.7,
    }

    fitness, other_info = evaluator.evaluate("phenotype", params)

    assert fitness == pytest.approx(-1.09)
    assert other_info["multi_task"]["passed"] == {
        "fmnist": True,
        "cifar10": False,
    }
    assert other_info["multi_task"]["failed_task"] == "cifar10"


def test_archive_multi_task_trial_sync_to_individual():
    from sge.engine import append_multi_task_trial_to_archive, sync_multi_task_trials_to_individual

    archive_entry = {"evaluations": [-1.6], "fitness": -1.6}
    multi_task = {
        "task_order": ["fmnist", "cifar10"],
        "scores": {"fmnist": 0.91, "cifar10": 0.60},
        "thresholds": {"fmnist": 0.8, "cifar10": 0.7},
        "passed": {"fmnist": True, "cifar10": False},
        "reached_depth": 2,
        "failed_task": "cifar10",
    }

    append_multi_task_trial_to_archive(archive_entry, multi_task)
    individual = {}
    sync_multi_task_trials_to_individual(archive_entry, individual)

    assert archive_entry["task_evaluations"]["fmnist"] == [0.91]
    assert archive_entry["task_passes"]["cifar10"] == [False]
    assert individual["task_trials"]["cifar10"] == [0.60]
    assert individual["reached_depth_trials"] == [2]
    assert individual["failed_task_trials"] == ["cifar10"]
    assert individual["multi_task_trials"][0]["failed_task"] == "cifar10"


def test_gated_cascade_eliminates_repeated_shallow_failures(racing_parameters):
    from sge.engine import candidate_is_clearly_worse_gated
    from sge.parameters import params

    params["RACING_ALPHA"] = 0.99
    archive = {
        "best": {
            "fitness": -2.0,
            "evaluations": [-2.0, -2.1],
            "multi_task_trials": [{}, {}],
            "reached_depths": [3, 3],
        },
        "candidate": {
            "fitness": -1.0,
            "evaluations": [-1.0, -1.1],
            "multi_task_trials": [{}, {}],
            "reached_depths": [1, 1],
        },
    }

    is_worse, p_value, metadata = candidate_is_clearly_worse_gated(archive, "best", "candidate")

    assert is_worse is True
    assert p_value is not None
    assert metadata["comparison_type"] == "reached_depth"


def test_gated_cascade_does_not_eliminate_after_one_gate_difference(racing_parameters):
    from sge.engine import candidate_is_clearly_worse_gated
    from sge.parameters import params

    params["RACING_ALPHA"] = 0.99
    archive = {
        "best": {
            "fitness": -2.0,
            "evaluations": [-2.0],
            "multi_task_trials": [{}],
            "reached_depths": [2],
        },
        "candidate": {
            "fitness": -1.0,
            "evaluations": [-1.0],
            "multi_task_trials": [{}],
            "reached_depths": [1],
        },
    }

    is_worse, p_value, metadata = candidate_is_clearly_worse_gated(archive, "best", "candidate")

    assert is_worse is False
    assert p_value is None
    assert metadata["comparison_type"] == "insufficient_gated_evidence"


def test_gated_cascade_eliminates_repeated_worse_task_scores(racing_parameters):
    from sge.engine import candidate_is_clearly_worse_gated
    from sge.parameters import params

    params["RACING_ALPHA"] = 0.99
    archive = {
        "best": {
            "fitness": -2.0,
            "evaluations": [-2.0, -2.1],
            "multi_task_trials": [{"task_order": ["fmnist"]}, {"task_order": ["fmnist"]}],
            "task_order": ["fmnist"],
            "reached_depths": [1, 1],
            "task_evaluations": {"fmnist": [0.90, 0.91]},
        },
        "candidate": {
            "fitness": -1.0,
            "evaluations": [-1.0, -1.1],
            "multi_task_trials": [{"task_order": ["fmnist"]}, {"task_order": ["fmnist"]}],
            "task_order": ["fmnist"],
            "reached_depths": [1, 1],
            "task_evaluations": {"fmnist": [0.40, 0.41]},
        },
    }

    is_worse, _, metadata = candidate_is_clearly_worse_gated(archive, "best", "candidate")

    assert is_worse is True
    assert metadata["comparison_type"] == "task_score"
    assert metadata["comparison_task"] == "fmnist"


def test_gated_cascade_missing_task_data_falls_back_to_scalar(racing_parameters):
    from sge.engine import candidate_is_clearly_worse_by_rule
    from sge.parameters import params

    params["RACING_DECISION_RULE"] = "gated_cascade"
    params["RACING_ALPHA"] = 0.99
    archive = {
        "best": {"fitness": -0.9, "evaluations": [-0.9, -0.91]},
        "candidate": {"fitness": -0.1, "evaluations": [-0.1, -0.11]},
    }

    is_worse, _, metadata = candidate_is_clearly_worse_by_rule(
        archive,
        "best",
        "candidate",
        archive["best"]["evaluations"],
    )

    assert is_worse is True
    assert metadata["comparison_type"] == "scalar_fallback"


def test_gated_cascade_keeps_candidate_when_gated_comparison_is_not_significant(racing_parameters):
    from sge.engine import candidate_is_clearly_worse_by_rule
    from sge.parameters import params

    params["RACING_DECISION_RULE"] = "gated_cascade"
    params["RACING_ALPHA"] = 0.05
    archive = {
        "best": {
            "fitness": -0.9,
            "evaluations": [-0.9, -0.91, -0.92],
            "multi_task_trials": [{}, {}, {}],
            "reached_depths": [2, 1, 2],
        },
        "candidate": {
            "fitness": -0.1,
            "evaluations": [-0.1, -0.11, -0.12],
            "multi_task_trials": [{}, {}, {}],
            "reached_depths": [1, 2, 1],
        },
    }

    is_worse, p_value, metadata = candidate_is_clearly_worse_by_rule(
        archive,
        "best",
        "candidate",
        archive["best"]["evaluations"],
    )

    assert is_worse is False
    assert p_value is None
    assert metadata["comparison_type"] == "gated_cascade_not_clearly_worse"
