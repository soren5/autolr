import pytest

from tests.helpers import DeterministicFitnessEvaluator


@pytest.fixture(autouse=True)
def reset_sge_state():
    from sge.grammar import grammar
    from sge.parameters import reset_parameters

    reset_parameters()
    grammar._reset_grammar()
    yield
    reset_parameters()
    grammar._reset_grammar()


@pytest.fixture
def deterministic_evaluator():
    return DeterministicFitnessEvaluator()


@pytest.fixture
def tiny_engine_parameters(tmp_path):
    artifact_root = tmp_path / "artifacts"
    parameters = {
        "SELECTION_TYPE": "tournament",
        "POPSIZE": 6,
        "GENERATIONS": 3,
        "ELITISM": 1,
        "SEED": 7,
        "PROB_CROSSOVER": 0.8,
        "PROB_MUTATION": 0.2,
        "TSIZE": 2,
        "GRAMMAR": "grammars/basic_optimizer.txt",
        "EXPERIMENT_NAME": "engine_fake_fitness_smoke",
        "RUN": 1,
        "INCLUDE_GENOTYPE": True,
        "SAVE_STEP": 1,
        "VERBOSE": False,
        "MIN_TREE_DEPTH": 2,
        "MAX_TREE_DEPTH": 5,
        "FITNESS_FLOOR": 0,
        "FAKE_FITNESS": True,
        "LOAD_ARCHIVE": True,
        "DUMPS_DIR": str(artifact_root / "dumps"),
        "LOGS_DIR": str(artifact_root / "logs"),
        "DATA_DIR": str(artifact_root / "data"),
        "MODELS_DIR": str(artifact_root / "models"),
    }
    return parameters
