import copy

import pytest


def basic_phenotype(grad_expr):
    return (
        "alpha_func, beta_func, sigma_func, grad_func = "
        "lambda shape, alpha, grad: grad, "
        "lambda shape, alpha, beta, grad: grad, "
        "lambda shape, alpha, beta, sigma, grad: grad, "
        f"lambda shape, alpha, beta, sigma, grad: {grad_expr}"
    )


class ScriptedFitnessEvaluator:
    def __init__(self, sequences, start_index=0):
        self.sequences = sequences
        self.indices = {key: start_index for key in sequences}
        self.calls = []

    def evaluate(self, phen, parameters):
        from utils.smart_phenotype import smart_phenotype

        key = smart_phenotype(phen)
        index = self.indices[key]
        sequence = self.sequences[key]
        if index >= len(sequence):
            value = sequence[-1]
        else:
            value = sequence[index]
        self.indices[key] = index + 1
        self.calls.append(key)
        return value, {"source": "scripted-test-evaluator"}

    def init_net(self, parameters):
        pass

    def init_data(self, parameters):
        pass

    def init_evaluation(self, parameters):
        pass


def make_individual(phenotype, indiv_id):
    from utils.smart_phenotype import smart_phenotype

    return {
        "id": indiv_id,
        "phenotype": phenotype,
        "smart_phenotype": smart_phenotype(phenotype),
        "mapping_values": [],
        "tree_depth": 0,
        "fitness": None,
        "operation": "initialization",
    }


def make_archive(population, initial_fitness):
    archive = {}
    for indiv in population:
        key = indiv["smart_phenotype"]
        archive[key] = {
            "id": indiv["id"],
            "evaluations": [initial_fitness[key]],
            "fitness": initial_fitness[key],
        }
    return archive


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


@pytest.mark.unit
def test_racing_disabled_preserves_archive(racing_parameters):
    from sge.engine import update_best_fitness
    from sge.parameters import params

    params["RACING"] = False
    population = [make_individual(basic_phenotype("grad"), 1)]
    archive = make_archive(population, {population[0]["smart_phenotype"]: -0.5})
    original_archive = copy.deepcopy(archive)

    _, updated_archive = update_best_fitness(population, archive, evaluation_function=None)

    assert updated_archive == original_archive


@pytest.mark.unit
def test_racing_reevaluates_close_candidates_until_max(racing_parameters):
    from sge.engine import update_best_fitness

    best = make_individual(basic_phenotype("grad"), 1)
    close = make_individual(
        basic_phenotype("tf.math.multiply(tf.constant(9.99847452e-01, dtype=tf.float32), grad)"),
        2,
    )
    population = [best, close]
    archive = make_archive(
        population,
        {
            best["smart_phenotype"]: -0.50,
            close["smart_phenotype"]: -0.50,
        },
    )
    evaluator = ScriptedFitnessEvaluator(
        {
            best["smart_phenotype"]: [-0.50, -0.50, -0.50, -0.50],
            close["smart_phenotype"]: [-0.50, -0.50, -0.50, -0.50],
        },
        start_index=1,
    )

    _, updated_archive = update_best_fitness(population, archive, evaluator)

    assert len(updated_archive[best["smart_phenotype"]]["evaluations"]) == 5
    assert len(updated_archive[close["smart_phenotype"]]["evaluations"]) == 5


@pytest.mark.unit
def test_racing_eliminates_clearly_worse_candidate_before_max(racing_parameters):
    from sge.engine import update_best_fitness
    from sge.parameters import params

    params["RACING_ALPHA"] = 0.99
    best = make_individual(basic_phenotype("grad"), 1)
    worse = make_individual(
        basic_phenotype("tf.math.multiply(tf.constant(5.55606489e-05, dtype=tf.float32), grad)"),
        2,
    )
    population = [best, worse]
    archive = make_archive(
        population,
        {
            best["smart_phenotype"]: -0.90,
            worse["smart_phenotype"]: -0.10,
        },
    )
    evaluator = ScriptedFitnessEvaluator(
        {
            best["smart_phenotype"]: [-0.90, -0.91, -0.92, -0.93],
            worse["smart_phenotype"]: [-0.10, -0.11, -0.12, -0.13],
        },
        start_index=1,
    )

    _, updated_archive = update_best_fitness(population, archive, evaluator)

    assert len(updated_archive[best["smart_phenotype"]]["evaluations"]) < params["RACING_MAX_EVALS"]
    assert len(updated_archive[worse["smart_phenotype"]]["evaluations"]) < params["RACING_MAX_EVALS"]


@pytest.mark.unit
def test_racing_does_not_reevaluate_invalid_no_grad_candidate(racing_parameters):
    from sge.engine import update_best_fitness

    valid = make_individual(basic_phenotype("grad"), 1)
    invalid = make_individual(basic_phenotype("tf.constant(1.0, dtype=tf.float32)"), 2)
    population = [valid, invalid]
    archive = make_archive(
        population,
        {
            valid["smart_phenotype"]: -0.50,
            invalid["smart_phenotype"]: 0,
        },
    )
    evaluator = ScriptedFitnessEvaluator(
        {
            valid["smart_phenotype"]: [-0.50, -0.50],
            invalid["smart_phenotype"]: [0, 0],
        },
        start_index=1,
    )

    _, updated_archive = update_best_fitness(population, archive, evaluator)

    assert len(updated_archive[valid["smart_phenotype"]]["evaluations"]) == 1
    assert len(updated_archive[invalid["smart_phenotype"]]["evaluations"]) == 1
    assert invalid["smart_phenotype"] not in evaluator.calls
