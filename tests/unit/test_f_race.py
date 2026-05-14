import copy
import csv
import json

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


def prepare_native_logger():
    import sge.logger as logger

    logger.prepare_dumps()
    return logger


def read_jsonl(path):
    with path.open("r") as report_file:
        return [json.loads(line) for line in report_file]


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
            worse["smart_phenotype"]: -0.25,
        },
    )
    evaluator = ScriptedFitnessEvaluator(
        {
            best["smart_phenotype"]: [-0.90, -0.91, -0.92, -0.93],
            worse["smart_phenotype"]: [-0.25, -0.26, -0.27, -0.28],
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


@pytest.mark.unit
def test_racing_does_not_reevaluate_candidate_with_score_in_no_reevaluation_band(racing_parameters):
    from sge.engine import update_best_fitness

    strong = make_individual(basic_phenotype("grad"), 1)
    weak = make_individual(
        basic_phenotype("tf.math.multiply(tf.constant(9.99847452e-01, dtype=tf.float32), grad)"),
        2,
    )
    population = [strong, weak]
    archive = make_archive(
        population,
        {
            strong["smart_phenotype"]: -0.50,
            weak["smart_phenotype"]: -0.10,
        },
    )
    evaluator = ScriptedFitnessEvaluator(
        {
            strong["smart_phenotype"]: [-0.50, -0.50, -0.50, -0.50],
            weak["smart_phenotype"]: [-0.10, -0.10, -0.10, -0.10],
        },
        start_index=1,
    )

    _, updated_archive = update_best_fitness(population, archive, evaluator)

    assert len(updated_archive[strong["smart_phenotype"]]["evaluations"]) == 5
    assert len(updated_archive[weak["smart_phenotype"]]["evaluations"]) == 1
    assert weak["smart_phenotype"] not in evaluator.calls


@pytest.mark.unit
def test_racing_stops_reevaluating_candidate_after_new_score_enters_no_reevaluation_band(racing_parameters):
    from sge.engine import update_best_fitness

    steady = make_individual(basic_phenotype("grad"), 1)
    drops_into_band = make_individual(
        basic_phenotype("tf.math.multiply(tf.constant(9.99847452e-01, dtype=tf.float32), grad)"),
        2,
    )
    population = [steady, drops_into_band]
    archive = make_archive(
        population,
        {
            steady["smart_phenotype"]: -0.50,
            drops_into_band["smart_phenotype"]: -0.50,
        },
    )
    evaluator = ScriptedFitnessEvaluator(
        {
            steady["smart_phenotype"]: [-0.50, -0.50, -0.50, -0.50],
            drops_into_band["smart_phenotype"]: [-0.10, -0.10, -0.10, -0.10],
        },
        start_index=0,
    )

    _, updated_archive = update_best_fitness(population, archive, evaluator)

    assert len(updated_archive[steady["smart_phenotype"]]["evaluations"]) == 5
    assert updated_archive[drops_into_band["smart_phenotype"]]["evaluations"] == [-0.50, -0.10]
    assert evaluator.calls.count(drops_into_band["smart_phenotype"]) == 1


@pytest.mark.unit
def test_pre_race_snapshot_records_duplicates_invalids_and_ranks(racing_parameters):
    from sge.engine import build_pre_race_snapshot

    best = make_individual(basic_phenotype("grad"), 1)
    duplicate = make_individual(best["phenotype"], 2)
    worse = make_individual(
        basic_phenotype("tf.math.multiply(tf.constant(9.99847452e-01, dtype=tf.float32), grad)"),
        3,
    )
    invalid = make_individual(basic_phenotype("tf.constant(1.0, dtype=tf.float32)"), 4)
    population = [best, duplicate, worse, invalid]
    archive = make_archive(
        [best, worse, invalid],
        {
            best["smart_phenotype"]: -0.50,
            worse["smart_phenotype"]: -0.25,
            invalid["smart_phenotype"]: 0,
        },
    )
    archive[best["smart_phenotype"]]["evaluations"] = [-0.40, -0.60]
    archive[best["smart_phenotype"]]["fitness"] = -0.50

    snapshot = build_pre_race_snapshot(population, archive)

    assert snapshot[best["smart_phenotype"]]["population_ids"] == [1, 2]
    assert snapshot[best["smart_phenotype"]]["first_fitness"] == -0.40
    assert snapshot[best["smart_phenotype"]]["pre_race_fitness"] == -0.50
    assert snapshot[best["smart_phenotype"]]["n_evals_before"] == 2
    assert snapshot[best["smart_phenotype"]]["initial_rank"] == 1
    assert snapshot[worse["smart_phenotype"]]["initial_rank"] == 2
    assert snapshot[invalid["smart_phenotype"]]["valid"] is False
    assert snapshot[invalid["smart_phenotype"]]["initial_rank"] is None


@pytest.mark.unit
def test_f_race_logging_writes_event_trace_and_summary(racing_parameters):
    from sge.engine import build_pre_race_snapshot, update_best_fitness
    from sge.parameters import params
    from tests.helpers import run_dump_dir

    params["RACING_MAX_EVALS"] = 3
    best = make_individual(basic_phenotype("grad"), 1)
    close = make_individual(
        basic_phenotype("tf.math.multiply(tf.constant(9.99847452e-01, dtype=tf.float32), grad)"),
        2,
    )
    invalid = make_individual(basic_phenotype("tf.constant(1.0, dtype=tf.float32)"), 3)
    population = [best, close, invalid]
    archive = make_archive(
        population,
        {
            best["smart_phenotype"]: -0.50,
            close["smart_phenotype"]: -0.50,
            invalid["smart_phenotype"]: 0,
        },
    )
    snapshot = build_pre_race_snapshot(population, archive)
    evaluator = ScriptedFitnessEvaluator(
        {
            best["smart_phenotype"]: [-0.50, -0.50, -0.50],
            close["smart_phenotype"]: [-0.50, -0.50, -0.50],
        },
        start_index=1,
    )
    logger = prepare_native_logger()

    update_best_fitness(population, archive, evaluator, logger, 0, snapshot)

    dump_dir = run_dump_dir(params)
    events = read_jsonl(dump_dir / "_race_f_race_report.jsonl")
    event_names = [event["event"] for event in events]
    assert "race_start" in event_names
    assert "candidate_snapshot" in event_names
    assert "reevaluation" in event_names
    assert "race_stop" in event_names
    invalid_snapshots = [
        event for event in events
        if event["event"] == "candidate_snapshot" and event["key"] == invalid["smart_phenotype"]
    ]
    assert invalid_snapshots[0]["valid"] is False

    with (dump_dir / "_race_f_race_summary.csv").open("r") as summary_file:
        rows = list(csv.DictReader(summary_file))
    assert rows[0]["generation"] == "0"
    assert rows[0]["eligible_count"] == "2"
    assert rows[0]["invalid_count"] == "1"
    assert int(rows[0]["extra_evaluations"]) > 0


@pytest.mark.unit
def test_f_race_logging_records_elimination_p_value(racing_parameters):
    from sge.engine import build_pre_race_snapshot, update_best_fitness
    from sge.parameters import params
    from tests.helpers import run_dump_dir

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
            worse["smart_phenotype"]: -0.25,
        },
    )
    snapshot = build_pre_race_snapshot(population, archive)
    evaluator = ScriptedFitnessEvaluator(
        {
            best["smart_phenotype"]: [-0.90, -0.91, -0.92],
            worse["smart_phenotype"]: [-0.25, -0.26, -0.27],
        },
        start_index=1,
    )
    logger = prepare_native_logger()

    update_best_fitness(population, archive, evaluator, logger, 0, snapshot)

    events = read_jsonl(run_dump_dir(params) / "_race_f_race_report.jsonl")
    eliminations = [event for event in events if event["event"] == "elimination"]
    assert eliminations
    assert eliminations[0]["key"] == worse["smart_phenotype"]
    assert eliminations[0]["p_value"] is not None


@pytest.mark.unit
def test_tournament_audit_reports_counterfactual_parent_without_changing_actual(racing_parameters, monkeypatch):
    from sge.engine import build_pre_race_snapshot, make_selection_audit_summary, tournament_selection
    from sge.parameters import params
    from tests.helpers import run_dump_dir

    one_shot_best = make_individual(basic_phenotype("grad"), 1)
    race_best = make_individual(
        basic_phenotype("tf.math.multiply(tf.constant(9.99847452e-01, dtype=tf.float32), grad)"),
        2,
    )
    population = [one_shot_best, race_best]
    archive = make_archive(
        population,
        {
            one_shot_best["smart_phenotype"]: -0.90,
            race_best["smart_phenotype"]: -0.10,
        },
    )
    snapshot = build_pre_race_snapshot(population, archive)
    one_shot_best["fitness"] = -0.20
    race_best["fitness"] = -0.95
    logger = prepare_native_logger()
    summary = make_selection_audit_summary()
    monkeypatch.setattr("sge.engine.random.sample", lambda sampled_population, tsize: population)

    selected = tournament_selection(population, logger, 0, snapshot, 0, summary)

    assert selected["id"] == race_best["id"]
    assert summary["tournament_events"] == 1
    assert summary["tournament_changed"] == 1
    events = read_jsonl(run_dump_dir(params) / "_race_selection_audit_report.jsonl")
    assert events[0]["event"] == "tournament_audit"
    assert events[0]["outcome_changed"] is True
    assert events[0]["actual_parent_id"] == race_best["id"]
    assert events[0]["counterfactual_parent_id"] == one_shot_best["id"]


@pytest.mark.unit
def test_elitism_audit_reports_counterfactual_unique_elite_set(racing_parameters):
    from sge.engine import build_pre_race_snapshot, make_selection_audit_summary, reproduce_via_elitism
    from sge.parameters import params
    from tests.helpers import run_dump_dir

    params["ELITISM"] = 1
    one_shot_best = make_individual(basic_phenotype("grad"), 1)
    race_best = make_individual(
        basic_phenotype("tf.math.multiply(tf.constant(9.99847452e-01, dtype=tf.float32), grad)"),
        2,
    )
    population = [race_best, one_shot_best]
    archive = make_archive(
        population,
        {
            one_shot_best["smart_phenotype"]: -0.90,
            race_best["smart_phenotype"]: -0.10,
        },
    )
    snapshot = build_pre_race_snapshot(population, archive)
    race_best["fitness"] = -0.95
    one_shot_best["fitness"] = -0.20
    logger = prepare_native_logger()
    summary = make_selection_audit_summary()

    elites, _, summary = reproduce_via_elitism(population, logger, 0, snapshot, summary)

    assert elites[0]["id"] == race_best["id"]
    assert summary["elitism_changed"] is True
    events = read_jsonl(run_dump_dir(params) / "_race_selection_audit_report.jsonl")
    assert events[0]["event"] == "elitism_audit"
    assert events[0]["elite_set_changed"] is True
    assert events[0]["actual_elite_ids"] == [race_best["id"]]
    assert events[0]["counterfactual_elite_ids"] == [one_shot_best["id"]]
