import copy
import csv
import json

import pytest

from tests.helpers import (
    load_archive,
    load_iteration,
    population_signature,
    run_dump_dir,
)


@pytest.mark.smoke
def test_engine_fake_fitness_writes_expected_artifacts(
    tiny_engine_parameters,
    deterministic_evaluator,
):
    import sge

    population = sge.evolutionary_algorithm(
        parameters=tiny_engine_parameters,
        evaluation_function=deterministic_evaluator,
    )

    dump_dir = run_dump_dir(tiny_engine_parameters)
    assert population
    assert (dump_dir / "_parameters.json").is_file()
    assert (dump_dir / "_progress_report.csv").is_file()
    assert (dump_dir / "iteration_0.json").is_file()
    assert (dump_dir / "population_3.json").is_file()
    assert (dump_dir / "z-archive_3.json").is_file()
    assert (dump_dir / "builtinstate_3").is_file()
    assert (dump_dir / "numpystate_3").is_file()

    archive = load_archive(tiny_engine_parameters, 3)
    evaluated_population = load_iteration(tiny_engine_parameters, 2)
    smart_keys = {indiv["smart_phenotype"] for indiv in evaluated_population}
    assert smart_keys.issubset(set(archive))
    assert all(entry["fitness"] <= 0 for entry in archive.values())


@pytest.mark.smoke
def test_engine_resume_matches_uninterrupted_fake_fitness_run(
    tiny_engine_parameters,
    deterministic_evaluator,
):
    import sge

    uninterrupted = sge.evolutionary_algorithm(
        parameters=tiny_engine_parameters,
        evaluation_function=deterministic_evaluator,
    )

    resumed_parameters = copy.deepcopy(tiny_engine_parameters)
    resumed_parameters["RESUME"] = 1
    resumed_parameters["LOAD_ARCHIVE"] = True

    resumed = sge.evolutionary_algorithm(
        parameters=resumed_parameters,
        evaluation_function=deterministic_evaluator,
    )

    assert population_signature(resumed) == population_signature(uninterrupted)


@pytest.mark.smoke
def test_engine_racing_resume_matches_uninterrupted_fake_fitness_run(
    tiny_engine_parameters,
    deterministic_evaluator,
):
    import sge

    tiny_engine_parameters["RACING"] = True
    tiny_engine_parameters["RACING_MAX_EVALS"] = 3
    tiny_engine_parameters["RACING_MIN_EVALS"] = 2

    uninterrupted = sge.evolutionary_algorithm(
        parameters=tiny_engine_parameters,
        evaluation_function=deterministic_evaluator,
    )

    resumed_parameters = copy.deepcopy(tiny_engine_parameters)
    resumed_parameters["RESUME"] = 1
    resumed_parameters["LOAD_ARCHIVE"] = True

    resumed = sge.evolutionary_algorithm(
        parameters=resumed_parameters,
        evaluation_function=deterministic_evaluator,
    )

    assert population_signature(resumed) == population_signature(uninterrupted)


@pytest.mark.smoke
def test_engine_racing_writes_utility_logging_artifacts(
    tiny_engine_parameters,
    deterministic_evaluator,
):
    import sge

    tiny_engine_parameters["RACING"] = True
    tiny_engine_parameters["RACING_MAX_EVALS"] = 3
    tiny_engine_parameters["RACING_MIN_EVALS"] = 2

    sge.evolutionary_algorithm(
        parameters=tiny_engine_parameters,
        evaluation_function=deterministic_evaluator,
    )

    dump_dir = run_dump_dir(tiny_engine_parameters)
    expected_files = [
        "_race_f_race_report.jsonl",
        "_race_f_race_summary.csv",
        "_race_selection_audit_report.jsonl",
        "_race_selection_audit_summary.csv",
    ]
    for file_name in expected_files:
        assert (dump_dir / file_name).is_file()

    with (dump_dir / "_race_f_race_report.jsonl").open("r") as report_file:
        assert all(json.loads(line) for line in report_file)

    with (dump_dir / "_race_f_race_summary.csv").open("r") as summary_file:
        assert list(csv.DictReader(summary_file))

    sorted_names = sorted(["_progress_report.csv"] + expected_files + ["builtinstate_3"])
    assert sorted_names == [
        "_progress_report.csv",
        "_race_f_race_report.jsonl",
        "_race_f_race_summary.csv",
        "_race_selection_audit_report.jsonl",
        "_race_selection_audit_summary.csv",
        "builtinstate_3",
    ]
