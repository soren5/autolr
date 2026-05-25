import json


def test_f_race_eval_cap_analysis_finds_first_pairwise_separation():
    from utils.analyze_f_race_eval_cap import analyze_generation

    trials = {
        "winner": [-0.90, -0.91, -0.92, -0.93],
        "loser": [-0.10, -0.11, -0.12, -0.13],
    }
    race_stop = {"final_best_key": "winner"}

    generation_row, pairwise_rows = analyze_generation(
        trials,
        race_stop,
        alpha=0.99,
        min_evals=2,
        max_cap=None,
    )

    assert generation_row["cap_to_match_final_best"] == 1
    assert generation_row["cap_to_separate_final_best"] == 2
    assert generation_row["final_best_separated_at_observed_max"] is True
    assert pairwise_rows[0]["candidate_key"] == "loser"
    assert pairwise_rows[0]["first_separation_cap"] == 2


def test_f_race_eval_cap_prefers_iteration_trials_over_partial_report(tmp_path):
    from utils.analyze_f_race_eval_cap import analyze_run

    run_dir = tmp_path / "experiment" / "run_1"
    run_dir.mkdir(parents=True)
    events = [
        {
            "generation": 0,
            "event": "candidate_snapshot",
            "key": "winner",
            "valid": True,
            "first_fitness": -0.9,
            "n_evals_before": 3,
        },
        {
            "generation": 0,
            "event": "candidate_snapshot",
            "key": "loser",
            "valid": True,
            "first_fitness": -0.1,
            "n_evals_before": 3,
        },
        {
            "generation": 0,
            "event": "race_stop",
            "final_best_key": "winner",
        },
    ]
    with (run_dir / "_race_f_race_report.jsonl").open("w") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")

    iteration = [
        {"id": 1, "smart_phenotype": "winner", "trials": [-0.9, -0.91, -0.92]},
        {"id": 2, "smart_phenotype": "loser", "trials": [-0.1, -0.11, -0.12]},
    ]
    (run_dir / "iteration_0.json").write_text(json.dumps(iteration))

    generation_rows, pairwise_rows = analyze_run(
        run_dir,
        tmp_path,
        alpha=0.99,
        min_evals=2,
        max_cap=None,
    )

    assert generation_rows[0]["iteration_trial_candidates"] == 2
    assert generation_rows[0]["partial_report_candidates"] == 0
    assert generation_rows[0]["missing_pre_race_trial_count"] == 0
    assert pairwise_rows[0]["first_separation_cap"] == 2
