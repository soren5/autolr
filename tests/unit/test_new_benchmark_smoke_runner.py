from types import SimpleNamespace


def test_smoke_runner_defaults_are_small():
    from benchmarks.run_new_benchmark_smoke import parse_args

    arguments = parse_args([])

    assert arguments.task == "mnist"
    assert arguments.trials == 3
    assert arguments.benchmark_repeats == 1
    assert not hasattr(arguments, "epochs")
    assert not hasattr(arguments, "batch_size")
    assert not hasattr(arguments, "train_limit")
    assert not hasattr(arguments, "test_limit")


def test_smoke_parameters_use_test_configuration_without_resource_overrides(tmp_path):
    from benchmarks.run_new_benchmark_smoke import smoke_parameters

    parameters = smoke_parameters("mnist", tmp_path)

    assert parameters["TEST_SIZE"] == 10000
    assert parameters["VALIDATION_SIZE"] == 7000
    assert parameters["EPOCHS"] == 1000
    assert parameters["PATIENCE"] == 10001
    assert parameters["BATCH_SIZE"] == 1000
    assert "BENCHMARK_TRAIN_LIMIT" not in parameters
    assert "BENCHMARK_TEST_LIMIT" not in parameters
    assert parameters["LOGS_DIR"] == str(tmp_path / "logs")


def test_smoke_runner_executes_both_subjects(monkeypatch, tmp_path):
    import benchmarks.run_new_benchmark_smoke as smoke

    calls = []
    study = SimpleNamespace()

    monkeypatch.setattr(smoke, "smoke_parameters", lambda *args, **kwargs: {})
    monkeypatch.setattr(smoke, "create_prebuilt_optimizer", lambda name: "adam")
    monkeypatch.setattr(
        smoke,
        "tune_optimizer",
        lambda **kwargs: calls.append(("tune", "adam")) or study,
    )
    monkeypatch.setattr(
        smoke,
        "benchmark_best_optimizer",
        lambda **kwargs: calls.append(("benchmark", "adam"))
        or {"best_tuning_score": 0.5, "mean_score": 0.4},
    )
    monkeypatch.setattr(
        smoke,
        "tune_phenotype",
        lambda **kwargs: calls.append(("tune", "evolved")) or study,
    )
    monkeypatch.setattr(
        smoke,
        "benchmark_best_phenotype",
        lambda **kwargs: calls.append(("benchmark", "evolved"))
        or {"best_tuning_score": 0.6, "mean_score": 0.5},
    )

    summaries = smoke.run_smoke("mnist", tmp_path, 2, 1, 7)

    assert calls == [
        ("tune", "adam"),
        ("benchmark", "adam"),
        ("tune", "evolved"),
        ("benchmark", "evolved"),
    ]
    assert summaries["adam"]["mean_score"] == 0.4
    assert summaries["evolved"]["mean_score"] == 0.5
    assert (tmp_path / "smoke_summary.json").is_file()
