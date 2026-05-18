import json

import pytest


@pytest.mark.unit
def test_race_behavior_key_normalizes_commutative_multiply_order():
    from utils.race_behavior_key import race_behavior_key

    assert race_behavior_key("multiply(lr, grad)") == race_behavior_key("multiply(grad, lr)")


@pytest.mark.unit
def test_race_behavior_key_flattens_nested_multiply_and_add():
    from utils.race_behavior_key import race_behavior_key

    assert race_behavior_key("multiply(a, multiply(c, b))") == "multiply(a, b, c)"
    assert race_behavior_key("add(z, add(y, x))") == "add(x, y, z)"


@pytest.mark.unit
def test_race_behavior_key_preserves_non_commutative_order():
    from utils.race_behavior_key import race_behavior_key

    assert race_behavior_key("subtract(a, b)") != race_behavior_key("subtract(b, a)")


@pytest.mark.unit
def test_race_behavior_key_returns_stable_key_for_malformed_expression():
    from utils.race_behavior_key import race_behavior_key

    assert race_behavior_key("multiply(grad,") == "multiply(grad,"


@pytest.mark.unit
def test_constants_method_normalizes_identity_and_zero_constants():
    from utils.race_behavior_key import race_behavior_key

    assert race_behavior_key("multiply(grad, constant(1.00000000e+00))", method="constants") == "grad"
    assert race_behavior_key("multiply(grad, constant(1.0, dtype=float32))", method="constants") == "grad"
    assert race_behavior_key("add(grad, constant(0.0))", method="constants") == "grad"
    assert race_behavior_key("multiply(grad, constant(0.0))", method="constants") == "constant(0)"
    assert race_behavior_key("constant(1.0, dtype=float32)", method="constants") == "constant(1)"
    assert race_behavior_key("constant(1.00000000e+00)", method="constants") == "constant(1)"


@pytest.mark.unit
def test_safe_method_does_not_apply_constant_identity_simplification():
    from utils.race_behavior_key import race_behavior_key

    assert race_behavior_key("multiply(grad, constant(1.0))") != race_behavior_key("grad")
    assert race_behavior_key("multiply(grad, constant(1.0))", method="constants") == race_behavior_key("grad", method="constants")


@pytest.mark.unit
def test_fingerprint_method_groups_behavioral_equivalents_beyond_syntax():
    from utils.race_behavior_key import race_behavior_key

    assert race_behavior_key("multiply(grad, constant(0.5))", method="fingerprint") == race_behavior_key("divide_no_nan(grad, constant(2.0))", method="fingerprint")
    assert race_behavior_key("subtract(grad, grad)", method="fingerprint") == race_behavior_key("constant(0.0)", method="fingerprint")


@pytest.mark.unit
def test_fingerprint_method_keeps_distinct_probe_behaviors_apart():
    from utils.race_behavior_key import race_behavior_key

    assert race_behavior_key("grad", method="fingerprint") != race_behavior_key("negative(grad)", method="fingerprint")


@pytest.mark.unit
def test_refined_fingerprint_splits_constant_one_from_divide_self():
    from utils.race_behavior_key import race_behavior_key

    assert race_behavior_key("constant(1.0)", method="fingerprint") == race_behavior_key("divide_no_nan(alpha, alpha)", method="fingerprint")
    assert race_behavior_key("constant(1.0)", method="fingerprint_refined") != race_behavior_key("divide_no_nan(alpha, alpha)", method="fingerprint_refined")


@pytest.mark.unit
def test_refined_fingerprint_keeps_clean_identity_aliases_grouped():
    from utils.race_behavior_key import race_behavior_key

    assert race_behavior_key("grad", method="fingerprint_refined") == race_behavior_key("negative(negative(grad))", method="fingerprint_refined")
    assert race_behavior_key("grad", method="fingerprint_refined") == race_behavior_key("multiply(grad, constant(1.0))", method="fingerprint_refined")


@pytest.mark.unit
def test_analyze_records_counts_archive_keys_collapsed_by_behavior_key(tmp_path):
    from utils.analyze_race_behavior_keys import analyze_records, read_iteration_records

    run_dir = tmp_path / "dumps" / "synthetic_experiment" / "run_1"
    run_dir.mkdir(parents=True)
    iteration = [
        {
            "id": 1,
            "key": "multiply(lr, grad)",
            "smart_phenotype": "multiply(lr, grad)",
            "fitness": -0.5,
        },
        {
            "id": 2,
            "key": "multiply(grad, lr)",
            "smart_phenotype": "multiply(grad, lr)",
            "fitness": -0.4,
        },
        {
            "id": 3,
            "key": "subtract(lr, grad)",
            "smart_phenotype": "subtract(lr, grad)",
            "fitness": -0.3,
        },
    ]
    (run_dir / "iteration_0.json").write_text(json.dumps(iteration))

    records = read_iteration_records(tmp_path / "dumps")
    summary_rows, groups = analyze_records(records, examples=2)

    global_summary = summary_rows[0]
    assert global_summary["individuals_seen"] == 3
    assert global_summary["archive_keys"] == 3
    assert global_summary["race_behavior_keys"] == 2
    assert global_summary["collapsed_behavior_groups"] == 1
    assert global_summary["collapsed_archive_keys"] == 2
    assert global_summary["collapsed_individual_occurrences"] == 2
    assert len(groups) == 1
    assert groups[0]["race_behavior_key"] == "multiply(grad, lr)"
    assert groups[0]["archive_keys"] == ["multiply(grad, lr)", "multiply(lr, grad)"]


@pytest.mark.unit
def test_constants_method_collapses_more_groups_than_safe_method(tmp_path):
    from utils.analyze_race_behavior_keys import analyze_records, read_iteration_records

    run_dir = tmp_path / "dumps" / "synthetic_experiment" / "run_1"
    run_dir.mkdir(parents=True)
    iteration = [
        {
            "id": 1,
            "key": "grad",
            "smart_phenotype": "grad",
            "fitness": -0.5,
        },
        {
            "id": 2,
            "key": "multiply(grad, constant(1.0))",
            "smart_phenotype": "multiply(grad, constant(1.0))",
            "fitness": -0.4,
        },
    ]
    (run_dir / "iteration_0.json").write_text(json.dumps(iteration))

    safe_records = read_iteration_records(tmp_path / "dumps", method="safe")
    constants_records = read_iteration_records(tmp_path / "dumps", method="constants")
    safe_summary, _ = analyze_records(safe_records, examples=2, method="safe")
    constants_summary, constants_groups = analyze_records(constants_records, examples=2, method="constants")

    assert safe_summary[0]["collapsed_behavior_groups"] == 0
    assert constants_summary[0]["method"] == "constants"
    assert constants_summary[0]["collapsed_behavior_groups"] == 1
    assert constants_summary[0]["collapsed_archive_keys"] == 2
    assert constants_groups[0]["race_behavior_key"] == "grad"


@pytest.mark.unit
def test_fingerprint_method_collapses_behavioral_equivalents_in_analysis(tmp_path):
    from utils.analyze_race_behavior_keys import analyze_records, read_iteration_records

    run_dir = tmp_path / "dumps" / "synthetic_experiment" / "run_1"
    run_dir.mkdir(parents=True)
    iteration = [
        {
            "id": 1,
            "key": "multiply(grad, constant(0.5))",
            "smart_phenotype": "multiply(grad, constant(0.5))",
            "fitness": -0.5,
        },
        {
            "id": 2,
            "key": "divide_no_nan(grad, constant(2.0))",
            "smart_phenotype": "divide_no_nan(grad, constant(2.0))",
            "fitness": -0.4,
        },
    ]
    (run_dir / "iteration_0.json").write_text(json.dumps(iteration))

    records = read_iteration_records(tmp_path / "dumps", method="fingerprint")
    summary, groups = analyze_records(records, examples=2, method="fingerprint")

    assert summary[0]["method"] == "fingerprint"
    assert summary[0]["race_behavior_keys"] == 1
    assert summary[0]["collapsed_behavior_groups"] == 1
    assert groups[0]["archive_keys"] == [
        "divide_no_nan(grad, constant(2.0))",
        "multiply(grad, constant(0.5))",
    ]


@pytest.mark.unit
def test_refined_fingerprint_reduces_false_positive_grouping_in_analysis(tmp_path):
    from utils.analyze_race_behavior_keys import analyze_records, read_iteration_records

    run_dir = tmp_path / "dumps" / "synthetic_experiment" / "run_1"
    run_dir.mkdir(parents=True)
    iteration = [
        {
            "id": 1,
            "key": "constant(1.0)",
            "smart_phenotype": "constant(1.0)",
            "fitness": -0.5,
        },
        {
            "id": 2,
            "key": "divide_no_nan(alpha, alpha)",
            "smart_phenotype": "divide_no_nan(alpha, alpha)",
            "fitness": -0.4,
        },
    ]
    (run_dir / "iteration_0.json").write_text(json.dumps(iteration))

    coarse_records = read_iteration_records(tmp_path / "dumps", method="fingerprint")
    refined_records = read_iteration_records(tmp_path / "dumps", method="fingerprint_refined")
    coarse_summary, _ = analyze_records(coarse_records, examples=2, method="fingerprint")
    refined_summary, _ = analyze_records(refined_records, examples=2, method="fingerprint_refined")

    assert coarse_summary[0]["collapsed_behavior_groups"] == 1
    assert refined_summary[0]["method"] == "fingerprint_refined"
    assert refined_summary[0]["collapsed_behavior_groups"] == 0
