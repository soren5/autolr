import json

import pytest


def _write_summary(path, scores):
    path.write_text(json.dumps({"scores": scores}))
    return path


def test_parse_args_accepts_summary_and_threshold():
    from utils.analyze_fitness_summary_normality import parse_args

    args = parse_args(
        [
            "--summary",
            "fitness_summary.json",
            "--threshold",
            "0.9999",
        ]
    )

    assert args.summary == "fitness_summary.json"
    assert args.threshold == 0.9999
    assert args.alpha == 0.05


def test_valid_normal_scores_report_lower_tail_cutoff(tmp_path):
    from utils.analyze_fitness_summary_normality import analyze_summary

    summary_path = _write_summary(
        tmp_path / "fitness_summary.json",
        [
            0.8417272567749023,
            0.8391818404197693,
            0.8404726982116699,
            0.8425636291503906,
            0.8398908972740173,
            0.8374000191688538,
            0.8369272947311401,
            0.8414182066917419,
            0.8335636258125305,
            0.8385999798774719,
            0.8336363434791565,
            0.8391273021697998,
            0.8412727117538452,
            0.8429636359214783,
            0.8414182066917419,
            0.8395454287528992,
            0.8416727185249329,
            0.8442909121513367,
            0.8406363725662231,
            0.844036340713501,
            0.8342727422714233,
            0.8369818329811096,
            0.8435454368591309,
            0.839054524898529,
            0.8391636610031128,
            0.8384363651275635,
            0.8411999940872192,
            0.8409636616706848,
            0.8406727313995361,
            0.8444908857345581,
        ],
    )

    report = analyze_summary(summary_path, threshold=0.9999)

    assert report["runs"] == 30
    assert report["normality_test"] == "shapiro"
    assert report["normality_passed"] is True
    assert report["normality_p_value"] == pytest.approx(0.1321906894)
    assert report["mean_score"] == pytest.approx(0.8399709086)
    assert report["std_score"] == pytest.approx(0.0029085410)
    assert report["lower_tail_probability"] == pytest.approx(0.0001)
    assert report["cutoff_score"] == pytest.approx(0.8291539966)


def test_missing_scores_raises_clear_error(tmp_path):
    from utils.analyze_fitness_summary_normality import load_scores

    summary_path = tmp_path / "fitness_summary.json"
    summary_path.write_text(json.dumps({"runs": 30}))

    with pytest.raises(ValueError, match="Missing required 'scores'"):
        load_scores(summary_path)


def test_non_numeric_scores_raise_clear_error(tmp_path):
    from utils.analyze_fitness_summary_normality import load_scores

    summary_path = _write_summary(tmp_path / "fitness_summary.json", [0.1, "bad", 0.2])

    with pytest.raises(ValueError, match="index 1 is not numeric"):
        load_scores(summary_path)


def test_fewer_than_three_scores_raise_clear_error(tmp_path):
    from utils.analyze_fitness_summary_normality import load_scores

    summary_path = _write_summary(tmp_path / "fitness_summary.json", [0.1, 0.2])

    with pytest.raises(ValueError, match="At least 3 scores"):
        load_scores(summary_path)


def test_invalid_threshold_raises_clear_error():
    from utils.analyze_fitness_summary_normality import analyze_scores

    with pytest.raises(ValueError, match="threshold"):
        analyze_scores([0.1, 0.2, 0.3], threshold=1.0)


def test_normality_failure_raises_clear_error():
    from utils.analyze_fitness_summary_normality import analyze_scores

    with pytest.raises(ValueError, match="do not pass"):
        analyze_scores([0.0] * 20 + [100.0] * 10, threshold=0.9999)
