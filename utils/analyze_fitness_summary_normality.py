#!/usr/bin/env python3
"""Fit a normal distribution to fitness-run scores after a normality check."""

import argparse
import json
import math
import statistics
from pathlib import Path

from scipy import stats


DEFAULT_ALPHA = 0.05


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(
        description=(
            "Read a fitness_summary.json file, test score normality, and report "
            "a conservative lower-tail cutoff from the fitted normal distribution."
        )
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Path to a fitness_summary.json file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help=(
            "Probability mass expected above the cutoff. For example, 0.9999 "
            "reports x such that P(score >= x) = 0.9999."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Normality-test rejection threshold. Default: 0.05.",
    )
    return parser.parse_args(arguments)


def load_scores(summary_path):
    summary_path = Path(summary_path)
    with summary_path.open() as summary_file:
        data = json.load(summary_file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {summary_path}")
    if "scores" not in data:
        raise ValueError(f"Missing required 'scores' field in {summary_path}")
    if not isinstance(data["scores"], list) or not data["scores"]:
        raise ValueError(f"'scores' must be a non-empty list in {summary_path}")

    scores = []
    for index, score in enumerate(data["scores"]):
        try:
            numeric_score = float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Score at index {index} is not numeric: {score!r}"
            ) from exc
        if not math.isfinite(numeric_score):
            raise ValueError(f"Score at index {index} is not finite: {score!r}")
        scores.append(numeric_score)

    if len(scores) < 3:
        raise ValueError("At least 3 scores are required for the Shapiro-Wilk test")
    return scores


def validate_thresholds(threshold, alpha):
    if not 0 < threshold < 1:
        raise ValueError("--threshold must satisfy 0 < threshold < 1")
    if not 0 < alpha < 1:
        raise ValueError("--alpha must satisfy 0 < alpha < 1")


def analyze_scores(scores, threshold, alpha=DEFAULT_ALPHA):
    validate_thresholds(threshold, alpha)

    mean_score = statistics.mean(scores)
    std_score = statistics.stdev(scores)
    if std_score <= 0:
        raise ValueError("Score standard deviation must be positive")

    shapiro_result = stats.shapiro(scores)
    shapiro_statistic = float(shapiro_result.statistic)
    shapiro_p_value = float(shapiro_result.pvalue)
    normality_passed = shapiro_p_value >= alpha
    if not normality_passed:
        raise ValueError(
            "Scores do not pass the Shapiro-Wilk normality test: "
            f"p_value={shapiro_p_value}, alpha={alpha}"
        )

    lower_tail_probability = 1.0 - threshold
    cutoff_score = stats.norm(loc=mean_score, scale=std_score).ppf(
        lower_tail_probability
    )

    return {
        "runs": len(scores),
        "mean_score": float(mean_score),
        "std_score": float(std_score),
        "normality_test": "shapiro",
        "normality_statistic": shapiro_statistic,
        "normality_p_value": shapiro_p_value,
        "alpha": float(alpha),
        "normality_passed": normality_passed,
        "threshold": float(threshold),
        "lower_tail_probability": float(lower_tail_probability),
        "cutoff_score": float(cutoff_score),
    }


def analyze_summary(summary_path, threshold, alpha=DEFAULT_ALPHA):
    scores = load_scores(summary_path)
    report = analyze_scores(scores, threshold=threshold, alpha=alpha)
    report["summary_path"] = str(Path(summary_path))
    return report


def main(arguments=None):
    args = parse_args(arguments)
    try:
        report = analyze_summary(args.summary, args.threshold, alpha=args.alpha)
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
