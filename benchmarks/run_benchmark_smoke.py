"""Run a small real-data benchmark for Adam and one evolved optimizer."""

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.benchmark_runner import (
    benchmark_best_optimizer,
    benchmark_best_phenotype,
    create_prebuilt_optimizer,
    load_task_parameters,
    tune_optimizer,
    tune_phenotype,
)


EVOLVED_PHENOTYPE = (
    "alpha_func, beta_func, sigma_func, grad_func = "
    "lambda shape, alpha, grad: tf.constant(9.99916780e-01, dtype=tf.float32), "
    "lambda shape, alpha, beta, grad: grad, "
    "lambda shape, alpha, beta, sigma, grad: tf.constant(3.76354517e-01, dtype=tf.float32), "
    "lambda shape, alpha, beta, sigma, grad: tf.math.divide_no_nan(beta, alpha)"
)


def smoke_parameters(task, output_dir):
    parameters = load_task_parameters(task, use_test_data=True)
    parameters.update(
        {
            "EXPERIMENT_NAME": "benchmark_smoke",
            "LOGS_DIR": str(output_dir / "logs"),
            "RUN": 0,
        }
    )
    return parameters


def run_smoke(task, output_dir, trials, benchmark_repeats, seed):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = smoke_parameters(task, output_dir)
    summaries = {}

    adam_dir = output_dir / "adam"
    adam_study = tune_optimizer(
        optimizer=create_prebuilt_optimizer("adam"),
        task_name=task,
        parameters=parameters,
        n_trials=trials,
        output_dir=adam_dir,
        study_name=f"{task}_adam_smoke",
        seed=seed,
    )
    summaries["adam"] = benchmark_best_optimizer(
        study=adam_study,
        task_name=task,
        parameters=parameters,
        output_dir=adam_dir,
        repeats=benchmark_repeats,
    )

    evolved_dir = output_dir / "evolved"
    evolved_study = tune_phenotype(
        phenotype=EVOLVED_PHENOTYPE,
        task_name=task,
        parameters=parameters,
        n_trials=trials,
        output_dir=evolved_dir,
        study_name=f"{task}_evolved_smoke",
        seed=seed,
    )
    summaries["evolved"] = benchmark_best_phenotype(
        study=evolved_study,
        task_name=task,
        parameters=parameters,
        output_dir=evolved_dir,
        repeats=benchmark_repeats,
    )

    summary_path = output_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True))
    return summaries


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(
        description="Smoke-test the complete benchmark flow with Adam and an evolved optimizer."
    )
    parser.add_argument("--task", default="mnist")
    parser.add_argument(
        "--output-dir", default="benchmarks/benchmark_smoke_results"
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--benchmark-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(arguments)


def main(arguments=None):
    args = parse_args(arguments)
    summaries = run_smoke(
        task=args.task,
        output_dir=args.output_dir,
        trials=args.trials,
        benchmark_repeats=args.benchmark_repeats,
        seed=args.seed,
    )
    for subject, summary in summaries.items():
        print(
            f"{subject}: validation best={summary['best_tuning_score']:.6f}, "
            f"test mean={summary['mean_score']:.6f}"
        )


if __name__ == "__main__":
    main()
