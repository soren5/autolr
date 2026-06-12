"""Inspect readability and tuning-specific constant abstraction.

Run the built-in examples:

    python utils/compare_constant_abstraction.py

Compare one or more phenotype files:

    python utils/compare_constant_abstraction.py phenotype_1.txt phenotype_2.txt
"""

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.smart_phenotype import (
    abstract_active_constants,
    abstract_constants,
    advanced_readable_phenotype,
    materialize_constants,
    smart_phenotype,
)


BUILT_IN_PHENOTYPES = {
    "inactive and active constants": (
        "alpha_func, beta_func, sigma_func, grad_func = "
        "lambda shape, alpha, grad: tf.constant(0.25, dtype=tf.float32), "
        "lambda shape, alpha, beta, grad: tf.constant(0.75, dtype=tf.float32), "
        "lambda shape, alpha, beta, sigma, grad: grad, "
        "lambda shape, alpha, beta, sigma, grad: "
        "tf.math.multiply(tf.constant(0.25, dtype=tf.float32), grad)"
    ),
    "two distinct active constants": (
        "alpha_func, beta_func, sigma_func, grad_func = "
        "lambda shape, alpha, grad: grad, "
        "lambda shape, alpha, beta, grad: grad, "
        "lambda shape, alpha, beta, sigma, grad: grad, "
        "lambda shape, alpha, beta, sigma, grad: "
        "tf.math.add("
        "tf.math.multiply(tf.constant(1.0e-03, dtype=tf.float32), grad), "
        "tf.constant(2.5e-01, dtype=tf.float32))"
    ),
    "repeated active constant": (
        "alpha_func, beta_func, sigma_func, grad_func = "
        "lambda shape, alpha, grad: grad, "
        "lambda shape, alpha, beta, grad: grad, "
        "lambda shape, alpha, beta, sigma, grad: grad, "
        "lambda shape, alpha, beta, sigma, grad: "
        "tf.math.add("
        "tf.math.multiply(tf.constant(0.5, dtype=tf.float32), grad), "
        "tf.constant(0.5, dtype=tf.float32))"
    ),
    "negative active constant": (
        "alpha_func, beta_func, sigma_func, grad_func = "
        "lambda shape, alpha, grad: grad, "
        "lambda shape, alpha, beta, grad: grad, "
        "lambda shape, alpha, beta, sigma, grad: grad, "
        "lambda shape, alpha, beta, sigma, grad: "
        "tf.math.add(grad, tf.constant(-2.5e-03, dtype=tf.float32))"
    ),
    "architectural optimizer": (
        "alpha_func, beta_func, sigma_func, grad_func = "
        "lambda layer_count, layer_num, shape, alpha, grad: grad, "
        "lambda layer_count, layer_num, shape, alpha, beta, grad: grad, "
        "lambda layer_count, layer_num, shape, alpha, beta, sigma, grad: grad, "
        "lambda layer_count, layer_num, shape, alpha, beta, sigma, grad: "
        "tf.math.multiply(tf.constant(0.01, dtype=tf.float32), "
        "tf.math.multiply(layer_num, grad))"
    ),
}


def _heading(text, width=88):
    print()
    print("=" * width)
    print(text)
    print("=" * width)


def compare_phenotype(name, phenotype):
    """Print both abstraction results and the proposed implementation's mapping."""

    _heading(name)
    print("\nORIGINAL PHENOTYPE")
    print(phenotype)

    print("\nACTIVE SMART PHENOTYPE")
    try:
        print(smart_phenotype(phenotype))
    except Exception as error:
        print(f"<smart_phenotype failed: {error}>")

    print("\nADVANCED READABLE PHENOTYPE")
    try:
        readable = advanced_readable_phenotype(phenotype)
        print(readable)
    except Exception as error:
        readable = None
        print(f"<advanced_readable_phenotype failed: {error}>")

    print("\nREADABILITY ABSTRACTION")
    if readable is not None:
        print(abstract_constants(readable))
    else:
        print("<unavailable>")

    print("\nACTIVE TUNING ABSTRACTION")
    try:
        template, parameters = abstract_active_constants(phenotype)
        print(template)
        print("\nACTIVE PARAMETER MAPPING")
        if parameters:
            for name, value in parameters.items():
                print(f"{name}: initial={value!r}")
        else:
            print("<no active constants>")

        round_trip = materialize_constants(template, parameters)
        print("\nACTIVE ABSTRACTION ROUND TRIP USING INITIAL VALUES")
        print(round_trip)
        print(f"\nROUND TRIP EXACT TEXT MATCHES ORIGINAL: {round_trip == phenotype}")
        print(
            "ROUND TRIP ACTIVE SMART PHENOTYPE:\n"
            f"{smart_phenotype(round_trip)}"
        )
    except Exception as error:
        print(f"<proposed abstraction failed: {error}>")


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(
        description="Compare legacy and proposed phenotype constant abstraction."
    )
    parser.add_argument(
        "phenotype_files",
        nargs="*",
        help="Optional text files containing full optimizer phenotypes.",
    )
    return parser.parse_args(arguments)


def main(arguments=None):
    args = parse_args(arguments)
    if args.phenotype_files:
        examples = {
            str(path): Path(path).read_text().strip() for path in args.phenotype_files
        }
    else:
        examples = BUILT_IN_PHENOTYPES

    for name, phenotype in examples.items():
        compare_phenotype(name, phenotype)


if __name__ == "__main__":
    main()
