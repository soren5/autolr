import pytest

from utils.smart_phenotype import (
    abstract_active_constants,
    abstract_constants,
    advanced_readable_phenotype,
    materialize_constants,
    smart_phenotype,
)


SOURCE_CONSTANTS = {
    "alpha": 0.11,
    "beta": 0.22,
    "sigma": 0.33,
    "grad": 0.44,
}


def basic_phenotype(alpha, beta, sigma, grad):
    return (
        "alpha_func, beta_func, sigma_func, grad_func = "
        f"lambda shape, alpha, grad: {alpha}, "
        f"lambda shape, alpha, beta, grad: {beta}, "
        f"lambda shape, alpha, beta, sigma, grad: {sigma}, "
        f"lambda shape, alpha, beta, sigma, grad: {grad}"
    )


def architectural_phenotype(alpha, beta, sigma, grad):
    return (
        "alpha_func, beta_func, sigma_func, grad_func = "
        f"lambda layer_count, layer_num, shape, alpha, grad: {alpha}, "
        f"lambda layer_count, layer_num, shape, alpha, beta, grad: {beta}, "
        f"lambda layer_count, layer_num, shape, alpha, beta, sigma, grad: {sigma}, "
        f"lambda layer_count, layer_num, shape, alpha, beta, sigma, grad: {grad}"
    )


def source_expression(source_name, dependencies=()):
    expression = "grad"
    for dependency in dependencies:
        expression = f"tf.math.add({dependency}, {expression})"
    return (
        f"tf.math.add(tf.constant({SOURCE_CONSTANTS[source_name]}, "
        f"dtype=tf.float32), {expression})"
    )


def assert_reconstruction_round_trip(phenotype, template, constants):
    reconstructed = materialize_constants(template, constants)

    assert reconstructed == phenotype
    assert "CONST_" not in reconstructed
    assert smart_phenotype(reconstructed) == smart_phenotype(phenotype)


DEPENDENCY_MATRIX = [
    pytest.param((), (), (), {"grad"}, id="grad-only"),
    pytest.param((), (), ("alpha",), {"alpha", "grad"}, id="grad-to-alpha"),
    pytest.param((), (), ("beta",), {"beta", "grad"}, id="grad-to-beta"),
    pytest.param((), (), ("sigma",), {"sigma", "grad"}, id="grad-to-sigma"),
    pytest.param(
        ("alpha",),
        (),
        ("beta",),
        {"alpha", "beta", "grad"},
        id="grad-to-beta-to-alpha",
    ),
    pytest.param(
        (),
        ("alpha",),
        ("sigma",),
        {"alpha", "sigma", "grad"},
        id="grad-to-sigma-to-alpha",
    ),
    pytest.param(
        (),
        ("beta",),
        ("sigma",),
        {"beta", "sigma", "grad"},
        id="grad-to-sigma-to-beta",
    ),
    pytest.param(
        ("alpha",),
        ("beta",),
        ("sigma",),
        {"alpha", "beta", "sigma", "grad"},
        id="grad-to-sigma-to-beta-to-alpha",
    ),
    pytest.param(
        (),
        (),
        ("alpha", "beta", "sigma"),
        {"alpha", "beta", "sigma", "grad"},
        id="grad-direct-branch-to-all",
    ),
    pytest.param(
        ("alpha",),
        ("alpha", "beta"),
        ("beta", "sigma"),
        {"alpha", "beta", "sigma", "grad"},
        id="branching-and-transitive-dependencies",
    ),
]


@pytest.mark.parametrize(
    "beta_dependencies,sigma_dependencies,grad_dependencies,expected_active",
    DEPENDENCY_MATRIX,
)
def test_active_constant_dependency_matrix(
    beta_dependencies,
    sigma_dependencies,
    grad_dependencies,
    expected_active,
):
    phenotype = basic_phenotype(
        source_expression("alpha"),
        source_expression("beta", beta_dependencies),
        source_expression("sigma", sigma_dependencies),
        source_expression("grad", grad_dependencies),
    )

    template, constants = abstract_active_constants(phenotype)

    assert set(constants.values()) == {
        SOURCE_CONSTANTS[source_name] for source_name in expected_active
    }
    assert len(constants) == len(expected_active)
    for source_name, value in SOURCE_CONSTANTS.items():
        source_constant = f"tf.constant({value}, dtype=tf.float32)"
        if source_name in expected_active:
            assert source_constant not in template
        else:
            assert source_constant in template
    assert_reconstruction_round_trip(phenotype, template, constants)

    tuned_values = {
        constant_name: constant_value + 1.0
        for constant_name, constant_value in constants.items()
    }
    tuned = materialize_constants(template, tuned_values)
    assert "CONST_" not in tuned
    for source_name, original_value in SOURCE_CONSTANTS.items():
        if source_name in expected_active:
            assert f"tf.constant({original_value + 1.0}, dtype=tf.float32)" in tuned
            assert f"tf.constant({original_value}, dtype=tf.float32)" not in tuned
        else:
            assert f"tf.constant({original_value}, dtype=tf.float32)" in tuned


def test_abstract_constants_preserves_readability_use_case():
    phenotype = basic_phenotype(
        "grad",
        "grad",
        "grad",
        "tf.math.multiply(tf.constant(1.0e-03, dtype=tf.float32), grad)",
    )

    readable = advanced_readable_phenotype(phenotype)
    abstracted = abstract_constants(readable)

    assert abstracted == "weights = weights - multiply(constant(CONST_0), grad)\n"
    assert "floatCONST" not in abstracted


def test_abstract_constants_numbers_each_readable_occurrence_independently():
    text = "weights = weights - add(constant(0.5), constant(0.5))"

    assert abstract_constants(text) == (
        "weights = weights - add(constant(CONST_0), constant(CONST_1))"
    )


def test_active_constants_keep_equal_source_occurrences_independent():
    phenotype = basic_phenotype(
        "grad",
        "grad",
        "grad",
        "tf.math.add(tf.constant(0.5, dtype=tf.float32), "
        "tf.constant(0.5, dtype=tf.float32))",
    )

    template, constants = abstract_active_constants(phenotype)

    assert constants == {"CONST_0": 0.5, "CONST_1": 0.5}
    assert template.count("CONST_0") == 1
    assert template.count("CONST_1") == 1
    assert_reconstruction_round_trip(phenotype, template, constants)

    independently_tuned = materialize_constants(
        template,
        {"CONST_0": 0.25, "CONST_1": 0.75},
    )
    assert "tf.constant(0.25, dtype=tf.float32)" in independently_tuned
    assert "tf.constant(0.75, dtype=tf.float32)" in independently_tuned


def test_active_constant_dependency_expansion_remains_linked():
    phenotype = basic_phenotype(
        "tf.math.multiply(tf.constant(0.5, dtype=tf.float32), grad)",
        "grad",
        "grad",
        "tf.math.add(alpha, alpha)",
    )

    template, constants = abstract_active_constants(phenotype)
    tuned = materialize_constants(template, {"CONST_0": 0.25})

    assert constants == {"CONST_0": 0.5}
    assert template.count("CONST_0") == 1
    assert smart_phenotype(tuned).count("constant(0.25)") == 2
    assert_reconstruction_round_trip(phenotype, template, constants)


def test_active_constants_leave_inactive_functions_untouched():
    phenotype = basic_phenotype(
        "tf.constant(0.25, dtype=tf.float32)",
        "tf.constant(0.75, dtype=tf.float32)",
        "grad",
        "tf.math.multiply(tf.constant(0.5, dtype=tf.float32), grad)",
    )

    template, constants = abstract_active_constants(phenotype)

    assert constants == {"CONST_0": 0.5}
    assert "tf.constant(0.25, dtype=tf.float32)" in template
    assert "tf.constant(0.75, dtype=tf.float32)" in template
    assert_reconstruction_round_trip(phenotype, template, constants)

    tuned = materialize_constants(template, {"CONST_0": 0.9})
    assert "tf.constant(0.25, dtype=tf.float32)" in tuned
    assert "tf.constant(0.75, dtype=tf.float32)" in tuned
    assert "tf.constant(0.9, dtype=tf.float32)" in tuned


@pytest.mark.parametrize(
    "phenotype_factory,constant_literal,initial_value,replacement",
    [
        pytest.param(basic_phenotype, "1.0e-03", 0.001, 0.002, id="scientific-notation"),
        pytest.param(basic_phenotype, "-2.5e-03", -0.0025, -0.005, id="negative"),
        pytest.param(architectural_phenotype, "0.01", 0.01, 0.02, id="architectural"),
    ],
)
def test_active_constant_reconstruction_across_numeric_and_signature_variants(
    phenotype_factory,
    constant_literal,
    initial_value,
    replacement,
):
    phenotype = phenotype_factory(
        "grad",
        "grad",
        "grad",
        f"tf.math.multiply(tf.constant({constant_literal}, dtype=tf.float32), grad)",
    )

    template, constants = abstract_active_constants(phenotype)
    reconstructed = materialize_constants(template, constants)
    tuned = materialize_constants(template, {"CONST_0": replacement})

    assert constants == {"CONST_0": initial_value}
    assert "CONST_" not in reconstructed
    assert "CONST_" not in tuned
    assert f"tf.constant({repr(initial_value)}, dtype=tf.float32)" in reconstructed
    assert f"tf.constant({repr(replacement)}, dtype=tf.float32)" in tuned
    assert smart_phenotype(reconstructed) == (
        f"multiply(constant({repr(initial_value)}), grad)"
    )
    assert smart_phenotype(tuned) == f"multiply(constant({repr(replacement)}), grad)"


def test_materialize_constants_requires_every_placeholder():
    try:
        materialize_constants("constant(CONST_0)", {})
    except ValueError as error:
        assert "CONST_0" in str(error)
    else:
        raise AssertionError("Expected missing constant value to raise ValueError")
