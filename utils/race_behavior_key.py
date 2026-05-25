from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json

import numpy as np


COMMUTATIVE_ASSOCIATIVE_OPS = {"add", "multiply"}
CONSTANT_FUNCTION = "constant"


@dataclass(frozen=True)
class FunctionExpression:
    name: str
    args: tuple


@dataclass(frozen=True)
class AtomExpression:
    text: str


def race_behavior_key(archive_key, method="safe"):
    """Return a conservative canonical behavior key for an active optimizer key."""

    expression = parse_expression(str(archive_key).strip())
    if method == "safe":
        return render_canonical(expression)
    if method == "constants":
        return render_canonical_with_constants(expression)
    if method == "fingerprint":
        return behavior_fingerprint(expression)
    if method == "fingerprint_refined":
        return behavior_fingerprint(expression, refined=True)
    raise ValueError(f"Unknown race behavior key method: {method}")


def parse_expression(text):
    text = text.strip()
    parsed = parse_function_call(text)
    if parsed is None:
        return AtomExpression(text)
    name, arg_texts = parsed
    return FunctionExpression(name, tuple(parse_expression(arg) for arg in arg_texts))


def parse_function_call(text):
    if not text.endswith(")"):
        return None
    open_index = text.find("(")
    if open_index <= 0:
        return None
    name = text[:open_index].strip()
    if not name or not name.replace("_", "").replace(".", "").isalnum():
        return None
    inner = text[open_index + 1:-1]
    if not outer_call_is_balanced(inner):
        return None
    args = split_top_level_args(inner)
    if args is None:
        return None
    return name, args


def outer_call_is_balanced(inner):
    depth = 0
    for char in inner:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def split_top_level_args(text):
    args = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return None
        elif char == "," and depth == 0:
            arg = text[start:index].strip()
            if arg == "":
                return None
            args.append(arg)
            start = index + 1
    if depth != 0:
        return None
    final_arg = text[start:].strip()
    if final_arg == "":
        return None
    args.append(final_arg)
    return args


def render_canonical(expression):
    if isinstance(expression, AtomExpression):
        return expression.text

    rendered_args = []
    for arg in expression.args:
        if expression.name in COMMUTATIVE_ASSOCIATIVE_OPS and is_same_function(arg, expression.name):
            rendered_args.extend(render_canonical(child) for child in arg.args)
        else:
            rendered_args.append(render_canonical(arg))

    if expression.name in COMMUTATIVE_ASSOCIATIVE_OPS:
        rendered_args = sorted(rendered_args)

    return f"{expression.name}({', '.join(rendered_args)})"


def is_same_function(expression, name):
    return isinstance(expression, FunctionExpression) and expression.name == name


def render_canonical_with_constants(expression):
    simplified = simplify_constants(expression)
    return render_canonical(simplified)


def simplify_constants(expression):
    if isinstance(expression, AtomExpression):
        return expression

    simplified_args = tuple(simplify_constants(arg) for arg in expression.args)
    if expression.name == CONSTANT_FUNCTION:
        return normalize_constant_expression(simplified_args)
    if expression.name == "add":
        return simplify_addition(simplified_args)
    if expression.name == "multiply":
        return simplify_multiplication(simplified_args)
    return FunctionExpression(expression.name, simplified_args)


def normalize_constant_expression(args):
    if len(args) == 0:
        return FunctionExpression(CONSTANT_FUNCTION, args)
    argument = args[0]
    if not isinstance(argument, AtomExpression):
        return FunctionExpression(CONSTANT_FUNCTION, args)
    decimal_value = parse_decimal(argument.text)
    if decimal_value is None:
        return FunctionExpression(CONSTANT_FUNCTION, args)
    return FunctionExpression(CONSTANT_FUNCTION, (AtomExpression(render_decimal(decimal_value)),))


def simplify_addition(args):
    kept_args = [arg for arg in args if not expression_is_numeric_constant(arg, Decimal("0"))]
    if not kept_args:
        return numeric_constant_expression(Decimal("0"))
    if len(kept_args) == 1:
        return kept_args[0]
    return FunctionExpression("add", tuple(kept_args))


def simplify_multiplication(args):
    if any(expression_is_numeric_constant(arg, Decimal("0")) for arg in args):
        return numeric_constant_expression(Decimal("0"))
    kept_args = [arg for arg in args if not expression_is_numeric_constant(arg, Decimal("1"))]
    if not kept_args:
        return numeric_constant_expression(Decimal("1"))
    if len(kept_args) == 1:
        return kept_args[0]
    return FunctionExpression("multiply", tuple(kept_args))


def expression_is_numeric_constant(expression, expected_value):
    value = numeric_constant_value(expression)
    return value is not None and value == expected_value


def numeric_constant_value(expression):
    if not isinstance(expression, FunctionExpression):
        return None
    if expression.name != CONSTANT_FUNCTION or len(expression.args) != 1:
        return None
    argument = expression.args[0]
    if not isinstance(argument, AtomExpression):
        return None
    return parse_decimal(argument.text)


def numeric_constant_expression(value):
    return FunctionExpression(CONSTANT_FUNCTION, (AtomExpression(render_decimal(value)),))


def parse_decimal(text):
    try:
        return Decimal(str(text).strip())
    except InvalidOperation:
        return None


def render_decimal(value):
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal("1")))
    return format(normalized, "f")


def refined_race_behavior_key(archive_key, constants):
    expression = parse_expression(str(archive_key).strip())
    probes = fingerprint_probes() + domain_refinement_probes() + probes_for_constants(constants)
    return behavior_fingerprint(expression, probes=probes)


def behavior_fingerprint(expression, refined=False, probes=None):
    try:
        if probes is None:
            probes = refined_fingerprint_probes(expression) if refined else fingerprint_probes()
        outputs = [
            serialize_probe_output(evaluate_expression(expression, probe), probe)
            for probe in probes
        ]
    except Exception:
        return "fingerprint_error:" + render_canonical_with_constants(expression)
    payload = json.dumps(outputs, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return "fingerprint:" + digest


def fingerprint_probes():
    base = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float64)
    return [
        {
            "grad": base,
            "alpha": np.array([0.25, 0.5, 1.0, 1.5, 2.0], dtype=np.float64),
            "beta": np.array([1.0, 0.75, 0.5, 0.25, 0.0], dtype=np.float64),
            "sigma": np.array([-1.0, -0.25, 0.25, 0.75, 1.25], dtype=np.float64),
        },
        {
            "grad": np.array([1.0, -1.0, 3.0, -3.0, 0.125], dtype=np.float64),
            "alpha": np.array([2.0, -2.0, 0.5, -0.5, 1.0], dtype=np.float64),
            "beta": np.array([0.1, 0.2, 0.4, 0.8, 1.6], dtype=np.float64),
            "sigma": np.array([1.6, 0.8, 0.4, 0.2, 0.1], dtype=np.float64),
        },
        {
            "grad": np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float64),
            "alpha": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            "beta": np.array([[4.0, 3.0], [2.0, 1.0]], dtype=np.float64),
            "sigma": np.array([[0.5, -0.5], [1.5, -1.5]], dtype=np.float64),
        },
    ]


def refined_fingerprint_probes(expression):
    probes = list(fingerprint_probes())
    probes.extend(domain_refinement_probes())
    return probes


def domain_refinement_probes():
    eps = 1e-7
    return [
        {
            "grad": np.array([0.0, 1.0, -1.0, 0.5, -0.5], dtype=np.float64),
            "alpha": np.array([1.0, 0.0, -1.0, 0.5, -0.5], dtype=np.float64),
            "beta": np.array([1.0, -1.0, 0.0, 0.5, -0.5], dtype=np.float64),
            "sigma": np.array([1.0, -1.0, 0.5, 0.0, -0.5], dtype=np.float64),
        },
        {
            "grad": np.array([0.125, 0.5, 1.0, 2.0, 4.0], dtype=np.float64),
            "alpha": np.array([0.25, 0.75, 1.25, 2.5, 5.0], dtype=np.float64),
            "beta": np.array([0.1, 0.3, 0.9, 1.7, 3.3], dtype=np.float64),
            "sigma": np.array([0.2, 0.4, 0.8, 1.6, 3.2], dtype=np.float64),
        },
        {
            "grad": np.array([-4.0, -2.0, -1.0, -0.5, -0.125], dtype=np.float64),
            "alpha": np.array([-3.0, -1.5, -0.75, -0.25, 0.25], dtype=np.float64),
            "beta": np.array([-2.5, -1.25, -0.625, 0.125, 0.5], dtype=np.float64),
            "sigma": np.array([-1.75, -0.875, -0.375, 0.375, 0.75], dtype=np.float64),
        },
        {
            "grad": np.array([-eps, 0.0, eps, 1.0, -1.0], dtype=np.float64),
            "alpha": np.array([eps, -eps, 0.0, 2.0, -2.0], dtype=np.float64),
            "beta": np.array([0.0, eps, -eps, 3.0, -3.0], dtype=np.float64),
            "sigma": np.array([eps, 0.0, -eps, 4.0, -4.0], dtype=np.float64),
        },
        {
            "grad": np.array([[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]], dtype=np.float64),
            "alpha": np.array([[1.0, 0.0, 2.0], [-1.0, 3.0, -3.0]], dtype=np.float64),
            "beta": np.array([[0.5, -0.5, 0.0], [1.5, -1.5, 2.5]], dtype=np.float64),
            "sigma": np.array([[2.0, -2.0, 1.0], [0.0, 0.25, -0.25]], dtype=np.float64),
        },
    ]


def constant_targeted_probes(expression):
    constants = sorted(extract_numeric_constants(expression))
    return probes_for_constants(constants)


def probes_for_constants(constants):
    probes = []
    for constant in sorted(constants)[:12]:
        values = targeted_values_for_constant(constant)
        probes.append(make_targeted_probe(values))
    return probes


def extract_numeric_constants(expression):
    constants = set()
    if isinstance(expression, FunctionExpression):
        value = constant_numeric_value(expression)
        if value is not None and np.isfinite(value):
            constants.add(float(value))
        for arg in expression.args:
            constants.update(extract_numeric_constants(arg))
    return constants


def targeted_values_for_constant(constant):
    eps = max(abs(constant) * 1e-6, 1e-7)
    values = [0.0, 1.0, -1.0, constant, -constant, constant - eps, constant + eps]
    if constant != 0:
        values.extend([1.0 / constant, -1.0 / constant])
    return values[:9]


def make_targeted_probe(values):
    array = np.array(values, dtype=np.float64)
    return {
        "grad": array,
        "alpha": np.roll(array, 1),
        "beta": np.roll(array, 2),
        "sigma": np.roll(array, 3),
    }


def evaluate_expression(expression, variables):
    if isinstance(expression, AtomExpression):
        if expression.text in variables:
            return variables[expression.text]
        decimal_value = parse_decimal(expression.text)
        if decimal_value is not None:
            return np.array(float(decimal_value), dtype=np.float64)
        raise ValueError(f"Unknown atom in fingerprint expression: {expression.text}")

    name = expression.name
    if name == "constant":
        if len(expression.args) == 0:
            raise ValueError("constant requires a value")
        value = constant_numeric_value(expression)
        if value is None:
            raise ValueError(f"Unsupported constant expression: {render_canonical(expression)}")
        return np.array(value, dtype=np.float64)
    args = [evaluate_expression(arg, variables) for arg in expression.args]
    if name == "add":
        with np.errstate(all="ignore"):
            return args[0] + args[1]
    if name == "multiply":
        with np.errstate(all="ignore"):
            return args[0] * args[1]
    if name == "subtract":
        with np.errstate(all="ignore"):
            return args[0] - args[1]
    if name == "negative":
        with np.errstate(all="ignore"):
            return -args[0]
    if name == "square":
        with np.errstate(all="ignore"):
            return np.square(args[0])
    if name == "sqrt":
        with np.errstate(all="ignore"):
            return np.sqrt(args[0])
    if name == "pow":
        with np.errstate(all="ignore"):
            return np.power(args[0], args[1])
    if name == "divide_no_nan":
        numerator, denominator = args
        with np.errstate(divide="ignore", invalid="ignore"):
            shape = np.broadcast(numerator, denominator).shape
            return np.divide(
                numerator,
                denominator,
                out=np.zeros(shape, dtype=np.float64),
                where=denominator != 0,
            )
    raise ValueError(f"Unsupported fingerprint function: {name}")


def constant_numeric_value(expression):
    if not isinstance(expression, FunctionExpression):
        return None
    if expression.name != "constant" or len(expression.args) == 0:
        return None
    first_arg = expression.args[0]
    if not isinstance(first_arg, AtomExpression):
        return None
    value = parse_decimal(first_arg.text)
    if value is None:
        return None
    return float(value)


def serialize_probe_output(output, probe):
    output = np.asarray(output, dtype=np.float64)
    if output.shape == ():
        output = np.full_like(probe["grad"], float(output), dtype=np.float64)
    output = np.nan_to_num(output, nan=123456789.0, posinf=987654321.0, neginf=-987654321.0)
    rounded = np.round(output, decimals=8)
    return rounded.tolist()
