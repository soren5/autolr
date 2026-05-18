import tensorflow as tf

from optimizers.custom_optimizer import CustomOptimizer
from utils.smart_phenotype import smart_phenotype


NEW_PADDING_PHENOTYPE = (
    "alpha_func, beta_func, sigma_func, grad_func = "
    "lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.constant(0.0, dtype=tf.float32), "
    "lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.constant(0.0, dtype=tf.float32), "
    "lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.constant(0.0, dtype=tf.float32), "
    "lambda strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: padding"
)


LEGACY_ARCHITECTURE_PHENOTYPE = (
    "alpha_func, beta_func, sigma_func, grad_func = "
    "lambda strides, kernel_size, filters, dilation_rate, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.constant(0.0, dtype=tf.float32), "
    "lambda strides, kernel_size, filters, dilation_rate, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.constant(0.0, dtype=tf.float32), "
    "lambda strides, kernel_size, filters, dilation_rate, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.constant(0.0, dtype=tf.float32), "
    "lambda strides, kernel_size, filters, dilation_rate, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: grad"
)


AGGREGATOR_PADDING_PHENOTYPE = (
    "alpha_func, beta_func, sigma_func, grad_func, momentum_const, variance_const, layer_wise_lr_const = "
    "lambda momentum, variance, layer_wise_lr, strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, grad: tf.constant(0.0, dtype=tf.float32), "
    "lambda momentum, variance, layer_wise_lr, strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, grad: tf.constant(0.0, dtype=tf.float32), "
    "lambda momentum, variance, layer_wise_lr, strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.constant(0.0, dtype=tf.float32), "
    "lambda momentum, variance, layer_wise_lr, strides, kernel_size, filters, dilation_rate, padding, units, pool_size, layer_count, layer_num, shape, alpha, beta, sigma, grad: padding, "
    "0.0, 0.0, 0.0"
)


def _conv_model():
    inputs = tf.keras.Input(shape=(8, 8, 1))
    x = tf.keras.layers.Conv2D(2, 3, padding="same", name="conv_same")(inputs)
    x = tf.keras.layers.Conv2D(2, 3, padding="valid", name="conv_valid")(x)
    return tf.keras.Model(inputs, x)


def test_custom_optimizer_exposes_convolution_padding_constants():
    model = _conv_model()
    optimizer = CustomOptimizer(phen=NEW_PADDING_PHENOTYPE, model=model)

    assert optimizer._variables_used["padding"] is True

    padding_by_weight = {
        weight_name: float(value.numpy())
        for weight_name, value in optimizer._padding.items()
        if value is not None
    }
    assert padding_by_weight["conv_same/kernel:0"] == 1.0
    assert padding_by_weight["conv_same/bias:0"] == 1.0
    assert padding_by_weight["conv_valid/kernel:0"] == 0.0
    assert padding_by_weight["conv_valid/bias:0"] == 0.0


def test_new_padding_signature_is_passed_to_deep_architecture_optimizer():
    model = _conv_model()
    optimizer = CustomOptimizer(phen=NEW_PADDING_PHENOTYPE, model=model)
    variable = model.get_layer("conv_same").kernel

    parameters = optimizer._get_optimizer_type_parameters(
        grad=tf.ones_like(variable),
        var=variable,
        variable_name=variable.name,
    )

    assert len(parameters) == 12
    assert float(parameters[4].numpy()) == 1.0


def test_padding_is_part_of_existing_architecture_optimizer_parameter_contract():
    model = _conv_model()
    optimizer = CustomOptimizer(phen=NEW_PADDING_PHENOTYPE, model=model)
    variable = model.get_layer("conv_same").kernel

    parameters = optimizer._get_optimizer_type_parameters(
        grad=tf.ones_like(variable),
        var=variable,
        variable_name=variable.name,
    )

    assert optimizer.optimizer_type == "deep_architecture_optimizer"
    assert float(parameters[4].numpy()) == 1.0


def test_smart_phenotype_accepts_new_and_legacy_architecture_signatures():
    assert smart_phenotype(NEW_PADDING_PHENOTYPE) == "padding"
    assert smart_phenotype(LEGACY_ARCHITECTURE_PHENOTYPE) == "grad"
    assert smart_phenotype(AGGREGATOR_PADDING_PHENOTYPE) == "padding"
