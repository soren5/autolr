import tensorflow as tf
from tensorflow import keras
from tensorflow.python.training import training_ops

class Sign(keras.optimizers.Optimizer):
    def __init__(self,
                            name="Sign",
                            beta_1=0.0009,
                            **kwargs):

        super(Sign, self).__init__(name, **kwargs)

        self._grad_func = lambda shape, grad: tf.math.negative(
            tf.math.multiply(
                tf.constant(-beta_1, shape=shape, dtype=tf.float32),
                tf.math.sign(grad)
            )
        )

    def _create_slots(self, var_list):
        pass

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(Sign, self)._prepare_local(var_device, var_dtype, apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        #print("_resource_apply_dense")
        variable_name = var.name
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                                        or self._fallback_apply_state(var_device, var_dtype))
        
        foo = training_ops.resource_apply_gradient_descent(
                var.handle, tf.constant(1.0), self._grad_func(var.shape, grad), use_locking=self._use_locking)
        return foo