import tensorflow as tf
from tensorflow import keras
from tensorflow.python.training import training_ops

class ADES(keras.optimizers.Optimizer):
    def __init__(self,
                            name="ADES",
                            alpha=None,
                            beta_1=0.08922,
                            beta_2=0.0891,
                            **kwargs):

        super(ADES, self).__init__(name, **kwargs)

        self._alpha_dict = alpha
        """
        self._alpha_func = lambda shape, alpha, grad: tf.math.add(
                tf.math.multiply(
                    tf.constant(beta_1, shape=shape, dtype=tf.float32),
                    tf.math.multiply(
                        alpha,
                        alpha
                    )
                ),
                tf.math.add(
                    tf.math.multiply(
                        tf.math.multiply(
                            alpha,
                            grad
                        ),
                        tf.constant(beta_2, shape=shape, dtype=tf.float32)
                    ),
                    tf.math.multiply(
                        tf.constant(beta_2, shape=shape, dtype=tf.float32),
                        grad
                    )
                )
            )
        """
        self._alpha_func = lambda shape, alpha, grad: tf.math.multiply(
            tf.constant(1 - beta_1, shape=shape, dtype=tf.float32), 
            tf.math.add(
                alpha, 
                tf.math.multiply(tf.math.add(alpha, tf.constant(beta_2, shape=shape, dtype=tf.float32)), tf.math.add(alpha, grad))))

        self._grad_func = lambda shape, alpha, grad: tf.math.negative(alpha)

    def _create_slots(self, var_list):
        pass

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(ADES, self)._prepare_local(var_device, var_dtype, apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        #print("_resource_apply_dense")
        variable_name = var.name
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                                        or self._fallback_apply_state(var_device, var_dtype))
        
        if self._alpha_func != None:
            training_ops.resource_apply_gradient_descent(
                    self._alpha_dict[variable_name].handle, tf.constant(1.0), self._alpha_func(var.shape, self._alpha_dict[variable_name], grad), use_locking=self._use_locking)
        
        foo = training_ops.resource_apply_gradient_descent(
                var.handle, tf.constant(1.0), self._grad_func(var.shape, self._alpha_dict[variable_name], grad), use_locking=self._use_locking)
        return foo

    def get_config(self):
        config = {
                'lr': bfloat16(K.get_value(self.lr)),
        }
        base_config = super(ADES, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
