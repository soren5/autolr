import tensorflow as tf
from tensorflow import keras
from tensorflow.python.training import training_ops
from optimizers.custom_optimizer import CustomOptimizer

class ADES(CustomOptimizer):
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


    def _resource_apply_dense(self, grad, var, apply_state=None):
        #print("_resource_apply_dense")
        variable_name = var.name
        #print(variable_name)
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                                        or self._fallback_apply_state(var_device, var_dtype))
        
        training_ops.resource_apply_gradient_descent(
                    self._alpha_dict[variable_name].handle, tf.constant(1.0), self._alpha_func(var.shape, self._alpha_dict[variable_name], grad), use_locking=self._use_locking)
            
        foo = training_ops.resource_apply_gradient_descent(
                var.handle, tf.constant(1.0), self._grad_func(var.shape, self._alpha_dict[variable_name], grad), use_locking=self._use_locking)
        return foo