import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops

class ADES(tf.keras.optimizers.Optimizer):
    def __init__(self,
                            name="ADES",
                            beta1=0.9,
                            beta2=0.999,
                            **kwargs):

        super(ADES, self).__init__(name, **kwargs)

        self.beta1 = beta1
        self.beta2 = beta2
        self._grad_func = lambda shape, y, grad: tf.math.negative(y)
        self._y_func = lambda shape, y, grad: tf.math.multiply(tf.constant(1 - beta1, shape=shape, dtype=tf.float32), tf.math.add(y, tf.math.multiply(tf.math.add(y, tf.constant(beta2, shape=shape, dtype=tf.float32)), tf.math.add(y, grad))))

    def _create_slots(self, var_list):
        for var in var_list:
          self.add_slot(var, 'y')
        pass

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(ADES, self)._prepare_local(var_device, var_dtype, apply_state)


    def _resource_apply_dense(self, grad, var, apply_state=None):
        variable_name = var.name
        y = self.get_slot(var, 'y')
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                                        or self._fallback_apply_state(var_device, var_dtype))
        
        training_ops.resource_apply_gradient_descent(
            y.handle, tf.constant(1.0), self._y_func(var.shape, y, grad), use_locking=self._use_locking)
        foo = training_ops.resource_apply_gradient_descent(
                var.handle, tf.constant(1.0), self._grad_func(var.shape, y, grad), use_locking=self._use_locking)
        return foo



    def get_config(self):
        config = {
                'lr': bfloat16(K.get_value(self.lr)),
        }
        base_config = super(ADES, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == "__main__":
    foo = ADES()