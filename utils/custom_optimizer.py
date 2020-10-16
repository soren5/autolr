import tensorflow as tf
from tensorflow import keras
from tensorflow.python.training import training_ops

class CustomOptimizer(keras.optimizers.Optimizer):
    def __init__(self,
                            name="CustomOptimizer",
                            grad_func=None,
                            alpha=None,
                            alpha_func=None,
                            beta=None,
                            beta_func=None,
                            sigma=None,
                            sigma_func=None,
                            **kwargs):

        super(CustomOptimizer, self).__init__(name, **kwargs)

        self._alpha_dict = alpha
        self._beta_dict = beta
        self._sigma_dict = sigma

        self._alpha_func = alpha_func
        self._beta_func = beta_func
        self._sigma_func = sigma_func
        self._grad_func = grad_func

    def _create_slots(self, var_list):
        pass

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(CustomOptimizer, self)._prepare_local(var_device, var_dtype, apply_state)


    def _resource_apply_dense(self, grad, var, apply_state=None):
        #print("_resource_apply_dense")
        variable_name = var.name
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                                        or self._fallback_apply_state(var_device, var_dtype))
        
        if self._alpha_func != None:
            training_ops.resource_apply_gradient_descent(
                    self._alpha_dict[variable_name].handle, tf.constant(1.0), self._alpha_func(var.shape, self._alpha_dict[variable_name], grad), use_locking=self._use_locking)
            if self._beta_func != None:
                training_ops.resource_apply_gradient_descent(
                        self._beta_dict[variable_name].handle, tf.constant(1.0), self._beta_func(var.shape, self._alpha_dict[variable_name], self._beta_dict[variable_name], grad), use_locking=self._use_locking)
                if self._sigma_func!= None:
                    training_ops.resource_apply_gradient_descent(
                            self._sigma_dict[variable_name].handle, tf.constant(1.0), self._sigma_func(var.shape, self._alpha_dict[variable_name], self._beta_dict[variable_name], self._sigma_dict[variable_name], grad), use_locking=self._use_locking)
        
        foo = training_ops.resource_apply_gradient_descent(
                var.handle, tf.constant(1.0), self._grad_func(var.shape, self._alpha_dict[variable_name], self._beta_dict[variable_name], self._sigma_dict[variable_name], grad), use_locking=self._use_locking)
        return foo

    def get_config(self):
        config = {
                'lr': bfloat16(K.get_value(self.lr)),
        }
        base_config = super(CustomOptimizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
