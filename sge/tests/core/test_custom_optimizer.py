import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import ops
#from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tests.core.utilities.native_optimizers_test import SGD, Adagrad, RMSprop, Adam
from tests.core.utilities.custom_optimizer_internal_functions import *
import pytest
class CustomOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self,
                            name="CustomOptimizer",
                            grad_func=grad_func_filler,
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
        self.custom_alpha = []
        self.custom_beta = []
        self.custom_sigma = []
        self.custom_grad = []

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
        #self.custom_alpha.append(self._alpha_dict[variable_name].numpy())
        #self.custom_beta.append(self._beta_dict[variable_name].numpy())
        #self.custom_sigma.append(self._sigma_dict[variable_name].numpy())
        
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


def test_optimizer(native_optimizer=None, alpha_func=None, beta_func=None, sigma_func=None, grad_func=None, test_size=5, debug=False):
    loss_function = lambda: 100*(var2-var1*var1)**2+(1-var1)**2

    alpha1 = tf.Variable(0.0, name="alpha1")
    alpha2 = tf.Variable(0.0, name="alpha2")
    beta1 = tf.Variable(0.0, name="beta1")
    beta2 = tf.Variable(0.0, name="beta2")
    sigma1 = tf.Variable(0.0, name="sigma1")
    sigma2 = tf.Variable(0.0, name="sigma2")

    alpha_dict = {
            "var1:0": alpha1,
            "var2:0": alpha2,
    }
    beta_dict = {
            "var1:0": beta1,
            "var2:0": beta2,
    }
    sigma_dict = {
            "var1:0": sigma1,
            "var2:0": sigma2,
    }

    opt = CustomOptimizer(alpha=alpha_dict, alpha_func=alpha_func, beta=beta_dict, beta_func=beta_func, sigma=sigma_dict, sigma_func=sigma_func, grad_func=grad_func)
    custom_var = []
    custom_grads = []
    custom_loss = 99

    var1 = tf.Variable(0.0, name="var1")
    var2 = tf.Variable(0.0, name="var2")
    for i in range(test_size):
        with tf.GradientTape() as tape:
            loss = loss_function()
            custom_loss = loss if loss < custom_loss else custom_loss
        var_list = [var1, var2]
        grads = (tape.gradient(loss, var_list))
        custom_grads.append(grads)
        opt.apply_gradients(zip(grads, var_list))
        [custom_var.append(x.numpy()) for x in var_list]

    native_var = []
    native_grads = []
    native_loss = 99

    var1 = tf.Variable(0.0, name="var1")
    var2 = tf.Variable(0.0, name="var2")
    for i in range(test_size):
        with tf.GradientTape() as tape:
            loss = loss_function()
            native_loss = loss if loss < native_loss else native_loss
        var_list = [var1, var2]
        grads = (tape.gradient(loss, var_list))
        native_grads.append(grads)
        native_optimizer.apply_gradients(zip(grads, var_list))
        [native_var.append(x.numpy()) for x in var_list]
    
    if debug:
        print("Custom alpha1 values ", [opt.custom_alpha[i] for i in range(0,test_size*2,2)])
        print("Native alpha1 values ", [native_optimizer.native_alpha[i] for i in range(0,test_size*2,2)])
        print("Custom alpha2 values ", [opt.custom_alpha[i] for i in range(1,test_size*2,2)])
        print("Native alpha2 values ", [native_optimizer.native_alpha[i] for i in range(1,test_size*2,2)])
        print("Custom beta1 values ", [opt.custom_beta[i] for i in range(0,test_size*2,2)])
        print("Native beta1 values ", [native_optimizer.native_beta[i] for i in range(0,test_size*2,2)])
        print("Custom beta2 values ", [opt.custom_beta[i] for i in range(1,test_size*2,2)])
        print("Native beta2 values ", [native_optimizer.native_beta[i] for i in range(1,test_size*2,2)])
        print("Custom sigma1 values ", [opt.custom_sigma[i] for i in range(0,test_size*2,2)])
        print("Custom sigma2 values ", [opt.custom_sigma[i] for i in range(1,test_size*2,2)])
        print("Custom var1 values ", [custom_var[i] for i in range(0,test_size*2,2)])
        print("Native var1 values ", [native_var[i] for i in range(0,test_size*2,2)])
        print("Custom var2 values ", [custom_var[i] for i in range(1,test_size*2,2)])
        print("Native var2 values ", [native_var[i] for i in range(1,test_size*2,2)])
        print("Custom grad values ", [[custom_grads[i][0].numpy(), custom_grads[i][1].numpy()] for i in range(0,test_size)])
        print("Native grad values ", [[native_grads[i][0].numpy(), native_grads[i][1].numpy()] for i in range(0,test_size)])

    assert abs(custom_loss.numpy() - native_loss.numpy()) < 0.001

# test_optimizer(
#     native_optimizer=SGD(momentum=0.99), 
#     alpha_func=alpha_func_momentum, 
#     beta_func=beta_func_filler, 
#     grad_func=grad_func_momentum, 
#     test_size=100
# )

# test_optimizer(
#     native_optimizer=Adagrad(), 
#     alpha_func=alpha_func_adagrad, 
#     beta_func=beta_func_filler, 
#     grad_func=grad_func_adagrad, 
#     test_size=100
# )

# test_optimizer(
#     native_optimizer=RMSprop(), 
#     alpha_func=alpha_func_rmsprop, 
#     beta_func=beta_func_rmsprop, 
#     grad_func=grad_func_rmsprop, 
#     test_size=100
# )

# test_optimizer(
#     native_optimizer=Adam(), 
#     alpha_func=alpha_func_adam, 
#     beta_func=beta_func_adam, 
#     sigma_func=sigma_func_adam,
#     grad_func=grad_func_adam, 
#     test_size=100
# )

#test_optimizer(
#    native_optimizer=Adadelta(), 
#    alpha_func=alpha_func_adam, 
#    beta_func=beta_func_adam, 
#    sigma_func=sigma_func_adam,
#    grad_func=grad_func_adam, 
#    test_size=100
#)
