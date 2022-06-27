import tensorflow as tf
from tensorflow import keras
from tensorflow.python.training import training_ops
import torch
from torch.optim import _functional as F
from torch.optim import Optimizer

class CustomOptimizerTorch(Optimizer):
    def __init__(self, params, lr=0.01,
            alpha_func=None,
            beta_func=None,
            sigma_func=None,
            grad_func=None,
            **kwargs):

        self.alpha_func = alpha_func
        self.beta_func = beta_func
        self.sigma_func = sigma_func
        self.grad_func = grad_func

        defaults = dict(lr=lr)
        super(CustomOptimizerTorch, self).__init__(params, defaults)
        for group in self.param_groups:
            group['alpha'] = []
            group['beta'] = []
            group['sigma'] = []
            for p in group['params']:
                group['alpha'].append(torch.zeros_like(p.data))
                group['beta'].append(torch.zeros_like(p.data))
                group['sigma'].append(torch.zeros_like(p.data))
    
    @torch.no_grad()
    def step(self, closure=None):
            """Performs a single optimization step.

            Args:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
            """
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                params_with_grad = [] #weights
                d_p_list = [] #gradients
                momentum_buffer_list = []
                lr = group['lr']

                for p, alpha, beta, sigma in zip(group['params'], group['alpha'], group['beta'], group['sigma']):
                    if p.grad is not None:
                        params_with_grad.append(p)
                        d_p_list.append(p.grad)

                        state = self.state[p]
                        alpha = torch.add(alpha, self.alpha_func(p.grad, alpha))
                        beta = torch.add(beta, self.beta_func(p.grad, alpha, beta))
                        sigma = torch.add(sigma, self.sigma_func(p.grad, alpha, beta, sigma))
                        p.add_(self.grad_func(p.grad, alpha, beta, sigma), alpha=1.0)

            return loss
            
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

