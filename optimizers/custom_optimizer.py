import tensorflow as tf
from tensorflow import keras
import torch
import numpy as np
from utils.smart_phenotype import readable_phenotype, get_optimizer_type

class CustomOptimizer(keras.optimizers.Optimizer):
    def __init__(self,
                            name="CustomOptimizer",
                            phen=None,
                            model=None,
                            **kwargs):

        super(CustomOptimizer, self).__init__(name, **kwargs)
        self.optimizer_type = get_optimizer_type(phen)

        if phen == None:
            raise Exception("Phenotype is None")
        if model == None:
            raise Exception("Model is None")

        # Sometimes "model" is not a tensorflow model but a set of variables.
        # In that case we need a different init procedure.
        # We can check this by checking if the model has layers, if it doesn't we assume it is a set of variables.
        if hasattr(model, 'layers'):  
            # This is a tensorflow model, we can initialize the variables in the normal way
            self._init_all_optimizer_variables_for_tf_model(phen, model)
        else:
            self._init_all_optimizer_variables_for_non_model(phen, model)

        
        exec_env = {"tf": tf}
        exec(phen, exec_env)
        
        self._alpha_func = exec_env["alpha_func"] if self._variables_used['alpha'] else None
        self._beta_func = exec_env["beta_func"] if self._variables_used['beta'] else None
        self._sigma_func = exec_env["sigma_func"] if self._variables_used['sigma'] else None
        self._grad_func = exec_env["grad_func"]
        self.training_ops = None

    def _get_variables_used(self, phen):
        readable_phen, alpha_phen, beta_phen, sigma_phen, grad_phen = readable_phenotype(phen, full_return=True)
        #This is an ugly way to make sure we account for cascading dependencies, we should turn this into a recursive function instead
        for key in self._variables_used.keys():
            if key in grad_phen:
                self._variables_used[key] = True
                if key == 'alpha':
                    for second_key in self._variables_used.keys():
                        if second_key in alpha_phen:
                            self._variables_used[second_key] = True
                elif key == 'beta':
                    for second_key in self._variables_used.keys():
                        if second_key in beta_phen:
                            self._variables_used[second_key] = True
                elif key == 'sigma':
                    for second_key in self._variables_used.keys():
                        if second_key in sigma_phen:
                            self._variables_used[second_key] = True

        for func_key, func_phen in zip(['alpha', 'beta', 'sigma'], [alpha_phen, beta_phen, sigma_phen]):
            if self._variables_used[func_key]:
                for key in self._variables_used.keys():
                    if key in func_phen:
                        self._variables_used[key] = True

        for func_key, func_phen in zip(['alpha', 'beta', 'sigma'], [alpha_phen, beta_phen, sigma_phen]):
            if self._variables_used[func_key]:
                for key in self._variables_used.keys():
                    if key in func_phen:
                        self._variables_used[key] = True

    def _init_optimizer_variable(self, variable_name, variable_dict, trainable_weight, constant_value=None):
        if self._variables_used[variable_name]:
            if constant_value != None:
                variable_dict[trainable_weight.name] = tf.constant(constant_value, shape=trainable_weight.shape, dtype=tf.float32) if self._variables_used[variable_name] else None
            else:
                variable_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name=variable_name + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
        else:
            variable_dict[trainable_weight.name] = None

    def _init_all_optimizer_variables_for_tf_model(self, phen, model):
        self._variables_used = {
            'layer_count': False,
            'layer_num': False,
            'alpha': False,
            'beta': False,
            'sigma': False,
            'strides': False,
            'kernel_size': False,
            'filters': False,
            'dilation_rate': False,
            'units': False,
            'pool_size': False,
            'momentum': False,
            'variance': False,
            'layer_wise_lr': False,
        }

        self._alpha_dict = {}
        self._beta_dict = {}
        self._sigma_dict = {}
        self._depth_dict = {}
        self._layer_count = {}

        self._momentum = {}
        self._variance = {}
        self._layer_wise_lr = {}
            
        self._strides = {}
        self._kernel = {}
        self._filters = {}
        self._dilation_rate = {}

        self._pool_size = {}

        self._units = {}
        
        depth = 0

        self._get_variables_used(phen)

        for layer in model.layers:
            for trainable_weight in layer._trainable_weights:
                # Auxiliary variables
                self._init_optimizer_variable('alpha', self._alpha_dict, trainable_weight)
                self._init_optimizer_variable('beta', self._beta_dict, trainable_weight)
                self._init_optimizer_variable('sigma', self._sigma_dict, trainable_weight)

                # Depth
                self._init_optimizer_variable('layer_num', self._depth_dict, trainable_weight, constant_value=depth)

                # Convolutional variables   
                self._init_optimizer_variable('strides', self._strides, trainable_weight, layer.strides[0] if hasattr(layer, 'strides') else 0.0)
                self._init_optimizer_variable('kernel_size', self._kernel, trainable_weight, layer.kernel_size[0] if hasattr(layer, 'kernel_size') else 0.0)
                self._init_optimizer_variable('filters', self._filters, trainable_weight, layer.filters if hasattr(layer, 'filters') else 0.0)
                self._init_optimizer_variable('dilation_rate', self._dilation_rate, trainable_weight, layer.dilation_rate[0] if hasattr(layer, 'dilation_rate') else 0.0)

                # Dense variables
                self._init_optimizer_variable('units', self._units, trainable_weight, layer.units if hasattr(layer, 'units') else 0.0)
                
                # Pooling variables
                self._init_optimizer_variable('pool_size', self._pool_size, trainable_weight, layer.pool_size[0] if hasattr(layer, 'pool_size') else 0.0)

                # Aggregate variables
                self._init_optimizer_variable('momentum', self._momentum, trainable_weight)
                self._init_optimizer_variable('variance', self._variance, trainable_weight)
                self._init_optimizer_variable('layer_wise_lr', self._layer_wise_lr, trainable_weight)

                depth += 1

            for layer in model.layers:
                for trainable_weight in layer._trainable_weights:
                    self._init_optimizer_variable('layer_count', self._layer_count, trainable_weight, constant_value=depth)
    
    def _init_all_optimizer_variables_for_non_model(self, phen, variables):
        # For this case, layer_count and layer_depth are technically not applicable
        # However, we can still set them to be the number of variables and the index of the variable respectively
        # this way optimizers that use these variables can still be applied to non-model variables
        # The other model-specific variables are set to 0, so any architectural behavior is disabled.

        self._variables_used = {
            'layer_count': False,
            'layer_num': False,
            'alpha': False,
            'beta': False,
            'sigma': False,
            'strides': False,
            'kernel_size': False,
            'filters': False,
            'dilation_rate': False,
            'units': False,
            'pool_size': False,
            'momentum': False,
            'variance': False,
            'layer_wise_lr': False,
        }

        self._alpha_dict = {}
        self._beta_dict = {}
        self._sigma_dict = {}
        self._depth_dict = {}
        self._layer_count = {}

        self._momentum = {}
        self._variance = {}
        self._layer_wise_lr = {}
            
        self._strides = {}
        self._kernel = {}
        self._filters = {}
        self._dilation_rate = {}

        self._pool_size = {}

        self._units = {}
        
        self._get_variables_used(phen)

        for i, var in zip(range(len(variables)), variables):
            # Auxiliary variables
            self._init_optimizer_variable('alpha', self._alpha_dict, var)
            self._init_optimizer_variable('beta', self._beta_dict, var)
            self._init_optimizer_variable('sigma', self._sigma_dict, var)

            # Depth
            self._init_optimizer_variable('layer_num', self._depth_dict, var, constant_value=i)
            self._init_optimizer_variable('layer_count', self._layer_count, var, constant_value=len(variables))

             # Convolutional variables   
            self._init_optimizer_variable('strides', self._strides, var, 0.0)
            self._init_optimizer_variable('kernel_size', self._kernel, var, 0.0)
            self._init_optimizer_variable('filters', self._filters, var, 0.0)
            self._init_optimizer_variable('dilation_rate', self._dilation_rate, var, 0.0)

            # Dense variables
            self._init_optimizer_variable('units', self._units, var, 0.0)
            
            # Pooling variables
            self._init_optimizer_variable('pool_size', self._pool_size, var, 0.0)

            # Aggregate variables
            self._init_optimizer_variable('momentum', self._momentum, var)
            self._init_optimizer_variable('variance', self._variance, var)

            # While most aggregates work, this one is especially model dependent so it must also be set to 0
            self._init_optimizer_variable('layer_wise_lr', self._layer_wise_lr, var, constant_value=0.0)


    def check_slots(self):
        return self._alpha_dict == None and self._beta_dict == None and self._sigma_dict == None

    def init_variables(self, var_list):
        import numpy as np
        create_alpha_flag = self._alpha_dict == None
        create_beta_flag = self._beta_dict == None
        create_sigma_flag = self._sigma_dict == None

        #If var dict has not been created create and empty dict
        self._alpha_dict = {} if create_alpha_flag else self._alpha_dict
        self._beta_dict = {} if create_beta_flag else self._beta_dict
        self._sigma_dict = {} if create_sigma_flag else self._sigma_dict

        for var in var_list:
            if create_alpha_flag:
                self._alpha_dict[var.name] = tf.Variable(np.zeros(var.shape), name="alpha" + var.name[:-2], shape=var.shape, dtype=tf.float32)
            if create_beta_flag:
                self._beta_dict[var.name] = tf.Variable(np.zeros(var.shape), name="beta" + var.name[:-2], shape=var.shape, dtype=tf.float32)
            if create_sigma_flag:
                self._sigma_dict[var.name] = tf.Variable(np.zeros(var.shape), name="sigma" + var.name[:-2], shape=var.shape, dtype=tf.float32)
                
    def _create_slots(self, var_list):
        pass

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(CustomOptimizer, self)._prepare_local(var_device, var_dtype, apply_state)

    def _get_optimizer_type_parameters(self, grad, var, variable_name):
        if self.optimizer_type == 'deep_architecture_optimizer_with_aggregators':
            parameters = [
                    self._momentum[variable_name],
                    self._variance[variable_name],
                    self._layer_wise_lr[variable_name],
                    self._strides[variable_name],
                    self._kernel[variable_name],
                    self._filters[variable_name],
                    self._dilation_rate[variable_name],
                    self._units[variable_name],
                    self._pool_size[variable_name],
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape, 
                    self._alpha_dict[variable_name], 
                    grad
            ]
        elif self.optimizer_type == 'deep_architecture_optimizer':
            parameters = [
                    self._strides[variable_name],
                    self._kernel[variable_name],
                    self._filters[variable_name],
                    self._dilation_rate[variable_name],
                    self._units[variable_name],
                    self._pool_size[variable_name],
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape, 
                    self._alpha_dict[variable_name], 
                    grad
            ]
        elif self.optimizer_type == 'basic_architecture_optimizer':
            parameters = [
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape, 
                    self._alpha_dict[variable_name], 
                    grad
            ]        
        elif self.optimizer_type == 'basic_optimizer':
            parameters = [
                    var.shape, 
                    self._alpha_dict[variable_name], 
                    grad
            ]              
        else:
            raise Exception(f"Optimizer type {self.optimizer_type} is deprectated, either adapt it to be deep_architecture_optimizer or find older code to run it.")
        return parameters

    def _resource_apply_dense(self, grad, var, apply_state=None):
        if self.training_ops == None:
            from tensorflow.python.training import training_ops
            self.training_ops = training_ops
        #print("_resource_apply_dense")
        variable_name = var.name
        #print(f"#: {variable_name}")

        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                                        or self._fallback_apply_state(var_device, var_dtype))
        
        # Each optimizer type takes different parameters in their auxiliary functions so let's retrieve them
        parameters = self._get_optimizer_type_parameters(grad, var, variable_name)

        # For optimizers with aggregators, we need to run the aggregators before the rest of the procedure
        if self._variables_used['momentum']:
            self.training_ops.resource_apply_gradient_descent( 
                self._momentum[variable_name].handle,
                tf.constant(1.0),
                momentum_function(grad, self._momentum[variable_name], self.momentum_const))
        if self._variables_used['variance']:
            self.training_ops.resource_apply_gradient_descent( 
                self._variance[variable_name].handle,
                tf.constant(1.0),
                variance_function(grad, self._variance[variable_name], self.variance_const))
        if self._variables_used['layer_wise_lr']:
            self.training_ops.resource_apply_gradient_descent( 
                self._layer_wise_lr[variable_name].handle,
                tf.constant(1.0),
                layer_wise_lr_function(grad, self._layer_wise_lr[variable_name], var, self.layer_wise_lr_const))

        if self._alpha_func != None:
            # We unroll the parameters for alpha function
            self.training_ops.resource_apply_gradient_descent(
                        self._alpha_dict[variable_name].handle, 
                        tf.constant(1.0), 
                        self._alpha_func(
                            *parameters
                        ), 
                        use_locking=self._use_locking)

        # In the remaining functions it gets a bit hacky
        # We need to add function specific parameters to the parameter list so we just place it in the correct index
        # These function specific parameters are always present at the end before the gradient, regardless of optimizer type
        if self._beta_func != None:
            beta_parameters = parameters[:-1] + [self._beta_dict[variable_name]] + [parameters[-1]]
            self.training_ops.resource_apply_gradient_descent(
                            self._beta_dict[variable_name].handle, 
                            tf.constant(1.0), 
                            self._beta_func(
                                *beta_parameters
                            ), 
                            use_locking=self._use_locking)

        if self._sigma_func!= None:
            sigma_parameters = parameters[:-1] + [self._beta_dict[variable_name], self._sigma_dict[variable_name]] + [parameters[-1]]
            self.training_ops.resource_apply_gradient_descent(
                                self._sigma_dict[variable_name].handle, 
                                tf.constant(1.0), 
                                self._sigma_func(
                                    *sigma_parameters
                                ), use_locking=self._use_locking)

        weight_parameters = parameters[:-1] + [self._beta_dict[variable_name], self._sigma_dict[variable_name]] + [parameters[-1]]
        updated_weights = self.training_ops.resource_apply_gradient_descent(
                var.handle, 
                tf.constant(1.0), 
                self._grad_func(
                    *weight_parameters
                ), 
                use_locking=self._use_locking)

        return updated_weights
    # More recent tensorflow versions require the update step to be defined separately
    def update_step(self, gradient, variable, learning_rate):
        # TODO
        pass

class CustomOptimizerArch(keras.optimizers.Optimizer):
    def __init__(self,
                            name="CustomOptimizerArch",
                            phen=None,
                            model=None,
                            grad_func=None,
                            alpha=None,
                            alpha_func=None,
                            beta=None,
                            beta_func=None,
                            sigma=None,
                            sigma_func=None,
                            vars=None,
                            **kwargs):

        super().__init__(name, **kwargs)
        self.phen = phen
        self._learning_rate = 1.0
        self._name = name

        if phen == None:
            self._alpha_func = alpha_func
            self._beta_func = beta_func
            self._sigma_func = sigma_func
            self._grad_func = grad_func
        else:
            exec_env = {"tf": tf}
            exec(phen, exec_env)
            self._alpha_func = exec_env["alpha_func"]
            self._beta_func = exec_env["beta_func"]
            self._sigma_func = exec_env["sigma_func"]
            self._grad_func = exec_env["grad_func"]

        if alpha != None:
            print("Loading Alpha ", alpha)
            self._alpha_dict = alpha
            self._beta_dict = beta
            self._sigma_dict = sigma
        else:
            self._alpha_dict = {}
            self._beta_dict = {}
            self._sigma_dict = {}
            self._depth_dict = {}
            self._layer_count = {}
            depth = 0

            if model != None:
                for layer in model.layers:
                    for trainable_weight in layer._trainable_weights:
                        #print(trainable_weight.name)
                        self._depth_dict[trainable_weight.name] = tf.constant(depth, shape=trainable_weight.shape, dtype=tf.float32)
                        self._alpha_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="alpha" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
                        self._beta_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="beta" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
                        self._sigma_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="sigma" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
                        depth += 1
                for layer in model.layers:
                    for trainable_weight in layer._trainable_weights:
                        #print(trainable_weight.name)
                        self._layer_count[trainable_weight.name] = tf.constant(depth, shape=trainable_weight.shape, dtype=tf.float32)
            elif vars != None:
                for var in vars:
                    self._depth_dict[var.name] = tf.constant(depth, shape=var.shape, dtype=tf.float32)
                    self._alpha_dict[var.name] = tf.Variable(np.zeros(var.shape) , name="alpha" + var.name[:-2], shape=var.shape, dtype=tf.float32)
                    self._beta_dict[var.name] = tf.Variable(np.zeros(var.shape) , name="beta" + var.name[:-2], shape=var.shape, dtype=tf.float32)
                    self._sigma_dict[var.name] = tf.Variable(np.zeros(var.shape) , name="sigma" + var.name[:-2], shape=var.shape, dtype=tf.float32)
                    depth += 1
                for var in vars:
                    #print(trainable_weight.name)
                    self._layer_count[var.name] = tf.constant(depth, shape=var.shape, dtype=tf.float32)
            else:
                raise Exception("Nothing to optimize")

    def check_slots(self):
        return self._alpha_dict == None and self._beta_dict == None and self._sigma_dict == None

    def init_variables(self, var_list):
        import numpy as np
        create_alpha_flag = self._alpha_dict == None
        create_beta_flag = self._beta_dict == None
        create_sigma_flag = self._sigma_dict == None

        #If var dict has not been created create and empty dict
        self._alpha_dict = {} if create_alpha_flag else self._alpha_dict
        self._beta_dict = {} if create_beta_flag else self._beta_dict
        self._sigma_dict = {} if create_sigma_flag else self._sigma_dict

        for var in var_list:
            if create_alpha_flag:
                self._alpha_dict[var.name] = tf.Variable(np.zeros(var.shape), name="alpha" + var.name[:-2], shape=var.shape, dtype=tf.float32)
            if create_beta_flag:
                self._beta_dict[var.name] = tf.Variable(np.zeros(var.shape), name="beta" + var.name[:-2], shape=var.shape, dtype=tf.float32)
            if create_sigma_flag:
                self._sigma_dict[var.name] = tf.Variable(np.zeros(var.shape), name="sigma" + var.name[:-2], shape=var.shape, dtype=tf.float32)
                
    def _create_slots(self, var_list):
        pass

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(CustomOptimizerArch, self)._prepare_local(var_device, var_dtype, apply_state)


    def _resource_apply_dense(self, grad, var, apply_state=None):
        #print(self.phen)
        #print("_resource_apply_dense")
        variable_name = var.name
        #print(self._layer_count[variable_name])
        #print(self._depth_dict[variable_name])
        if variable_name not in self._alpha_dict:
            self._alpha_dict[var.name] = tf.Variable(np.zeros(var.shape) , name="alpha" + var.name[:-2], shape=var.shape, dtype=tf.float32)
            self._beta_dict[var.name] = tf.Variable(np.zeros(var.shape) , name="beta" + var.name[:-2], shape=var.shape, dtype=tf.float32)
            self._sigma_dict[var.name] = tf.Variable(np.zeros(var.shape) , name="sigma" + var.name[:-2], shape=var.shape, dtype=tf.float32)
            
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                                        or self._fallback_apply_state(var_device, var_dtype))
        
        if self._alpha_func != None:
            training_ops.resource_apply_gradient_descent(
                self._alpha_dict[variable_name].handle, 
                tf.constant(1.0), 
                self._alpha_func(
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape, 
                    self._alpha_dict[variable_name], 
                    grad), use_locking=self._use_locking)
        if self._beta_func != None:
            training_ops.resource_apply_gradient_descent(
                self._beta_dict[variable_name].handle, 
                tf.constant(1.0), 
                self._beta_func(
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape, 
                    self._alpha_dict[variable_name], 
                    self._beta_dict[variable_name], 
                    grad), use_locking=self._use_locking)
        if self._sigma_func!= None:
            training_ops.resource_apply_gradient_descent(
                self._sigma_dict[variable_name].handle, 
                tf.constant(1.0), 
                self._sigma_func(
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape, 
                    self._alpha_dict[variable_name], 
                    self._beta_dict[variable_name], 
                    self._sigma_dict[variable_name], 
                    grad), use_locking=self._use_locking)
            
        foo = training_ops.resource_apply_gradient_descent(
                var.handle, 
                tf.constant(1.0), 
                self._grad_func(
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape, 
                    self._alpha_dict[variable_name], 
                    self._beta_dict[variable_name], 
                    self._sigma_dict[variable_name], 
                    grad), use_locking=self._use_locking)
        return foo

    def update_step(self, grad, var):
        #print(self.phen)
        #print("_resource_apply_dense")
        variable_name = var.name
        #print(variable_name)
        if variable_name not in self._alpha_dict:
            self._alpha_dict[var.name] = tf.Variable(np.zeros(var.shape) , name="alpha" + var.name[:-2], shape=var.shape, dtype=tf.float32)
            self._beta_dict[var.name] = tf.Variable(np.zeros(var.shape) , name="beta" + var.name[:-2], shape=var.shape, dtype=tf.float32)
            self._sigma_dict[var.name] = tf.Variable(np.zeros(var.shape) , name="sigma" + var.name[:-2], shape=var.shape, dtype=tf.float32)

        if self._alpha_func != None:
            training_ops.resource_apply_gradient_descent(
                self._alpha_dict[variable_name].handle,
                tf.constant(1.0),
                self._alpha_func(
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape,
                    self._alpha_dict[variable_name],
                    grad))
        if self._beta_func != None:
            training_ops.resource_apply_gradient_descent(
                self._beta_dict[variable_name].handle,
                tf.constant(1.0),
                self._beta_func(
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape,
                    self._alpha_dict[variable_name],
                    self._beta_dict[variable_name],
                    grad))
        if self._sigma_func!= None:
            training_ops.resource_apply_gradient_descent(
                self._sigma_dict[variable_name].handle,
                tf.constant(1.0),
                self._sigma_func(
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape,
                    self._alpha_dict[variable_name],
                    self._beta_dict[variable_name],
                    self._sigma_dict[variable_name],
                    grad))

        foo = training_ops.resource_apply_gradient_descent(
                var.handle,
                tf.constant(1.0),
                self._grad_func(
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape,
                    self._alpha_dict[variable_name],
                    self._beta_dict[variable_name],
                    self._sigma_dict[variable_name],
                    grad))
        return foo

class CustomOptimizerTorch(torch.optim.Optimizer):
    def __init__(self,
                            params, 
                            lr=0.01,
                            name="CustomOptimizer",
                            phen=None,
                            grad_func=None,
                            alpha_func=None,
                            beta_func=None,
                            sigma_func=None,
                            device=None,
                            **kwargs):


        defaults = dict(lr=lr)
        super(CustomOptimizerTorch, self).__init__(params, defaults)
        self.device = device
        for group in self.param_groups:
            group['alpha'] = []
            group['beta'] = []
            group['sigma'] = []
            for p in group['params']:
                group['alpha'].append(torch.zeros_like(p.data, device=device))
                group['beta'].append(torch.zeros_like(p.data, device=device))
                group['sigma'].append(torch.zeros_like(p.data, device=device))
        if phen == None:
            self.alpha_func = alpha_func
            self.beta_func = beta_func
            self.sigma_func = sigma_func
            self.grad_func = grad_func
        else:
            exec_env = {"torch": torch}
            #print(phen)
            exec(phen, exec_env)
            self.alpha_func = exec_env["alpha_func"]
            self.beta_func = exec_env["beta_func"]
            self.sigma_func = exec_env["sigma_func"]
            self.grad_func = exec_env["grad_func"]

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
                        if str(p.grad.device) != 'cuda:0' or str(alpha.device) != 'cuda:0' or str(beta.device) != 'cuda:0' or str(sigma.device) != 'cuda:0':
                            print(p.grad.device, alpha.device, beta.device, sigma.device)
                        p.grad = p.grad.to(self.device)
                        params_with_grad.append(p)
                        d_p_list.append(p.grad)

                        state = self.state[p]
                        alpha = torch.add(alpha, self.alpha_func(p.size(), p.grad, alpha, self.device))
                        beta = torch.add(beta, self.beta_func(p.size(), p.grad, alpha, beta, self.device))
                        sigma = torch.add(sigma, self.sigma_func(p.size(), p.grad, alpha, beta, sigma, self.device))
                        p.add_(self.grad_func(p.size(), p.grad, alpha, beta, sigma, self.device), alpha=1.0)
            return loss
            
class CustomOptimizerLayerVar(keras.optimizers.Optimizer):
    def __init__(self,
                            name="CustomOptimizerLayerVar",
                            phen=None,
                            model=None,
                            grad_func=None,
                            alpha=None,
                            alpha_func=None,
                            beta=None,
                            beta_func=None,
                            sigma=None,
                            sigma_func=None,
                            vars=None,
                            **kwargs):

        super(CustomOptimizerLayerVar, self).__init__(name, **kwargs)
        self.phen = phen
        self._learning_rate = 1.0
        self._name = name

        if phen == None:
            self._alpha_func = alpha_func
            self._beta_func = beta_func
            self._sigma_func = sigma_func
            self._grad_func = grad_func
        else:
            exec_env = {"tf": tf}
            exec(phen, exec_env)
            #print(phen)
            self._alpha_func = exec_env["alpha_func"]
            self._beta_func = exec_env["beta_func"]
            self._sigma_func = exec_env["sigma_func"]
            self._grad_func = exec_env["grad_func"]

        self._variables_used = {
            'layer_count': False,
            'layer_num': False,
            'alpha': False,
            'beta': False,
            'sigma': False,
            'strides': False,
            'kernel_size': False,
            'filters': False,
            'dilation_rate': False,
            'units': False,
            'pool_size': False,
        }
        
        
        readable_phen, alpha_phen, beta_phen, sigma_phen, grad_phen = readable_phenotype(phen, full_return=True)

        #This is an ugly way to make sure we account for cascading dependencies, we should turn this into a recursive function instead
        for key in self._variables_used.keys():
            if key in grad_phen:
                self._variables_used[key] = True
                if key == 'alpha':
                    for second_key in self._variables_used.keys():
                        if second_key in alpha_phen:
                            self._variables_used[second_key] = True
                elif key == 'beta':
                    for second_key in self._variables_used.keys():
                        if second_key in beta_phen:
                            self._variables_used[second_key] = True
                elif key == 'sigma':
                    for second_key in self._variables_used.keys():
                        if second_key in sigma_phen:
                            self._variables_used[second_key] = True
        for func_key, func_phen in zip(['alpha', 'beta', 'sigma'], [alpha_phen, beta_phen, sigma_phen]):
            if self._variables_used[func_key]:
                for key in self._variables_used.keys():
                    if key in func_phen:
                        self._variables_used[key] = True
        for func_key, func_phen in zip(['alpha', 'beta', 'sigma'], [alpha_phen, beta_phen, sigma_phen]):
            if self._variables_used[func_key]:
                for key in self._variables_used.keys():
                    if key in func_phen:
                        self._variables_used[key] = True


        print("Variables used in optimizer: ", self._variables_used)
        if True:
            self._alpha_dict = {}
            self._beta_dict = {}
            self._sigma_dict = {}
            self._depth_dict = {}
            self._layer_count = {}

            self._strides = {}
            self._kernel = {}
            self._filters = {}
            self._dilation_rate = {}

            self._pool_size = {}

            self._units = {}
            
            depth = 0

            if model != None:
                    for layer in model.layers:
                        for trainable_weight in layer._trainable_weights:
                            
                            self._depth_dict[trainable_weight.name] = tf.constant(depth, shape=trainable_weight.shape, dtype=tf.float32) if self._variables_used['layer_num'] else None
                            self._alpha_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="alpha" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32) if self._variables_used['alpha'] else None
                            self._beta_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="beta" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32) if self._variables_used['beta'] else None
                            self._sigma_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="sigma" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32) if self._variables_used['sigma'] else None

                            if self._variables_used['strides']:
                                if hasattr(layer, 'strides'):
                                    print(f'strides {layer.strides[0]}')
                                    self._strides[trainable_weight.name] = tf.constant(layer.strides[0], shape=trainable_weight.shape, dtype=tf.float32)
                                else: 
                                    self._strides[trainable_weight.name] = tf.constant(0.0, shape=trainable_weight.shape, dtype=tf.float32)
                            else:
                                self._strides[trainable_weight.name] = None

                            if self._variables_used['kernel_size']:
                                if hasattr(layer, 'kernel_size'):
                                    print(f'kernel_size {layer.kernel_size[0]}')
                                    self._kernel[trainable_weight.name] = tf.constant(layer.kernel_size[0], shape=trainable_weight.shape, dtype=tf.float32)
                                else:
                                    self._kernel[trainable_weight.name] = tf.constant(0.0, shape=trainable_weight.shape, dtype=tf.float32)
                            else:
                                self._kernel[trainable_weight.name] = None

                            if self._variables_used['filters']:
                                if hasattr(layer, 'filters'):    
                                    print(f'filters {layer.filters}')
                                    self._filters[trainable_weight.name] = tf.constant(layer.filters, shape=trainable_weight.shape, dtype=tf.float32)
                                else:
                                    self._filters[trainable_weight.name] = tf.constant(0.0, shape=trainable_weight.shape, dtype=tf.float32)
                            else:
                                self._filters[trainable_weight.name] = None

                            if self._variables_used['dilation_rate']:
                                if hasattr(layer, 'dilation_rate'):
                                    print(f'dilation_rate {layer.dilation_rate[0]}')
                                    self._dilation_rate[trainable_weight.name] = tf.constant(layer.dilation_rate[0], shape=trainable_weight.shape, dtype=tf.float32)
                                else:
                                    self._dilation_rate[trainable_weight.name] = tf.constant(0.0, shape=trainable_weight.shape, dtype=tf.float32)
                            else:
                                self._dilation_rate[trainable_weight.name] = None

                            if self._variables_used['units']:
                                if hasattr(layer, 'units'):
                                    print(f'units {layer.units}')
                                    self._units[trainable_weight.name] = tf.constant(layer.units, shape=trainable_weight.shape, dtype=tf.float32)
                                else:
                                    self._units[trainable_weight.name] = tf.constant(0.0, shape=trainable_weight.shape, dtype=tf.float32) 
                            else:
                                self._units[trainable_weight.name] = None

                            if self._variables_used['pool_size']:
                                if hasattr(layer, 'pool_size'):
                                    print(f'pool_size {layer.pool_size[0]}')
                                    self._pool_size[trainable_weight.name] = tf.constant(layer.pool_size[0], shape=trainable_weight.shape, dtype=tf.float32)
                                else:
                                    self._pool_size[trainable_weight.name] = tf.constant(0.0, shape=trainable_weight.shape, dtype=tf.float32) 
                            else:
                                self._pool_size[trainable_weight.name] = None

                            depth += 1

                        for layer in model.layers:
                            for trainable_weight in layer._trainable_weights:
                                #print(trainable_weight.name)
                                if self._variables_used['layer_count']:
                                    self._layer_count[trainable_weight.name] = tf.constant(depth, shape=trainable_weight.shape, dtype=tf.float32)
                                else:
                                    self._layer_count[trainable_weight.name] = None
            elif vars != None:
                for var in vars:
                    self._depth_dict[var.name] = tf.constant(depth, shape=var.shape, dtype=tf.float32)
                    self._alpha_dict[var.name] = tf.Variable(np.zeros(var.shape) , name="alpha" + var.name[:-2], shape=var.shape, dtype=tf.float32)
                    self._beta_dict[var.name] = tf.Variable(np.zeros(var.shape) , name="beta" + var.name[:-2], shape=var.shape, dtype=tf.float32)
                    self._sigma_dict[var.name] = tf.Variable(np.zeros(var.shape) , name="sigma" + var.name[:-2], shape=var.shape, dtype=tf.float32)
                    depth += 1
                for var in vars:
                    #print(trainable_weight.name)
                    self._layer_count[var.name] = tf.constant(depth, shape=var.shape, dtype=tf.float32)
            else:
                raise Exception("Nothing to optimize")

    def init_variables(self, var_list):
        import numpy as np
        create_alpha_flag = self._alpha_dict == None
        create_beta_flag = self._beta_dict == None
        create_sigma_flag = self._sigma_dict == None

        #If var dict has not been created create and empty dict
        self._alpha_dict = {} if create_alpha_flag else self._alpha_dict
        self._beta_dict = {} if create_beta_flag else self._beta_dict
        self._sigma_dict = {} if create_sigma_flag else self._sigma_dict

        for var in var_list:
            if create_alpha_flag:
                self._alpha_dict[var.name] = tf.Variable(np.zeros(var.shape), name="alpha" + var.name[:-2], shape=var.shape, dtype=tf.float32)
            if create_beta_flag:
                self._beta_dict[var.name] = tf.Variable(np.zeros(var.shape), name="beta" + var.name[:-2], shape=var.shape, dtype=tf.float32)
            if create_sigma_flag:
                self._sigma_dict[var.name] = tf.Variable(np.zeros(var.shape), name="sigma" + var.name[:-2], shape=var.shape, dtype=tf.float32)                

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(CustomOptimizerLayerVar, self)._prepare_local(var_device, var_dtype, apply_state)


    def _resource_apply_dense(self, grad, var, apply_state=None):
        variable_name = var.name


        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                                        or self._fallback_apply_state(var_device, var_dtype))


        if self._alpha_func != None and self._variables_used['alpha']:
            training_ops.resource_apply_gradient_descent(
                self._alpha_dict[variable_name].handle, 
                tf.constant(1.0), 
                self._alpha_func(
                    self._strides[variable_name],
                    self._kernel[variable_name],
                    self._filters[variable_name],
                    self._dilation_rate[variable_name],
                    self._units[variable_name],
                    self._pool_size[variable_name],
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape, 
                    self._alpha_dict[variable_name], 
                    grad), use_locking=self._use_locking)
            
        if self._beta_func != None and self._variables_used['beta']:
            training_ops.resource_apply_gradient_descent(
                self._beta_dict[variable_name].handle, 
                tf.constant(1.0), 
                self._beta_func(
                    self._strides[variable_name],
                    self._kernel[variable_name],
                    self._filters[variable_name],
                    self._dilation_rate[variable_name],
                    self._units[variable_name],
                    self._pool_size[variable_name],
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape, 
                    self._alpha_dict[variable_name], 
                    self._beta_dict[variable_name], 
                    grad), use_locking=self._use_locking)
                
        if self._sigma_func!= None and self._variables_used['sigma']:
            training_ops.resource_apply_gradient_descent(
                self._sigma_dict[variable_name].handle, 
                tf.constant(1.0), 
                self._sigma_func(
                    self._strides[variable_name],
                    self._kernel[variable_name],
                    self._filters[variable_name],
                    self._dilation_rate[variable_name],
                    self._units[variable_name],
                    self._pool_size[variable_name],
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape, 
                    self._alpha_dict[variable_name], 
                    self._beta_dict[variable_name], 
                    self._sigma_dict[variable_name], 
                    grad), use_locking=self._use_locking)
            
        foo = training_ops.resource_apply_gradient_descent(
                var.handle, 
                tf.constant(1.0), 
                self._grad_func(
                    self._strides[variable_name],
                    self._kernel[variable_name],
                    self._filters[variable_name],
                    self._dilation_rate[variable_name],
                    self._units[variable_name],
                    self._pool_size[variable_name],
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape, 
                    self._alpha_dict[variable_name], 
                    self._beta_dict[variable_name], 
                    self._sigma_dict[variable_name], 
                    grad), use_locking=self._use_locking)
        return foo

    def update_step(self, grad, var):
        #print(self.phen)
        #print("_resource_apply_dense")
        variable_name = var.name
        #print(variable_name)
        if variable_name not in self._alpha_dict:
            self._alpha_dict[var.name] = tf.Variable(np.zeros(var.shape) , name="alpha" + var.name[:-2], shape=var.shape, dtype=tf.float32)
            self._beta_dict[var.name] = tf.Variable(np.zeros(var.shape) , name="beta" + var.name[:-2], shape=var.shape, dtype=tf.float32)
            self._sigma_dict[var.name] = tf.Variable(np.zeros(var.shape) , name="sigma" + var.name[:-2], shape=var.shape, dtype=tf.float32)

        if self._alpha_func != None:
            training_ops.resource_apply_gradient_descent(
                self._alpha_dict[variable_name].handle,
                tf.constant(1.0),
                self._alpha_func(
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape,
                    self._alpha_dict[variable_name],
                    grad))
        if self._beta_func != None:
            training_ops.resource_apply_gradient_descent(
                self._beta_dict[variable_name].handle,
                tf.constant(1.0),
                self._beta_func(
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape,
                    self._alpha_dict[variable_name],
                    self._beta_dict[variable_name],
                    grad))
        if self._sigma_func!= None:
            training_ops.resource_apply_gradient_descent(
                self._sigma_dict[variable_name].handle,
                tf.constant(1.0),
                self._sigma_func(
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape,
                    self._alpha_dict[variable_name],
                    self._beta_dict[variable_name],
                    self._sigma_dict[variable_name],
                    grad))

        foo = training_ops.resource_apply_gradient_descent(
                var.handle,
                tf.constant(1.0),
                self._grad_func(
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape,
                    self._alpha_dict[variable_name],
                    self._beta_dict[variable_name],
                    self._sigma_dict[variable_name],
                    grad))
        return foo
class CustomOptimizerAggregates(keras.optimizers.Optimizer):
    def __init__(self,
                            name="CustomOptimizerAggregates",
                            phen=None,
                            model=None,
                            grad_func=None,
                            alpha=None,
                            alpha_func=None,
                            beta=None,
                            beta_func=None,
                            sigma=None,
                            sigma_func=None,
                            vars=None,
                            **kwargs):

        super(CustomOptimizerAggregates, self).__init__(name, **kwargs)
        self.phen = phen
        self._learning_rate = 1.0
        self._name = name

        if phen == None:
            self._alpha_func = alpha_func
            self._beta_func = beta_func
            self._sigma_func = sigma_func
            self._grad_func = grad_func
        else:
            exec_env = {"tf": tf}
            exec(phen, exec_env)
            #print(phen)
            self._alpha_func = exec_env["alpha_func"]
            self._beta_func = exec_env["beta_func"]
            self._sigma_func = exec_env["sigma_func"]
            self._grad_func = exec_env["grad_func"]
            self.momentum_const = exec_env["momentum_const"]
            self.variance_const = exec_env["variance_const"]
            self.layer_wise_lr_const = exec_env["layer_wise_lr_const"]

        self._variables_used = {
            'layer_count': False,
            'layer_num': False,
            'alpha': False,
            'beta': False,
            'sigma': False,
            'strides': False,
            'kernel_size': False,
            'filters': False,
            'dilation_rate': False,
            'units': False,
            'pool_size': False,
            'momentum': False,
            'variance': False,
            'layer_wise_lr': False,
        }
        
        readable_phen, alpha_phen, beta_phen, sigma_phen, grad_phen = readable_phenotype(phen, full_return=True)

        #This is an ugly way to make sure we account for cascading dependencies, we should turn this into a recursive function instead
        for key in self._variables_used.keys():
            if key in grad_phen:
                self._variables_used[key] = True
                if key == 'alpha':
                    for second_key in self._variables_used.keys():
                        if second_key in alpha_phen:
                            self._variables_used[second_key] = True
                elif key == 'beta':
                    for second_key in self._variables_used.keys():
                        if second_key in beta_phen:
                            self._variables_used[second_key] = True
                elif key == 'sigma':
                    for second_key in self._variables_used.keys():
                        if second_key in sigma_phen:
                            self._variables_used[second_key] = True
        for func_key, func_phen in zip(['alpha', 'beta', 'sigma'], [alpha_phen, beta_phen, sigma_phen]):
            if self._variables_used[func_key]:
                for key in self._variables_used.keys():
                    if key in func_phen:
                        self._variables_used[key] = True
        for func_key, func_phen in zip(['alpha', 'beta', 'sigma'], [alpha_phen, beta_phen, sigma_phen]):
            if self._variables_used[func_key]:
                for key in self._variables_used.keys():
                    if key in func_phen:
                        self._variables_used[key] = True


        print("Variables used in optimizer: ", self._variables_used)
        if True:
            self._alpha_dict = {}
            self._beta_dict = {}
            self._sigma_dict = {}
            self._depth_dict = {}
            self._layer_count = {}
            self.momentum = {}
            self.variance = {}
            self.layer_wise_lr = {}

            self._strides = {}
            self._kernel = {}
            self._filters = {}
            self._dilation_rate = {}

            self._pool_size = {}

            self._units = {}
            
            depth = 0

            if model != None:
                    for layer in model.layers:
                        for trainable_weight in layer._trainable_weights:
                            
                            self._depth_dict[trainable_weight.name] = tf.constant(depth, shape=trainable_weight.shape, dtype=tf.float32) if self._variables_used['layer_num'] else None
                            self._alpha_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="alpha" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32) if self._variables_used['alpha'] else None
                            self._beta_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="beta" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32) if self._variables_used['beta'] else None
                            self._sigma_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="sigma" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32) if self._variables_used['sigma'] else None
                            self.momentum[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="momentum" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32) if self._variables_used['momentum'] else None
                            self.variance[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="variance" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32) if self._variables_used['variance'] else None
                            self.layer_wise_lr[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="layer_wise_lr" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32) if self._variables_used['layer_wise_lr'] else None

                            if self._variables_used['strides']:
                                if hasattr(layer, 'strides'):
                                    print(f'strides {layer.strides[0]}')
                                    self._strides[trainable_weight.name] = tf.constant(layer.strides[0], shape=trainable_weight.shape, dtype=tf.float32)
                                else: 
                                    self._strides[trainable_weight.name] = tf.constant(0.0, shape=trainable_weight.shape, dtype=tf.float32)
                            else:
                                self._strides[trainable_weight.name] = None

                            if self._variables_used['kernel_size']:
                                if hasattr(layer, 'kernel_size'):
                                    print(f'kernel_size {layer.kernel_size[0]}')
                                    self._kernel[trainable_weight.name] = tf.constant(layer.kernel_size[0], shape=trainable_weight.shape, dtype=tf.float32)
                                else:
                                    self._kernel[trainable_weight.name] = tf.constant(0.0, shape=trainable_weight.shape, dtype=tf.float32)
                            else:
                                self._kernel[trainable_weight.name] = None

                            if self._variables_used['filters']:
                                if hasattr(layer, 'filters'):    
                                    print(f'filters {layer.filters}')
                                    self._filters[trainable_weight.name] = tf.constant(layer.filters, shape=trainable_weight.shape, dtype=tf.float32)
                                else:
                                    self._filters[trainable_weight.name] = tf.constant(0.0, shape=trainable_weight.shape, dtype=tf.float32)
                            else:
                                self._filters[trainable_weight.name] = None

                            if self._variables_used['dilation_rate']:
                                if hasattr(layer, 'dilation_rate'):
                                    print(f'dilation_rate {layer.dilation_rate[0]}')
                                    self._dilation_rate[trainable_weight.name] = tf.constant(layer.dilation_rate[0], shape=trainable_weight.shape, dtype=tf.float32)
                                else:
                                    self._dilation_rate[trainable_weight.name] = tf.constant(0.0, shape=trainable_weight.shape, dtype=tf.float32)
                            else:
                                self._dilation_rate[trainable_weight.name] = None

                            if self._variables_used['units']:
                                if hasattr(layer, 'units'):
                                    print(f'units {layer.units}')
                                    self._units[trainable_weight.name] = tf.constant(layer.units, shape=trainable_weight.shape, dtype=tf.float32)
                                else:
                                    self._units[trainable_weight.name] = tf.constant(0.0, shape=trainable_weight.shape, dtype=tf.float32) 
                            else:
                                self._units[trainable_weight.name] = None

                            if self._variables_used['pool_size']:
                                if hasattr(layer, 'pool_size'):
                                    print(f'pool_size {layer.pool_size[0]}')
                                    self._pool_size[trainable_weight.name] = tf.constant(layer.pool_size[0], shape=trainable_weight.shape, dtype=tf.float32)
                                else:
                                    self._pool_size[trainable_weight.name] = tf.constant(0.0, shape=trainable_weight.shape, dtype=tf.float32) 
                            else:
                                self._pool_size[trainable_weight.name] = None

                            depth += 1

                        for layer in model.layers:
                            for trainable_weight in layer._trainable_weights:
                                #print(trainable_weight.name)
                                if self._variables_used['layer_count']:
                                    self._layer_count[trainable_weight.name] = tf.constant(depth, shape=trainable_weight.shape, dtype=tf.float32)
                                else:
                                    self._layer_count[trainable_weight.name] = None
            elif vars != None:
                for var in vars:
                    self._depth_dict[var.name] = tf.constant(depth, shape=var.shape, dtype=tf.float32)
                    self._alpha_dict[var.name] = tf.Variable(np.zeros(var.shape) , name="alpha" + var.name[:-2], shape=var.shape, dtype=tf.float32)
                    self._beta_dict[var.name] = tf.Variable(np.zeros(var.shape) , name="beta" + var.name[:-2], shape=var.shape, dtype=tf.float32)
                    self._sigma_dict[var.name] = tf.Variable(np.zeros(var.shape) , name="sigma" + var.name[:-2], shape=var.shape, dtype=tf.float32)
                    depth += 1
                for var in vars:
                    #print(trainable_weight.name)
                    self._layer_count[var.name] = tf.constant(depth, shape=var.shape, dtype=tf.float32)
            else:
                raise Exception("Nothing to optimize")

    def init_variables(self, var_list):
        import numpy as np
        create_alpha_flag = self._alpha_dict == None
        create_beta_flag = self._beta_dict == None
        create_sigma_flag = self._sigma_dict == None

        #If var dict has not been created create and empty dict
        self._alpha_dict = {} if create_alpha_flag else self._alpha_dict
        self._beta_dict = {} if create_beta_flag else self._beta_dict
        self._sigma_dict = {} if create_sigma_flag else self._sigma_dict

        for var in var_list:
            if create_alpha_flag:
                self._alpha_dict[var.name] = tf.Variable(np.zeros(var.shape), name="alpha" + var.name[:-2], shape=var.shape, dtype=tf.float32)
            if create_beta_flag:
                self._beta_dict[var.name] = tf.Variable(np.zeros(var.shape), name="beta" + var.name[:-2], shape=var.shape, dtype=tf.float32)
            if create_sigma_flag:
                self._sigma_dict[var.name] = tf.Variable(np.zeros(var.shape), name="sigma" + var.name[:-2], shape=var.shape, dtype=tf.float32)                

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(CustomOptimizerAggregates, self)._prepare_local(var_device, var_dtype, apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        variable_name = var.name

        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                                        or self._fallback_apply_state(var_device, var_dtype))

        if self._variables_used['momentum']:
            training_ops.resource_apply_gradient_descent( 
                self.momentum[variable_name].handle,
                tf.constant(1.0),
                momentum_function(grad, self.momentum[variable_name], self.momentum_const))
        if self._variables_used['variance']:
            training_ops.resource_apply_gradient_descent( 
                self.variance[variable_name].handle,
                tf.constant(1.0),
                variance_function(grad, self.variance[variable_name], self.variance_const))
        if self._variables_used['layer_wise_lr']:
            training_ops.resource_apply_gradient_descent( 
                self.layer_wise_lr[variable_name].handle,
                tf.constant(1.0),
                layer_wise_lr_function(grad, self.layer_wise_lr[variable_name], var, self.layer_wise_lr_const))
        
        if self._alpha_func != None and self._variables_used['alpha']:
            training_ops.resource_apply_gradient_descent(
                self._alpha_dict[variable_name].handle, 
                tf.constant(1.0), 
                self._alpha_func(
                    self.momentum[variable_name],
                    self.variance[variable_name],
                    self.layer_wise_lr[variable_name],
                    self._strides[variable_name],
                    self._kernel[variable_name],
                    self._filters[variable_name],
                    self._dilation_rate[variable_name],
                    self._units[variable_name],
                    self._pool_size[variable_name],
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape, 
                    self._alpha_dict[variable_name], 
                    grad), use_locking=self._use_locking)
            
        if self._beta_func != None and self._variables_used['beta']:
            training_ops.resource_apply_gradient_descent(
                self._beta_dict[variable_name].handle, 
                tf.constant(1.0), 
                self._beta_func(
                    self.momentum[variable_name],
                    self.variance[variable_name],
                    self.layer_wise_lr[variable_name],
                    self._strides[variable_name],
                    self._kernel[variable_name],
                    self._filters[variable_name],
                    self._dilation_rate[variable_name],
                    self._units[variable_name],
                    self._pool_size[variable_name],
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape, 
                    self._alpha_dict[variable_name], 
                    self._beta_dict[variable_name], 
                    grad), use_locking=self._use_locking)
                
        if self._sigma_func!= None and self._variables_used['sigma']:
            training_ops.resource_apply_gradient_descent(
                self._sigma_dict[variable_name].handle, 
                tf.constant(1.0), 
                self._sigma_func(
                    self.momentum[variable_name],
                    self.variance[variable_name],
                    self.layer_wise_lr[variable_name],
                    self._strides[variable_name],
                    self._kernel[variable_name],
                    self._filters[variable_name],
                    self._dilation_rate[variable_name],
                    self._units[variable_name],
                    self._pool_size[variable_name],
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape, 
                    self._alpha_dict[variable_name], 
                    self._beta_dict[variable_name], 
                    self._sigma_dict[variable_name], 
                    grad), use_locking=self._use_locking)
            
        foo = training_ops.resource_apply_gradient_descent(
                var.handle, 
                tf.constant(1.0), 
                self._grad_func(
                    self.momentum[variable_name],
                    self.variance[variable_name],
                    self.layer_wise_lr[variable_name],
                    self._strides[variable_name],
                    self._kernel[variable_name],
                    self._filters[variable_name],
                    self._dilation_rate[variable_name],
                    self._units[variable_name],
                    self._pool_size[variable_name],
                    self._layer_count[variable_name],
                    self._depth_dict[variable_name],
                    var.shape, 
                    self._alpha_dict[variable_name], 
                    self._beta_dict[variable_name], 
                    self._sigma_dict[variable_name], 
                    grad), use_locking=self._use_locking)
        return foo
    
@tf.function
def momentum_function(grad, prev_momentum, momentum_constant):
    return tf.subtract(
                tf.multiply(
                    tf.subtract(1.0, momentum_constant),
                    prev_momentum),
                tf.multiply(
                    tf.subtract(1.0, momentum_constant),
                    grad),
                )
@tf.function
def variance_function(grad, prev_variance, variance_constant):
    return tf.subtract(
                tf.multiply(
                    tf.subtract(1.0, variance_constant),
                    prev_variance),
                tf.multiply(
                    tf.subtract(1.0, variance_constant),
                    tf.norm(grad)),
                )
@tf.function
def layer_wise_lr_function(grad, prev_layer_wise_lr, weights, layer_wise_lr_constant):
    return tf.add(
                prev_layer_wise_lr,
                tf.multiply(
                    tf.math.divide_no_nan(
                        tf.norm(weights),
                        tf.norm(grad)
                    ),
                    layer_wise_lr_constant,
                )
                )