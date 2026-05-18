import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils.smart_phenotype import readable_phenotype, get_optimizer_type

PADDING_ENCODING = {
    'valid': 0.0,
    'same': 1.0,
    'causal': 1.0,
}


class CustomOptimizer(keras.optimizers.Optimizer):
    def __init__(self,
                            name="CustomOptimizer",
                            phen=None,
                            model=None,
                            **kwargs):

        super(CustomOptimizer, self).__init__(name, **kwargs)
        self.optimizer_type = get_optimizer_type(phen)
        self.phen = phen

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

    def _padding_value_for_layer(self, layer):
        padding = getattr(layer, 'padding', 'valid')
        return PADDING_ENCODING.get(str(padding).lower(), 0.0)

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
                variable_dict[trainable_weight.name] = tf.constant(constant_value, dtype=tf.float32) if self._variables_used[variable_name] else None
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
            'padding': False,
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
        self._padding = {}

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
                self._init_optimizer_variable('strides', self._strides, trainable_weight, constant_value=layer.strides[0] if hasattr(layer, 'strides') else 0.0)
                self._init_optimizer_variable('kernel_size', self._kernel, trainable_weight, constant_value=layer.kernel_size[0] if hasattr(layer, 'kernel_size') else 0.0)
                self._init_optimizer_variable('filters', self._filters, trainable_weight, constant_value=layer.filters if hasattr(layer, 'filters') else 0.0)
                self._init_optimizer_variable('dilation_rate', self._dilation_rate, trainable_weight, constant_value=layer.dilation_rate[0] if hasattr(layer, 'dilation_rate') else 0.0)
                self._init_optimizer_variable('padding', self._padding, trainable_weight, constant_value=self._padding_value_for_layer(layer))

                # Dense variables
                self._init_optimizer_variable('units', self._units, trainable_weight, constant_value=layer.units if hasattr(layer, 'units') else 0.0)
                
                # Pooling variables
                self._init_optimizer_variable('pool_size', self._pool_size, trainable_weight, constant_value=layer.pool_size[0] if hasattr(layer, 'pool_size') else 0.0)

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
            'padding': False,
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
        self._padding = {}

        self._pool_size = {}

        self._units = {}
        
        self._get_variables_used(phen)

        for i, var in zip(range(len(variables)), variables):
            var._name = f"var_{i}:0"  # We need to give the variables names so we can store the optimizer variables in dicts keyed by variable name
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
            self._init_optimizer_variable('padding', self._padding, var, 0.0)

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
                    self._padding[variable_name],
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
                    self._padding[variable_name],
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
                        tf.multiply(self._alpha_func(
                            *parameters
                        ), tf.ones_like(var)), 
                        use_locking=self._use_locking)

        # In the remaining functions it gets a bit hacky
        # We need to add function specific parameters to the parameter list so we just place it in the correct index
        # These function specific parameters are always present at the end before the gradient, regardless of optimizer type
        if self._beta_func != None:
            beta_parameters = parameters[:-1] + [self._beta_dict[variable_name]] + [parameters[-1]]
            self.training_ops.resource_apply_gradient_descent(
                            self._beta_dict[variable_name].handle, 
                            tf.constant(1.0), 
                            tf.multiply(self._beta_func(
                                *beta_parameters
                            ), tf.ones_like(var)), 
                            use_locking=self._use_locking)

        if self._sigma_func!= None:
            sigma_parameters = parameters[:-1] + [self._beta_dict[variable_name], self._sigma_dict[variable_name]] + [parameters[-1]]
            self.training_ops.resource_apply_gradient_descent(
                                self._sigma_dict[variable_name].handle, 
                                tf.constant(1.0), 
                                tf.multiply(self._sigma_func(
                                    *sigma_parameters
                                ), tf.ones_like(var)), 
                                use_locking=self._use_locking)

        weight_parameters = parameters[:-1] + [self._beta_dict[variable_name], self._sigma_dict[variable_name]] + [parameters[-1]]
        #print(self._alpha_dict, var.name)
        updated_weights = self.training_ops.resource_apply_gradient_descent(
                var.handle, 
                tf.constant(1.0), 
                tf.multiply(self._grad_func(
                    *weight_parameters
                ), tf.ones_like(var)), 
                use_locking=self._use_locking)

        return updated_weights
        
    # More recent tensorflow versions require the update step to be defined separately
    def update_step(self, gradient, variable, learning_rate):
        # TODO
        pass
