import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import *

class SGD(Optimizer):
  def __init__(self,
               learning_rate=0.01,
               momentum=0.0,
               nesterov=False,
               name="SGD",
               **kwargs):
    
    super(SGD, self).__init__(name, **kwargs)
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
    self._set_hyper("decay", self._initial_decay)

    self._momentum = False
    if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
      self._momentum = True
    if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
      raise ValueError("`momentum` must be between [0, 1].")
    self._set_hyper("momentum", momentum)

    self.nesterov = nesterov
    self.native_alpha = []

  def _create_slots(self, var_list):
    if self._momentum:
      for var in var_list:
        self.add_slot(var, "momentum")

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(SGD, self)._prepare_local(var_device, var_dtype, apply_state)
    apply_state[(var_device, var_dtype)]["momentum"] = array_ops.identity(
        self._get_hyper("momentum", var_dtype))

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    if self._momentum:
      momentum_var = self.get_slot(var, "momentum")

      self.native_alpha.append(momentum_var.numpy())
      
      return training_ops.resource_apply_keras_momentum(
          var.handle,
          momentum_var.handle,
          coefficients["lr_t"],
          grad,
          coefficients["momentum"],
          use_locking=self._use_locking,
          use_nesterov=self.nesterov)
    else:
      return training_ops.resource_apply_gradient_descent(
          var.handle, coefficients["lr_t"], grad, use_locking=self._use_locking)

  def get_config(self):
    config = super(SGD, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "decay": self._serialize_hyperparameter("decay"),
        "momentum": self._serialize_hyperparameter("momentum"),
        "nesterov": self.nesterov,
    })
    return config

class Adadelta(Optimizer):

  def __init__(self,
               learning_rate=0.001,
               rho=0.95,
               epsilon=1e-7,
               name='Adadelta',
               **kwargs):
    super(Adadelta, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('rho', rho)
    self.epsilon = epsilon or backend_config.epsilon()

  def _create_slots(self, var_list):
    # Separate for-loops to respect the ordering of slot variables from v1.
    for v in var_list:
      self.add_slot(v, 'accum_grad')
    for v in var_list:
      self.add_slot(v, 'accum_var')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(Adadelta, self)._prepare_local(var_device, var_dtype, apply_state)
    apply_state[(var_device, var_dtype)].update(dict(
        epsilon=ops.convert_to_tensor(self.epsilon, var_dtype),
        rho=array_ops.identity(self._get_hyper('rho', var_dtype))
    ))

  def set_weights(self, weights):
    params = self.weights
    # Override set_weights for backward compatibility of Keras V1 optimizer
    # since it does not include iteration at head of the weight list. Set
    # iteration to 0.
    if len(params) == len(weights) + 1:
      weights = [np.array(0)] + weights
    super(Adadelta, self).set_weights(weights)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    accum_grad = self.get_slot(var, 'accum_grad')
    accum_var = self.get_slot(var, 'accum_var')
    return training_ops.resource_apply_adadelta(
        var.handle,
        accum_grad.handle,
        accum_var.handle,
        coefficients['lr_t'],
        coefficients['rho'],
        coefficients['epsilon'],
        grad,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    accum_grad = self.get_slot(var, 'accum_grad')
    accum_var = self.get_slot(var, 'accum_var')
    return training_ops.resource_sparse_apply_adadelta(
        var.handle,
        accum_grad.handle,
        accum_var.handle,
        coefficients['lr_t'],
        coefficients['rho'],
        coefficients['epsilon'],
        grad,
        indices,
        use_locking=self._use_locking)

  def get_config(self):
    config = super(Adadelta, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._serialize_hyperparameter('decay'),
        'rho': self._serialize_hyperparameter('rho'),
        'epsilon': self.epsilon,
    })
    return config

class Adam(Optimizer):
  

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               amsgrad=False,
               name='Adam',
               **kwargs):
    

    super(Adam, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self.epsilon = epsilon or backend_config.epsilon()
    self.amsgrad = amsgrad
    self.native_alpha = []
    self.native_beta = []

  def _create_slots(self, var_list):
    # Create slots for the first and second moments.
    # Separate for-loops to respect the ordering of slot variables from v1.
    for var in var_list:
      self.add_slot(var, 'm')
    for var in var_list:
      self.add_slot(var, 'v')
    if self.amsgrad:
      for var in var_list:
        self.add_slot(var, 'vhat')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(Adam, self)._prepare_local(var_device, var_dtype, apply_state)

    local_step = math_ops.cast(self.iterations + 1, var_dtype)
    beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
    beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
    beta_1_power = math_ops.pow(beta_1_t, local_step)
    beta_2_power = math_ops.pow(beta_2_t, local_step)
    lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
          (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
    apply_state[(var_device, var_dtype)].update(dict(
        lr=lr,
        epsilon=ops.convert_to_tensor(self.epsilon, var_dtype),
        beta_1_t=beta_1_t,
        beta_1_power=beta_1_power,
        one_minus_beta_1_t=1 - beta_1_t,
        beta_2_t=beta_2_t,
        beta_2_power=beta_2_power,
        one_minus_beta_2_t=1 - beta_2_t
    ))

  def set_weights(self, weights):
    params = self.weights
    # If the weights are generated by Keras V1 optimizer, it includes vhats
    # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
    # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
    num_vars = int((len(params) - 1) / 2)
    if len(weights) == 3 * num_vars + 1:
      weights = weights[:len(params)]
    super(Adam, self).set_weights(weights)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    self.native_alpha.append(m.numpy())
    self.native_beta.append(v.numpy())
    if not self.amsgrad:
      return training_ops.resource_apply_adam(
          var.handle,
          m.handle,
          v.handle,
          coefficients['beta_1_power'],
          coefficients['beta_2_power'],
          coefficients['lr_t'],
          coefficients['beta_1_t'],
          coefficients['beta_2_t'],
          coefficients['epsilon'],
          grad,
          use_locking=self._use_locking)
    else:
      vhat = self.get_slot(var, 'vhat')
      return training_ops.resource_apply_adam_with_amsgrad(
          var.handle,
          m.handle,
          v.handle,
          vhat.handle,
          coefficients['beta_1_power'],
          coefficients['beta_2_power'],
          coefficients['lr_t'],
          coefficients['beta_1_t'],
          coefficients['beta_2_t'],
          coefficients['epsilon'],
          grad,
          use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, 'm')
    m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
    m_t = state_ops.assign(m, m * coefficients['beta_1_t'],
                           use_locking=self._use_locking)
    with ops.control_dependencies([m_t]):
      m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slot(var, 'v')
    v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
    v_t = state_ops.assign(v, v * coefficients['beta_2_t'],
                           use_locking=self._use_locking)
    with ops.control_dependencies([v_t]):
      v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

    if not self.amsgrad:
      v_sqrt = math_ops.sqrt(v_t)
      var_update = state_ops.assign_sub(
          var, coefficients['lr'] * m_t / (v_sqrt + coefficients['epsilon']),
          use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update, m_t, v_t])
    else:
      v_hat = self.get_slot(var, 'vhat')
      v_hat_t = math_ops.maximum(v_hat, v_t)
      with ops.control_dependencies([v_hat_t]):
        v_hat_t = state_ops.assign(
            v_hat, v_hat_t, use_locking=self._use_locking)
      v_hat_sqrt = math_ops.sqrt(v_hat_t)
      var_update = state_ops.assign_sub(
          var,
          coefficients['lr'] * m_t / (v_hat_sqrt + coefficients['epsilon']),
          use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update, m_t, v_t, v_hat_t])

  def get_config(self):
    config = super(Adam, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._serialize_hyperparameter('decay'),
        'beta_1': self._serialize_hyperparameter('beta_1'),
        'beta_2': self._serialize_hyperparameter('beta_2'),
        'epsilon': self.epsilon,
        'amsgrad': self.amsgrad,
    })
    return config

class Adamax(Optimizer):
  

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               name='Adamax',
               **kwargs):
    
    super(Adamax, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self.epsilon = epsilon or backend_config.epsilon()
    self.native_alpha = []
    self.native_beta = []

  def _create_slots(self, var_list):
    # Separate for-loops to respect the ordering of slot variables from v1.
    for var in var_list:
      self.add_slot(var, 'm')  # Create slots for the first moments.
    for var in var_list:
      self.add_slot(var, 'v')  # Create slots for the second moments.

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(Adamax, self)._prepare_local(var_device, var_dtype, apply_state)

    local_step = math_ops.cast(self.iterations + 1, var_dtype)
    beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
    beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
    beta_1_power = math_ops.pow(beta_1_t, local_step)
    lr_t = apply_state[(var_device, var_dtype)]['lr_t']

    apply_state[(var_device, var_dtype)].update(dict(
        neg_scaled_lr=-lr_t / (1 - beta_1_power),
        epsilon=ops.convert_to_tensor(self.epsilon, var_dtype),
        beta_1_t=beta_1_t,
        beta_1_power=beta_1_power,
        one_minus_beta_1_t=1 - beta_1_t,
    beta_2_t=beta_2_t,
        zero=array_ops.zeros((), dtype=dtypes.int64)
    ))

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')
    self.native_alpha.append(m.numpy()) 
    self.native_beta.append(v.numpy())

    return training_ops.resource_apply_ada_max(
        var.handle,
        m.handle,
        v.handle,
        coefficients['beta_1_power'],
        coefficients['lr_t'],
        coefficients['beta_1_t'],
        coefficients['beta_2_t'],
        coefficients['epsilon'],
        grad,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slot(var, 'm')
    m_slice = array_ops.gather(m, indices, axis=coefficients['zero'])
    m_t_slice = (m_slice * coefficients['beta_1_t'] +
                 grad * coefficients['one_minus_beta_1_t'])
    with ops.control_dependencies([m_t_slice]):
      m_t = self._resource_scatter_update(m, indices, m_t_slice)

    # u_t = max(beta2 * u, abs(g_t))
    v = self.get_slot(var, 'v')
    v_slice = array_ops.gather(v, indices, axis=coefficients['zero'])
    v_t_slice = math_ops.maximum(v_slice * coefficients['beta_2_t'],
                                 math_ops.abs(grad))
    with ops.control_dependencies([v_t_slice]):
      v_t = self._resource_scatter_update(v, indices, v_t_slice)
    # theta_t = theta - lr / (1 - beta1^t) * m_t / u_t
    var_slice = coefficients['neg_scaled_lr'] * (
        m_t_slice / (v_t_slice + coefficients['epsilon']))
    with ops.control_dependencies([var_slice]):
      var_update = self._resource_scatter_add(var, indices, var_slice)
    return control_flow_ops.group(*[var_update, m_t, v_t])

  def get_config(self):
    config = super(Adamax, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._serialize_hyperparameter('decay'),
        'beta_1': self._serialize_hyperparameter('beta_1'),
        'beta_2': self._serialize_hyperparameter('beta_2'),
        'epsilon': self.epsilon,
    })
    return config

class Ftrl(Optimizer):
  

  def __init__(self,
               learning_rate=0.001,
               learning_rate_power=-0.5,
               initial_accumulator_value=0.1,
               l1_regularization_strength=0.0,
               l2_regularization_strength=0.0,
               name='Ftrl',
               l2_shrinkage_regularization_strength=0.0,
               **kwargs):
    
    super(Ftrl, self).__init__(name, **kwargs)

    if initial_accumulator_value < 0.0:
      raise ValueError(
          'initial_accumulator_value %f needs to be positive or zero' %
          initial_accumulator_value)
    if learning_rate_power > 0.0:
      raise ValueError('learning_rate_power %f needs to be negative or zero' %
                       learning_rate_power)
    if l1_regularization_strength < 0.0:
      raise ValueError(
          'l1_regularization_strength %f needs to be positive or zero' %
          l1_regularization_strength)
    if l2_regularization_strength < 0.0:
      raise ValueError(
          'l2_regularization_strength %f needs to be positive or zero' %
          l2_regularization_strength)
    if l2_shrinkage_regularization_strength < 0.0:
      raise ValueError(
          'l2_shrinkage_regularization_strength %f needs to be positive'
          ' or zero' % l2_shrinkage_regularization_strength)
    
    self.native_alpha = []
    self.native_beta = []

    self._set_hyper('learning_rate', learning_rate)
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('learning_rate_power', learning_rate_power)
    self._set_hyper('l1_regularization_strength', l1_regularization_strength)
    self._set_hyper('l2_regularization_strength', l2_regularization_strength)
    self._initial_accumulator_value = initial_accumulator_value
    self._l2_shrinkage_regularization_strength = (
        l2_shrinkage_regularization_strength)

  def _create_slots(self, var_list):
    # Create the "accum" and "linear" slots.
    for var in var_list:
      dtype = var.dtype.base_dtype
      init = init_ops.constant_initializer(
          self._initial_accumulator_value, dtype=dtype)
      self.add_slot(var, 'accumulator', init)
      self.add_slot(var, 'linear')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(Ftrl, self)._prepare_local(var_device, var_dtype, apply_state)
    apply_state[(var_device, var_dtype)].update(dict(
        learning_rate_power=array_ops.identity(
            self._get_hyper('learning_rate_power', var_dtype)),
        l1_regularization_strength=array_ops.identity(
            self._get_hyper('l1_regularization_strength', var_dtype)),
        l2_regularization_strength=array_ops.identity(
            self._get_hyper('l2_regularization_strength', var_dtype)),
        l2_shrinkage_regularization_strength=math_ops.cast(
            self._l2_shrinkage_regularization_strength, var_dtype)
        ))

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    accum = self.get_slot(var, 'accumulator')
    linear = self.get_slot(var, 'linear')

    self.native_alpha.append(accum.numpy())
    self.native_beta.append(linear.numpy())

    if self._l2_shrinkage_regularization_strength <= 0.0:
      return training_ops.resource_apply_ftrl(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          coefficients['lr_t'],
          coefficients['l1_regularization_strength'],
          coefficients['l2_regularization_strength'],
          coefficients['learning_rate_power'],
          use_locking=self._use_locking)
    else:
      return training_ops.resource_apply_ftrl_v2(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          coefficients['lr_t'],
          coefficients['l1_regularization_strength'],
          coefficients['l2_regularization_strength'],
          coefficients['l2_shrinkage_regularization_strength'],
          coefficients['learning_rate_power'],
          use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    accum = self.get_slot(var, 'accumulator')
    linear = self.get_slot(var, 'linear')

    if self._l2_shrinkage_regularization_strength <= 0.0:
      return training_ops.resource_sparse_apply_ftrl(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          indices,
          coefficients['lr_t'],
          coefficients['l1_regularization_strength'],
          coefficients['l2_regularization_strength'],
          coefficients['learning_rate_power'],
          use_locking=self._use_locking)
    else:
      return training_ops.resource_sparse_apply_ftrl_v2(
          var.handle,
          accum.handle,
          linear.handle,
          grad,
          indices,
          coefficients['lr_t'],
          coefficients['l1_regularization_strength'],
          coefficients['l2_regularization_strength'],
          coefficients['l2_shrinkage_regularization_strength'],
          coefficients['learning_rate_power'],
          use_locking=self._use_locking)

  def get_config(self):
    config = super(Ftrl, self).get_config()
    config.update({
        'learning_rate':
            self._serialize_hyperparameter('learning_rate'),
        'decay':
            self._serialize_hyperparameter('decay'),
        'initial_accumulator_value':
            self._initial_accumulator_value,
        'learning_rate_power':
            self._serialize_hyperparameter('learning_rate_power'),
        'l1_regularization_strength':
            self._serialize_hyperparameter('l1_regularization_strength'),
        'l2_regularization_strength':
            self._serialize_hyperparameter('l2_regularization_strength'),
        'l2_shrinkage_regularization_strength':
            self._l2_shrinkage_regularization_strength,
    })
    return config

class Nadam(Optimizer):
  

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               name='Nadam',
               **kwargs):
    

    # Backwards compatibility with keras NAdam optimizer.
    kwargs['decay'] = kwargs.pop('schedule_decay', 0.004)
    learning_rate = kwargs.get('lr', learning_rate)
    if isinstance(learning_rate, learning_rate_schedule.LearningRateSchedule):
      raise ValueError('The Nadam optimizer does not support '
                       'tf.keras.optimizers.LearningRateSchedules as the '
                       'learning rate.')

    super(Nadam, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self.epsilon = epsilon or backend_config.epsilon()
    self._m_cache = None
    self.native_alpha = []
    self.native_beta = []

  def _create_slots(self, var_list):
    var_dtype = var_list[0].dtype.base_dtype
    if self._m_cache is None:
      self._m_cache = self.add_weight(
          'momentum_cache',
          shape=[],
          dtype=var_dtype,
          initializer='ones',
          trainable=False,
          aggregation=tf_variables.VariableAggregation.ONLY_FIRST_REPLICA)
      self._weights.append(self._m_cache)
    # Separate for-loops to respect the ordering of slot variables from v1.
    for var in var_list:
      # Create slots for the first moments.
      self.add_slot(var, 'm')
    for var in var_list:
      # Create slots for the second moments.
      self.add_slot(var, 'v')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    lr_t = array_ops.identity(self._get_hyper('learning_rate', var_dtype))
    beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
    beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
    local_step = math_ops.cast(self.iterations + 1, var_dtype)
    next_step = math_ops.cast(self.iterations + 2, var_dtype)

    decay_base = math_ops.cast(0.96, var_dtype)

    m_t = beta_1_t * (1. - 0.5 * (
        math_ops.pow(decay_base, self._initial_decay * local_step)))
    m_t_1 = beta_1_t * (1. - 0.5 * (
        math_ops.pow(decay_base, self._initial_decay * next_step)))

    m_schedule_new = math_ops.cast(self._m_cache_read, var_dtype) * m_t
    if var_dtype is self._m_cache.dtype:
      m_schedule_new = array_ops.identity(state_ops.assign(
          self._m_cache, m_schedule_new, use_locking=self._use_locking))
    m_schedule_next = m_schedule_new * m_t_1

    apply_state[(var_device, var_dtype)] = dict(
        lr_t=lr_t,
        neg_lr_t=-lr_t,
        epsilon=ops.convert_to_tensor(self.epsilon, var_dtype),
        beta_1_t=beta_1_t,
        beta_2_t=beta_2_t,
        m_t=m_t,
        m_t_1=m_t_1,

        one_minus_beta_1_t=1 - beta_1_t,
        one_minus_beta_2_t=1 - beta_2_t,
        one_minus_m_t=1. - m_t,
        one_minus_m_schedule_new=1. - m_schedule_new,
        one_minus_m_schedule_next=1. - m_schedule_next,
        v_t_prime_denominator=1. - math_ops.pow(beta_2_t, local_step),
    )

  def _prepare(self, var_list):
    # Get the value of the momentum cache before starting to apply gradients.
    self._m_cache_read = array_ops.identity(self._m_cache)
    return super(Nadam, self)._prepare(var_list)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    self.native_alpha.append(m.numpy())
    self.native_beta.append(v.numpy())

    g_prime = grad / coefficients['one_minus_m_schedule_new']
    m_t = (coefficients['beta_1_t'] * m +
           coefficients['one_minus_beta_1_t'] * grad)
    m_t = state_ops.assign(m, m_t, use_locking=self._use_locking)
    m_t_prime = m_t / coefficients['one_minus_m_schedule_next']
    v_t = (coefficients['beta_2_t'] * v +
           coefficients['one_minus_beta_2_t'] * math_ops.square(grad))
    v_t = state_ops.assign(v, v_t, use_locking=self._use_locking)
    v_t_prime = v_t / coefficients['v_t_prime_denominator']
    m_t_bar = (coefficients['one_minus_m_t'] * g_prime +
               coefficients['m_t_1'] * m_t_prime)
    var_t = var - coefficients['lr_t'] * m_t_bar / (
        math_ops.sqrt(v_t_prime) + coefficients['epsilon'])
    return state_ops.assign(var, var_t, use_locking=self._use_locking).op

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    m = self.get_slot(var, 'm')
    v = self.get_slot(var, 'v')

    g_prime = grad / coefficients['one_minus_m_schedule_new']

    # m_t = beta1 * m + (1 - beta1) * g_t
    m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
    m_t = state_ops.assign(m, m * coefficients['beta_1_t'],
                           use_locking=self._use_locking)

    with ops.control_dependencies([m_t]):
      m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
      m_t_slice = array_ops.gather(m_t, indices)

    m_t_prime = m_t_slice / coefficients['one_minus_m_schedule_next']
    m_t_bar = (coefficients['one_minus_m_t'] * g_prime +
               coefficients['m_t_1'] * m_t_prime)

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v_scaled_g_values = (grad * grad) * coefficients['one_minus_beta_2_t']
    v_t = state_ops.assign(v, v * coefficients['beta_2_t'],
                           use_locking=self._use_locking)

    with ops.control_dependencies([v_t]):
      v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)
      v_t_slice = array_ops.gather(v_t, indices)

    v_t_prime = v_t_slice / coefficients['v_t_prime_denominator']
    v_prime_sqrt_plus_eps = math_ops.sqrt(v_t_prime) + coefficients['epsilon']

    var_update = self._resource_scatter_add(
        var, indices,
        coefficients['neg_lr_t'] * m_t_bar / v_prime_sqrt_plus_eps)
    return control_flow_ops.group(*[var_update, m_t_bar, v_t])

  def get_config(self):
    config = super(Nadam, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._serialize_hyperparameter('decay'),
        'beta_1': self._serialize_hyperparameter('beta_1'),
        'beta_2': self._serialize_hyperparameter('beta_2'),
        'epsilon': self.epsilon,
    })
    return config

class RMSprop(Optimizer):
  

  def __init__(self,
               learning_rate=0.001,
               rho=0.9,
               momentum=0.0,
               epsilon=1e-7,
               centered=False,
               name="RMSprop",
               **kwargs):
    
    super(RMSprop, self).__init__(name, **kwargs)
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
    self._set_hyper("decay", self._initial_decay)
    self._set_hyper("rho", rho)

    self._momentum = False
    if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
      self._momentum = True
    if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
      raise ValueError("`momentum` must be between [0, 1].")
    self._set_hyper("momentum", momentum)

    self.epsilon = epsilon or backend_config.epsilon()
    self.centered = centered
    self.native_alpha = []
    self.native_beta = []

  def _create_slots(self, var_list):
    for var in var_list:
      self.add_slot(var, "rms")
    if self._momentum:
      for var in var_list:
        self.add_slot(var, "momentum")
    if self.centered:
      for var in var_list:
        self.add_slot(var, "mg")

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(RMSprop, self)._prepare_local(var_device, var_dtype, apply_state)

    rho = array_ops.identity(self._get_hyper("rho", var_dtype))
    apply_state[(var_device, var_dtype)].update(dict(
        neg_lr_t=-apply_state[(var_device, var_dtype)]["lr_t"],
        epsilon=ops.convert_to_tensor(self.epsilon, var_dtype),
        rho=rho,
        momentum=array_ops.identity(self._get_hyper("momentum", var_dtype)),
        one_minus_rho=1. - rho
    ))

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    rms = self.get_slot(var, "rms")
    if self._momentum:
      mom = self.get_slot(var, "momentum")
      self.native_alpha.append(rms.numpy())
      self.native_beta.append(mom.numpy())
      if self.centered:
        mg = self.get_slot(var, "mg")
        return training_ops.resource_apply_centered_rms_prop(
            var.handle,
            mg.handle,
            rms.handle,
            mom.handle,
            coefficients["lr_t"],
            coefficients["rho"],
            coefficients["momentum"],
            coefficients["epsilon"],
            grad,
            use_locking=self._use_locking)
      else:
        return training_ops.resource_apply_rms_prop(
            var.handle,
            rms.handle,
            mom.handle,
            coefficients["lr_t"],
            coefficients["rho"],
            coefficients["momentum"],
            coefficients["epsilon"],
            grad,
            use_locking=self._use_locking)
    else:
      rms_t = (coefficients["rho"] * rms +
               coefficients["one_minus_rho"] * math_ops.square(grad))
      rms_t = state_ops.assign(rms, rms_t, use_locking=self._use_locking)
      denom_t = rms_t
      if self.centered:
        mg = self.get_slot(var, "mg")
        mg_t = coefficients["rho"] * mg + coefficients["one_minus_rho"] * grad
        mg_t = state_ops.assign(mg, mg_t, use_locking=self._use_locking)
        denom_t = rms_t - math_ops.square(mg_t)
      var_t = var - coefficients["lr_t"] * grad / (
          math_ops.sqrt(denom_t) + coefficients["epsilon"])
      return state_ops.assign(var, var_t, use_locking=self._use_locking).op

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    rms = self.get_slot(var, "rms")
    if self._momentum:
      mom = self.get_slot(var, "momentum")
      if self.centered:
        mg = self.get_slot(var, "mg")
        return training_ops.resource_sparse_apply_centered_rms_prop(
            var.handle,
            mg.handle,
            rms.handle,
            mom.handle,
            coefficients["lr_t"],
            coefficients["rho"],
            coefficients["momentum"],
            coefficients["epsilon"],
            grad,
            indices,
            use_locking=self._use_locking)
      else:
        return training_ops.resource_sparse_apply_rms_prop(
            var.handle,
            rms.handle,
            mom.handle,
            coefficients["lr_t"],
            coefficients["rho"],
            coefficients["momentum"],
            coefficients["epsilon"],
            grad,
            indices,
            use_locking=self._use_locking)
    else:
      rms_scaled_g_values = (grad * grad) * coefficients["one_minus_rho"]
      rms_t = state_ops.assign(rms, rms * coefficients["rho"],
                               use_locking=self._use_locking)
      with ops.control_dependencies([rms_t]):
        rms_t = self._resource_scatter_add(rms, indices, rms_scaled_g_values)
        rms_slice = array_ops.gather(rms_t, indices)
      denom_slice = rms_slice
      if self.centered:
        mg = self.get_slot(var, "mg")
        mg_scaled_g_values = grad * coefficients["one_minus_rho"]
        mg_t = state_ops.assign(mg, mg * coefficients["rho"],
                                use_locking=self._use_locking)
        with ops.control_dependencies([mg_t]):
          mg_t = self._resource_scatter_add(mg, indices, mg_scaled_g_values)
          mg_slice = array_ops.gather(mg_t, indices)
          denom_slice = rms_slice - math_ops.square(mg_slice)
      var_update = self._resource_scatter_add(
          var, indices, coefficients["neg_lr_t"] * grad / (
              math_ops.sqrt(denom_slice) + coefficients["epsilon"]))
      if self.centered:
        return control_flow_ops.group(*[var_update, rms_t, mg_t])
      return control_flow_ops.group(*[var_update, rms_t])

  def set_weights(self, weights):
    params = self.weights
    # Override set_weights for backward compatibility of Keras V1 optimizer
    # since it does not include iteration at head of the weight list. Set
    # iteration to 0.
    if len(params) == len(weights) + 1:
      weights = [np.array(0)] + weights
    super(RMSprop, self).set_weights(weights)

  def get_config(self):
    config = super(RMSprop, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "decay": self._serialize_hyperparameter("decay"),
        "rho": self._serialize_hyperparameter("rho"),
        "momentum": self._serialize_hyperparameter("momentum"),
        "epsilon": self.epsilon,
        "centered": self.centered,
    })
    return config

class Adagrad(Optimizer):
  

  def __init__(self,
               learning_rate=0.001,
               initial_accumulator_value=0.1,
               epsilon=1e-7,
               name='Adagrad',
               **kwargs):
    
    if initial_accumulator_value < 0.0:
      raise ValueError('initial_accumulator_value must be non-negative: %s' %
                       initial_accumulator_value)
    if epsilon is None:
      epsilon = backend_config.epsilon()
    super(Adagrad, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._initial_accumulator_value = initial_accumulator_value
    self.epsilon = epsilon or backend_config.epsilon()
    self.native_alpha = []

  def _create_slots(self, var_list):
    for var in var_list:
      dtype = var.dtype.base_dtype
      init = init_ops.constant_initializer(
          self._initial_accumulator_value, dtype=dtype)
      self.add_slot(var, 'accumulator', init)

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(Adagrad, self)._prepare_local(var_device, var_dtype, apply_state)
    apply_state[(var_device, var_dtype)].update(dict(
        epsilon=ops.convert_to_tensor(self.epsilon, var_dtype),
        neg_lr_t=-apply_state[(var_device, var_dtype)]['lr_t'],
        zero=array_ops.zeros((), dtype=dtypes.int64)
    ))

  def set_weights(self, weights):
    params = self.weights
    # Override set_weights for backward compatibility of Keras V1 optimizer
    # since it does not include iteration at head of the weight list. Set
    # iteration to 0.
    if len(params) == len(weights) + 1:
      weights = [np.array(0)] + weights
    super(Adagrad, self).set_weights(weights)

  @classmethod
  def from_config(cls, config, custom_objects=None):
    
    if 'initial_accumulator_value' not in config:
      config['initial_accumulator_value'] = 0.
    if 'lr' in config:
      config['learning_rate'] = config.pop('lr')
    return cls(**config)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    acc = self.get_slot(var, 'accumulator')
    self.native_alpha.append(acc.numpy())
    return training_ops.resource_apply_adagrad_v2(
        var.handle,
        acc.handle,
        coefficients['lr_t'],
        coefficients['epsilon'],
        grad,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    acc = self.get_slot(var, 'accumulator')
    return training_ops.resource_sparse_apply_adagrad_v2(
        var.handle,
        acc.handle,
        coefficients['lr_t'],
        coefficients['epsilon'],
        grad,
        indices,
        use_locking=self._use_locking)

  def get_config(self):
    config = super(Adagrad, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'decay': self._serialize_hyperparameter('decay'),
        'initial_accumulator_value': self._initial_accumulator_value,
        'epsilon': self.epsilon,
    })
    return config