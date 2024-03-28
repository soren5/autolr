from main import Optimizer_Evaluator_Dual_Task
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from sge.parameters import (
    params,
    set_parameters
)
import sge
import sys

set_parameters(sys.argv[1:])   
phen = "alpha_func, beta_func, sigma_func, grad_func = lambda layer_count, layer_num, shape, alpha, grad: layer_count, lambda layer_count, layer_num, shape, alpha, beta, grad: tf.constant(0.01, shape=shape, dtype=tf.float32), lambda layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.divide_no_nan(grad, layer_count), lambda layer_count, layer_num, shape, alpha, beta, sigma, grad: tf.math.multiply(grad, tf.math.divide_no_nan(tf.math.divide_no_nan(grad, tf.math.negative(tf.math.multiply(grad, tf.math.subtract(layer_count, layer_num)))), tf.math.negative(layer_num)))"
evaluation_function = Optimizer_Evaluator_Dual_Task()
evaluation_function.init_net(params)
evaluation_function.init_data(params)
evaluation_function.init_evaluation(params)
evaluation_function.evaluate(phen, params)
