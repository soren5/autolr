import pytest
from examples.learning_rate_optimizer import Test_LROptimizer
import sge
import sys


def test_pop_random():
    print(sys.argv)
    print("TEST")
    sys.argv = ['0', '--parameters', '/Users/soren/Dropbox/Research/dsge_learning_rate/sge/parameters/test_params1.yml']
    sge.evolutionary_algorithm(evaluation_function=Test_LROptimizer())
    sys.argv = ['0', '--parameters', '/Users/soren/Dropbox/Research/dsge_learning_rate/sge/parameters/test_params2.yml']
    pop1 = sge.evolutionary_algorithm(evaluation_function=Test_LROptimizer(), resume_generation=5) 

    sys.argv = ['0', '--parameters', '/Users/soren/Dropbox/Research/dsge_learning_rate/sge/parameters/test_params1.yml']
    sge.evolutionary_algorithm(evaluation_function=Test_LROptimizer())
    sys.argv = ['0', '--parameters', '/Users/soren/Dropbox/Research/dsge_learning_rate/sge/parameters/test_params2.yml']
    pop2 = sge.evolutionary_algorithm(evaluation_function=Test_LROptimizer(), resume_generation=5) 
    for i, j in zip(pop1, pop2):
        print(i)
        print(j)
        print(i == j)

def test_keras_random():
    from examples.model_evaluator_adaptive import train_model_lite
    import random
    import numpy as np
    import tensorflow as tf
    random.seed(0)
    np.random.seed(random.randint(0, 1000000))
    tf.random.set_seed(random.randint(0, 1000000))
    train_model_lite('alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.add(tf.math.add(tf.constant(8.32200197e-05, shape=shape, dtype=tf.float32), grad), tf.math.subtract(tf.constant(5.55606489e-05, shape=shape, dtype=tf.float32), grad)), lambda shape,  alpha, beta, grad: tf.math.add(alpha, tf.math.add(tf.math.multiply(tf.math.sqrt(tf.math.sqrt(tf.math.multiply(grad, tf.math.negative(tf.math.negative(tf.constant(9.38616893e-01, shape=shape, dtype=tf.float32)))))), grad), grad)), lambda shape,  alpha, beta, sigma, grad: tf.math.multiply(tf.constant(-9.99581233e-02, shape=shape, dtype=tf.float32), grad), lambda shape,  alpha, beta, sigma, grad: tf.math.multiply(tf.constant(1.99581233e-02, shape=shape, dtype=tf.float32), grad)')
    random.seed(0)
    np.random.seed(random.randint(0, 1000000))
    tf.random.set_seed(random.randint(0, 1000000))
    train_model_lite('alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.add(tf.math.add(tf.constant(8.32200197e-05, shape=shape, dtype=tf.float32), grad), tf.math.subtract(tf.constant(5.55606489e-05, shape=shape, dtype=tf.float32), grad)), lambda shape,  alpha, beta, grad: tf.math.add(alpha, tf.math.add(tf.math.multiply(tf.math.sqrt(tf.math.sqrt(tf.math.multiply(grad, tf.math.negative(tf.math.negative(tf.constant(9.38616893e-01, shape=shape, dtype=tf.float32)))))), grad), grad)), lambda shape,  alpha, beta, sigma, grad: tf.math.multiply(tf.constant(-9.99581233e-02, shape=shape, dtype=tf.float32), grad), lambda shape,  alpha, beta, sigma, grad: tf.math.multiply(tf.constant(1.99581233e-02, shape=shape, dtype=tf.float32), grad)')
#test_keras_random()
#test_pop_random()