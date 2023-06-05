from benchmarks.evaluate_cifar_model import evaluate_cifar_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import random
import os
import pandas as pd
#from bayes_opt import BayesianOptimization
from optimizers.evolved.ades import ADES
from optimizers.evolved.sign import Sign
import math
from utils.bayesian_optimization import *

cwd_path = os.getcwd()

def evaluate_adam(learning_rate, beta_1, beta_2):
    optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    result = evaluate_cifar_model(optimizer=optimizer, verbose=0, epochs=1000, step=1000, experiment_name='adam_bo_cifar_results')

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "adam_bo_cifar_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 100, learning_rate, beta_1, beta_2]
    col_names = ["epochs", "learning_rate", "beta_1", "beta_2"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "adam_bo_cifar_results.csv"), index=False)

    return max(result[1]['val_accuracy'])

def evaluate_rmsprop(learning_rate, rho):
    optimizer = RMSprop(learning_rate=learning_rate, rho=rho)
    result = evaluate_cifar_model(optimizer=optimizer, verbose=0, epochs=100, experiment_name='rmsprop_bo_cifar_results')

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "rmsprop_bo_cifar_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 100, learning_rate, rho]
    col_names = ["epochs", "learning_rate", "rho"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "rmsprop_bo_cifar_results.csv"), index=False)

    return max(result[1]['val_accuracy'])

def evaluate_nesterov(learning_rate, momentum):
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    result = evaluate_cifar_model(optimizer=optimizer, verbose=0, epochs=1000, step=1000, experiment_name='nesterov_bo_cifar_results')

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "nesterov_bo_cifar_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 100, learning_rate, momentum]
    col_names = ["epochs", "learning_rate", "momentum"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "nesterov_bo_cifar_results.csv"), index=False)

    return max(result[1]['val_accuracy'])


def evaluate_ades(beta_1, beta_2):
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    import numpy as np
    model = load_model('models/cifar_model.h5', compile=False)
    alpha_dict = {}
    for layer in model.layers:
        for trainable_weight in layer._trainable_weights:
            alpha_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="alpha" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
    optimizer = ADES(beta_1=beta_1, beta_2=beta_2, alpha=alpha_dict)
    result = evaluate_cifar_model(optimizer=optimizer, model=model, verbose=0, epochs=1000, step=1000, experiment_name='ades_bo_cifar_results')
    #result = evaluate_cifar_model(optimizer=optimizer, model=model, verbose=0, epochs=10, experiment_name='ades_bo_cifar_results')

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "ades_bo_cifar_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 100, beta_1, beta_2]
    col_names = ["epochs", "beta_1", "beta_2"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "ades_bo_cifar_results.csv"), index=False)

    return max(result[1]['val_accuracy'])

def evaluate_sign(beta_1):
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    import numpy as np
    model = load_model('models/cifar_model.h5', compile=False)
    optimizer = Sign(beta_1=beta_1)
    result = evaluate_cifar_model(optimizer=optimizer, model=model, verbose=2, epochs=100, experiment_name='sign_bo_cifar_results')

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "sign_bo_cifar_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 100, beta_1]
    col_names = ["epochs", "beta_1"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "sign_bo_cifar_results.csv"), index=False)

    return max(result[1]['val_accuracy'])


def optimize_adam(n_inter, init_points):
    pbounds = {
        'learning_rate': (0, 1),
        'beta_1': (0, 1),
        'beta_2': (0, 1),
    }

    optimizer = BayesianOptimization(
        f=evaluate_adam,
        pbounds=pbounds,
        verbose=1
    )

    optimizer.probe(
        params={
            'learning_rate': 0.001,
            'beta_1': 0.9,
            'beta_2': 0.999,
            }
    )
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_inter,
    )

def optimize_rmsprop(n_inter, init_points):
    pbounds = {
        'learning_rate': (0, 1),
        'rho': (0, 1),
    }

    optimizer = BayesianOptimization(
        f=evaluate_rmsprop,
        pbounds=pbounds,
        verbose=1
    )

    optimizer.probe(
        params={
            'learning_rate': 0.001,
            'rho': 0.9,
            }
    )

    optimizer.maximize(
        init_points=init_points,
        n_iter=n_inter,
    )

def optimize_nesterov(n_inter, init_points):
    pbounds = {
        'learning_rate': (0, 1),
        'momentum': (0, 1)
    }

    optimizer = BayesianOptimization(
        f=evaluate_nesterov,
        pbounds=pbounds,
        verbose=1
    )

    optimizer.probe(
        params={
            'learning_rate': 0.01,
            'momentum': 0.9,
            }
    )

    optimizer.maximize(
        init_points=init_points,
        n_iter=n_inter,
    )


def optimize_ades(n_inter, init_points):
    pbounds = {
        'beta_1': (0, 1),
        'beta_2': (0, 1)
    }

    optimizer = BayesianOptimization(
        f=evaluate_ades,
        pbounds=pbounds,
        verbose=1
    )

    optimizer.probe(
        params={
            'beta_1': 0.08922,
            'beta_2': 0.0891,
            }
    )

    optimizer.maximize(
        init_points=init_points,
        n_iter=n_inter,
    )

def optimize_sign(n_inter, init_points):
    pbounds = {
        'beta_1': (0, 1)
    }

    optimizer = BayesianOptimization(
        f=evaluate_sign,
        pbounds=pbounds,
        verbose=1
    )

    optimizer.probe(
        params={
            'beta_1': 0.0009
            }
    )

    optimizer.maximize(
        init_points=init_points,
        n_iter=n_inter,
    )

def optimize_generic(phenotype, name, n_iter, init_points):
    constants, probes = get_constants_and_probe(phenotype)
    pbounds = {}
    pparams = {}
    i=0

    for constant, probe_value in zip(constants, probes):
        param_key = 'param_' + str(i)
        pbounds[param_key] = (0, 1)
        pparams[param_key] = probe_value
        phenotype.replace(constant, param_key, 1)
        i += 1
    f = create_evaluate_generic(phenotype, name)

    bayesian_optimizer = BayesianOptimization(f=f, pbounds=pbounds, verbose=2)
    bayesian_optimizer.probe(params=pparams)
    bayesian_optimizer.maximize(init_points=init_points, n_iter=n_iter)

def create_evaluate_generic(phenotype, name):
    def evaluate_generic(**kwargs):
        from tensorflow.keras.models import load_model
        import tensorflow as tf
        import numpy as np

        for key, value in kwargs.items():
            phenotype.replace(key, f"tf.constant({value}, shape=shape, dtype=tf.float32)")

        model = load_model('models/cifar_model.h5', compile=False)
        optimizer = CustomOptimizer(phen=phenotype, model=model)
        print("Going to evaluate")
        result = evaluate_cifar_model(optimizer=optimizer, model=model, verbose=0, epochs=100, experiment_name=f'{name}_bo_cifar_results')



        data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , f"{name}_bo_cifar_results.csv"))
        if len(data_frame) > 1:
            total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
        else:
            total_epochs = 0

        col_values = [total_epochs + 100]
        col_names = ["epochs"] 
        for key, value in kwargs.items():
            col_names.append(key)
            col_values.append(value)
        
        data_frame.loc[len(data_frame) - 1, col_names] = col_values
        data_frame.to_csv(os.path.join(cwd_path, 'results/' , f"{name}_bo_cifar_results.csv"), index=False)

        return max(result[1]['val_accuracy'])
    return evaluate_generic

#optimize_adam(90,10)
#optimize_rmsprop(90,10)
#optimize_nesterov(90,10)
#optimize_ades(90,10)
for i in range(29):
    beta_1 = 0.92226
    beta_2 = 0.69285
    evaluate_ades(beta_1, beta_2)
    learning_rate = 0.00163
    beta_1 = 0.81344
    beta_2 = 0.71023
    evaluate_adam(learning_rate, beta_1, beta_2)
    learning_rate = 0.00907
    momentum = 0.98433
    evaluate_nesterov(learning_rate, momentum)


#phenotype = "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.add(alpha, tf.constant(4.70911357e-03, shape=shape, dtype=tf.float32)), lambda shape,  alpha, beta, grad: tf.math.multiply(tf.math.add(tf.math.add(grad, grad), tf.math.add(tf.constant(9.98279874e-01, shape=shape, dtype=tf.float32), tf.math.sqrt(tf.math.square(tf.math.negative(tf.math.multiply(tf.constant(9.94242714e-01, shape=shape, dtype=tf.float32), tf.math.divide_no_nan(grad, grad))))))), grad), lambda shape,  alpha, beta, sigma, grad: tf.constant(9.99720385e-01, shape=shape, dtype=tf.float32), lambda shape,  alpha, beta, sigma, grad: tf.math.multiply(beta, alpha)"
#optimize_generic(phenotype, "best1.3", 90, 10)

#phenotype = "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.add(alpha, grad), lambda shape,  alpha, beta, grad: beta, lambda shape,  alpha, beta, sigma, grad: tf.constant(3.14881358e-03, shape=shape, dtype=tf.float32), lambda shape,  alpha, beta, sigma, grad: tf.math.multiply(sigma, alpha)"
#optimize_generic(phenotype, "1.3_epoch35_id742", 10, 1)
"""
phenotype = "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.add(alpha, tf.constant(4.70911357e-03, shape=shape, dtype=tf.float32)), lambda shape,  alpha, beta, grad: tf.math.multiply(tf.math.add(tf.math.add(grad, grad), tf.math.add(tf.constant(9.99847452e-01, shape=shape, dtype=tf.float32), tf.math.sqrt(tf.math.square(tf.math.negative(tf.math.multiply(grad, tf.constant(9.99944439e-01, shape=shape, dtype=tf.float32))))))), grad), lambda shape,  alpha, beta, sigma, grad: beta, lambda shape,  alpha, beta, sigma, grad: tf.math.multiply(beta, alpha)"
optimize_generic(phenotype, "1.3_epoch36_id775", 10, 1)

phenotype = "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.add(alpha, tf.math.subtract(grad, tf.constant(1.90885420e-02, shape=shape, dtype=tf.float32))), lambda shape,  alpha, beta, grad: tf.math.multiply(tf.math.add(tf.math.add(grad, grad), grad), tf.math.subtract(tf.math.negative(tf.constant(1.52547986e-04, shape=shape, dtype=tf.float32)), tf.constant(3.85103236e-03, shape=shape, dtype=tf.float32))), lambda shape,  alpha, beta, sigma, grad: grad, lambda shape,  alpha, beta, sigma, grad: beta"
optimize_generic(phenotype, "1.3_epoch42_id999", 10, 1)

phenotype = "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.add(alpha, tf.math.subtract(grad, alpha)), lambda shape,  alpha, beta, grad: tf.math.multiply(tf.math.add(tf.math.add(grad, grad), grad), tf.math.subtract(tf.math.negative(tf.constant(2.57431039e-03, shape=shape, dtype=tf.float32)), tf.constant(7.67413430e-04, shape=shape, dtype=tf.float32))), lambda shape,  alpha, beta, sigma, grad: grad, lambda shape,  alpha, beta, sigma, grad: beta"
optimize_generic(phenotype, "1.3_epoch45_id1134", 10, 1)

phenotype = "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.add(alpha, tf.constant(4.70911357e-03, shape=shape, dtype=tf.float32)), lambda shape,  alpha, beta, grad: tf.math.multiply(tf.math.add(tf.math.add(grad, grad), tf.math.add(tf.constant(9.99875353e-01, shape=shape, dtype=tf.float32), tf.math.sqrt(tf.math.square(tf.math.negative(tf.math.multiply(alpha, tf.constant(9.99581233e-01, shape=shape, dtype=tf.float32))))))), grad), lambda shape,  alpha, beta, sigma, grad: beta, lambda shape,  alpha, beta, sigma, grad: tf.math.multiply(beta, alpha)"
optimize_generic(phenotype, "1.3_epoch47_id1202", 10, 1)

phenotype = "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.add(alpha, tf.constant(4.70911357e-03, shape=shape, dtype=tf.float32)), lambda shape,  alpha, beta, grad: tf.math.multiply(tf.math.add(tf.math.add(grad, grad), tf.math.add(tf.constant(9.99847452e-01, shape=shape, dtype=tf.float32), tf.math.sqrt(tf.math.square(tf.math.negative(tf.math.multiply(grad, tf.constant(8.59898661e-03, shape=shape, dtype=tf.float32))))))), grad), lambda shape,  alpha, beta, sigma, grad: tf.constant(3.14881358e-03, shape=shape, dtype=tf.float32), lambda shape,  alpha, beta, sigma, grad: tf.math.multiply(beta, alpha)"
optimize_generic(phenotype, "1.3_epoch54_id1519", 10, 1)

phenotype = "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.add(alpha, tf.constant(4.70911357e-03, shape=shape, dtype=tf.float32)), lambda shape,  alpha, beta, grad: tf.math.multiply(tf.math.add(tf.math.add(grad, grad), tf.math.add(tf.constant(9.99847452e-01, shape=shape, dtype=tf.float32), tf.math.sqrt(tf.math.square(tf.math.negative(tf.math.multiply(grad, tf.constant(3.76354517e-01, shape=shape, dtype=tf.float32))))))), grad), lambda shape,  alpha, beta, sigma, grad: tf.constant(3.14881358e-03, shape=shape, dtype=tf.float32), lambda shape,  alpha, beta, sigma, grad: tf.math.multiply(beta, alpha)"
optimize_generic(phenotype, "1.3_epoch55_id1522", 10, 1)

phenotype = "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.add(alpha, tf.constant(4.70911357e-03, shape=shape, dtype=tf.float32)), lambda shape,  alpha, beta, grad: tf.math.multiply(tf.math.add(tf.math.add(grad, grad), tf.math.add(tf.constant(9.98279874e-01, shape=shape, dtype=tf.float32), tf.math.sqrt(tf.math.square(tf.math.negative(tf.math.multiply(tf.constant(9.94242714e-01, shape=shape, dtype=tf.float32), tf.math.divide_no_nan(grad, grad))))))), grad), lambda shape,  alpha, beta, sigma, grad: tf.constant(9.99720385e-01, shape=shape, dtype=tf.float32), lambda shape,  alpha, beta, sigma, grad: tf.math.multiply(beta, alpha)"
optimize_generic(phenotype, "1.3_epoch75_id2458", 10, 1)
"""
