from asyncio import constants
from socket import NI_NUMERICHOST
from tabnanny import verbose
from deap import creator, base, tools
from benchmarks.evaluate_fashion_mnist_model import evaluate_fashion_mnist_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from optimizers.evolved.ades import ADES
from optimizers.evolved.signved.sign import Sign
import random
import os
import pandas as pd
from bayes_opt import BayesianOptimization
import math
from utils.bayesian_optimization import *

cwd_path = os.getcwd()

def evaluate_adam(learning_rate, beta_1, beta_2):
    optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    result = evaluate_fashion_mnist_model(optimizer=optimizer, verbose=0, epochs=100, experiment_name='adam_bo_fashion_results')

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "adam_bo_fashion_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 100, learning_rate, beta_1, beta_2]
    col_names = ["epochs", "learning_rate", "beta_1", "beta_2"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "adam_bo_fashion_results.csv"), index=False)

    return max(result[1]['val_accuracy'])

def evaluate_rmsprop(learning_rate, rho):
    optimizer = RMSprop(learning_rate=learning_rate, rho=rho)
    result = evaluate_fashion_mnist_model(optimizer=optimizer, verbose=0, epochs=100, experiment_name='rmsprop_bo_fashion_results')

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "rmsprop_bo_fashion_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 100, learning_rate, rho]
    col_names = ["epochs", "learning_rate", "rho"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "rmsprop_bo_fashion_results.csv"), index=False)

    return max(result[1]['val_accuracy'])

def evaluate_nesterov(learning_rate, momentum):
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    result = evaluate_fashion_mnist_model(optimizer=optimizer, verbose=0, epochs=100, experiment_name='nesterov_bo_fashion_results')

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "nesterov_bo_fashion_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 100, learning_rate, momentum]
    col_names = ["epochs", "learning_rate", "momentum"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "nesterov_bo_fashion_results.csv"), index=False)

    return max(result[1]['val_accuracy'])

def evaluate_ades(beta_1, beta_2):
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    import numpy as np
    model = load_model('models/mnist_model.h5', compile=False)
    alpha_dict = {}
    for layer in model.layers:
        for trainable_weight in layer._trainable_weights:
            alpha_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="alpha" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
    optimizer = ADES(beta_1=beta_1, beta_2=beta_2, alpha=alpha_dict)
    result = evaluate_fashion_mnist_model(optimizer=optimizer, model=model, verbose=0, epochs=100, experiment_name='ades_bo_fashion_results')

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "ades_bo_fashion_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 100, beta_1, beta_2]
    col_names = ["epochs", "beta_1", "beta_2"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "ades_bo_fashion_results.csv"), index=False)

    return max(result[1]['val_accuracy'])

def evaluate_sign(beta_1):
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    import numpy as np
    model = load_model('models/mnist_model.h5', compile=False)
    optimizer = Sign(beta_1=beta_1)
    result = evaluate_fashion_mnist_model(optimizer=optimizer, model=model, verbose=2, epochs=100, experiment_name='sign_bo_fashion_results')

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "sign_bo_fashion_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 100, beta_1]
    col_names = ["epochs", "beta_1"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "sign_bo_fashion_results.csv"), index=False)

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

def optimize_generic(phenotype, n_iter, init_points):
    constants, probes = get_constants_and_probe(phenotype)
    pbounds = {}
    pparams = {}
    for constant, probe_value in zip(constants, probes):
        param_key = 'param_' + str(i)
        pbounds[param_key] = (0, 1)
        pparams[param_key] = probe_value
        phenotype.replace(constant, param_key, 1)
        i += 1
    f = create_evaluate_generic(phenotype)

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

        model = load_model('models/mnist_model.h5', compile=False)
        optimizer = CustomOptimizer(phen=phenotype)
        result = evaluate_fashion_mnist_model(optimizer=optimizer, model=model, verbose=2, epochs=100, experiment_name=f'{name}_bo_fashion_results')

        data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , f"{name}_bo_fashion_results.csv"))
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
        data_frame.to_csv(os.path.join(cwd_path, 'results/' , f"{name}_bo_fashion_results.csv"), index=False)

        return max(result[1]['val_accuracy'])

#optimize_adam(90,10)
#optimize_rmsprop(90,10)
#optimize_nesterov(90,10)
#optimize_ades(90, 10)
#optimize_sign(90, 10)