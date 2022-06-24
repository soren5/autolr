from deap import creator, base, tools
from benchmarks.evaluate_fashion_mnist_model import evaluate_fashion_mnist_model
from benchmarks.evaluate_cifar_model import evaluate_cifar_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from optimizers.ades import ADES
from optimizers.sign import Sign
import random
import os
import pandas as pd
from bayes_opt import BayesianOptimization
import math

cwd_path = os.getcwd()

def evaluate_adam_fashion(learning_rate, beta_1, beta_2):
    optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    result = evaluate_fashion_mnist_model(optimizer=optimizer, verbose=2, epochs=100, step=100, experiment_name='adam_fashion_results', save_best_only=True)

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "adam_fashion_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 1000, learning_rate, beta_1, beta_2]
    col_names = ["epochs", "learning_rate", "beta_1", "beta_2"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "adam_fashion_results.csv"), index=False)

    return max(result[1]['val_accuracy'])

def evaluate_rmsprop_fashion(learning_rate, rho):
    optimizer = RMSprop(learning_rate=learning_rate, rho=rho)
    result = evaluate_fashion_mnist_model(optimizer=optimizer, verbose=2, epochs=100, step=100, experiment_name='rmsprop_fashion_results', save_best_only=True)

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "rmsprop_fashion_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 1000, learning_rate, rho]
    col_names = ["epochs", "learning_rate", "rho"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "rmsprop_fashion_results.csv"), index=False)

    return max(result[1]['val_accuracy'])

def evaluate_nesterov_fashion(learning_rate, momentum):
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    result = evaluate_fashion_mnist_model(optimizer=optimizer, verbose=2, epochs=100, step=100, experiment_name='nesterov_fashion_results', save_best_only=True)

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "nesterov_fashion_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 1000, learning_rate, momentum]
    col_names = ["epochs", "learning_rate", "momentum"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "nesterov_fashion_results.csv"), index=False)

    return max(result[1]['val_accuracy'])

def evaluate_ades_fashion(beta_1, beta_2):
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    import numpy as np
    model = load_model('models/mnist_model.h5', compile=False)
    alpha_dict = {}
    for layer in model.layers:
        for trainable_weight in layer._trainable_weights:
            alpha_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="alpha" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
    optimizer = ADES(beta_1=beta_1, beta_2=beta_2, alpha=alpha_dict)
    result = evaluate_fashion_mnist_model(optimizer=optimizer, model=model, verbose=2, epochs=100, step=100, experiment_name='ades_fashion_results',save_best_only= True)

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "ades_fashion_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 1000, beta_1, beta_2]
    col_names = ["epochs", "beta_1", "beta_2"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "ades_fashion_results.csv"), index=False)

    return max(result[1]['val_accuracy'])

def evaluate_sign_fashion(beta_1):
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    import numpy as np
    model = load_model('models/mnist_model.h5', compile=False)
    optimizer = Sign(beta_1=beta_1)
    result = evaluate_fashion_mnist_model(optimizer=optimizer, model=model, verbose=2, epochs=100, step=100, experiment_name='sign_fashion_results', save_best_only=True)

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "sign_fashion_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 1000, beta_1]
    col_names = ["epochs", "beta_1"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "sign_fashion_results.csv"), index=False)

    return max(result[1]['val_accuracy'])

def evaluate_adam_cifar(learning_rate, beta_1, beta_2):
    optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    result = evaluate_cifar_model(optimizer=optimizer, verbose=2, epochs=100, step=100, experiment_name='adam_cifar_results', save_best_only=True)

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "adam_cifar_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 1000, learning_rate, beta_1, beta_2]
    col_names = ["epochs", "learning_rate", "beta_1", "beta_2"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "adam_cifar_results.csv"), index=False)

    return max(result[1]['val_accuracy'])

def evaluate_rmsprop_cifar(learning_rate, rho):
    optimizer = RMSprop(learning_rate=learning_rate, rho=rho)
    result = evaluate_cifar_model(optimizer=optimizer, verbose=2, epochs=100, step=100, experiment_name='rmsprop_cifar_results', save_best_only=True)

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "rmsprop_cifar_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 1000, learning_rate, rho]
    col_names = ["epochs", "learning_rate", "rho"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "rmsprop_cifar_results.csv"), index=False)

    return max(result[1]['val_accuracy'])

def evaluate_nesterov_cifar(learning_rate, momentum):
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    result = evaluate_cifar_model(optimizer=optimizer, verbose=2, epochs=100, step=100, experiment_name='nesterov_cifar_results', save_best_only=True)

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "nesterov_cifar_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 1000, learning_rate, momentum]
    col_names = ["epochs", "learning_rate", "momentum"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "nesterov_cifar_results.csv"), index=False)

    return max(result[1]['val_accuracy'])

def evaluate_ades_cifar(beta_1, beta_2):
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    import numpy as np
    model = load_model('models/cifar_model.h5', compile=False)
    alpha_dict = {}
    for layer in model.layers:
        for trainable_weight in layer._trainable_weights:
            alpha_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="alpha" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
    optimizer = ADES(beta_1=beta_1, beta_2=beta_2, alpha=alpha_dict)
    result = evaluate_cifar_model(optimizer=optimizer, model=model, verbose=2, epochs=100, step=100, experiment_name='ades_cifar_results', save_best_only=True)

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "ades_cifar_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 1000, beta_1, beta_2]
    col_names = ["epochs", "beta_1", "beta_2"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "ades_cifar_results.csv"), index=False)

    return max(result[1]['val_accuracy'])

def evaluate_sign_cifar(beta_1):
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    import numpy as np
    model = load_model('models/cifar_model.h5', compile=False)
    optimizer = Sign(beta_1=beta_1)
    result = evaluate_cifar_model(optimizer=optimizer, model=model, verbose=2, epochs=100, step=100, experiment_name='sign_cifar_results', save_best_only=True)

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "sign_cifar_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 1000, beta_1]
    col_names = ["epochs", "beta_1"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "sign_cifar_results.csv"), index=False)

    return max(result[1]['val_accuracy'])






