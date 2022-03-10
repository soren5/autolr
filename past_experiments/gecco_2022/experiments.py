from benchmarks.evaluate_fashion_mnist_model import evaluate_fashion_mnist_model
from benchmarks.evaluate_cifar_model import evaluate_cifar_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from optimizers.ades import ADES
from optimizers.sign import Sign
from utils.custom_optimizer import CustomOptimizer
import random
import os
import pandas as pd
import math

cwd_path = os.getcwd()

def evaluate_adam_fashion(learning_rate, beta_1, beta_2):
    optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    result = evaluate_fashion_mnist_model(optimizer=optimizer, verbose=2, epochs=1000, step=1000, experiment_name='adam_fashion_results', save_best_only=True)

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
    result = evaluate_fashion_mnist_model(optimizer=optimizer, verbose=2, epochs=1000, step=1000, experiment_name='rmsprop_fashion_results', save_best_only=True)

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
    result = evaluate_fashion_mnist_model(optimizer=optimizer, verbose=2, epochs=1000, step=1000, experiment_name='nesterov_fashion_results', save_best_only=True)

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
    result = evaluate_fashion_mnist_model(optimizer=optimizer, model=model, verbose=2, epochs=1000, step=1000, experiment_name='ades_fashion_results',save_best_only= True)

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
    result = evaluate_fashion_mnist_model(optimizer=optimizer, model=model, verbose=2, epochs=1000, step=1000, experiment_name='sign_fashion_results', save_best_only=True)

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
    result = evaluate_cifar_model(optimizer=optimizer, verbose=2, epochs=1000, step=1000, experiment_name='adam_cifar_results', save_best_only=True)

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
    result = evaluate_cifar_model(optimizer=optimizer, verbose=2, epochs=1000, step=1000, experiment_name='rmsprop_cifar_results', save_best_only=True)

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
    result = evaluate_cifar_model(optimizer=optimizer, verbose=2, epochs=1000, step=1000, experiment_name='nesterov_cifar_results', save_best_only=True)

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
    result = evaluate_cifar_model(optimizer=optimizer, model=model, verbose=2, epochs=1000, step=1000, experiment_name='ades_cifar_results', save_best_only=True)

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
    result = evaluate_cifar_model(optimizer=optimizer, model=model, verbose=2, epochs=1000, step=1000, experiment_name='sign_cifar_results', save_best_only=True)

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


    return max(result[1]['val_accuracy'])

def evaluate_rmsprop_var_fashion(var_number, phen):
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    import numpy as np
    model = load_model('models/mnist_model.h5', compile=False)
    alpha_dict = {}
    beta_dict = {}
    sigma_dict = {}
    for layer in model.layers:
        for trainable_weight in layer._trainable_weights:
            alpha_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="alpha" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
            beta_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="beta" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
            sigma_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="sigma" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
    foo = {"tf": tf}
    exec(phen, foo)
    alpha_func = foo["alpha_func"]
    beta_func = foo["beta_func"]
    sigma_func = foo["sigma_func"]
    grad_func = foo["grad_func"]
    optimizer = CustomOptimizer(alpha=alpha_dict, alpha_func=alpha_func, beta=beta_dict, beta_func=beta_func, sigma=sigma_dict, sigma_func=sigma_func, grad_func=grad_func)
    result = evaluate_fashion_mnist_model(optimizer=optimizer, model=model, verbose=2, epochs=1000, step=1000, experiment_name='rms_var_fashion_results',save_best_only= True)

    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "rms_var_fashion_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 1000, var_number]
    col_names = ["epochs", "var"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "rms_var_fashion_results.csv"), index=False)

    return max(result[1]['val_accuracy'])

for seed in range(20, 50):
    random.seed(seed)
    #evaluate_rmsprop_var_fashion(1, "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(tf.constant(0.0, shape=shape, dtype=tf.float32), tf.constant(1.07052146e-01, shape=shape, dtype=tf.float32)), alpha), tf.math.multiply(tf.constant(9.96148968e-01, shape=shape, dtype=tf.float32), tf.math.square(grad)))), lambda shape,  alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(tf.constant(0.0, shape=shape, dtype=tf.float32), tf.constant(1.05038445e-02, shape=shape, dtype=tf.float32)), beta), tf.math.divide_no_nan(tf.math.multiply(tf.constant(1.14904229e-03, shape=shape, dtype=tf.float32), grad), tf.math.sqrt(tf.math.add(alpha, tf.constant(1.14904229e-03, shape=shape, dtype=tf.float32)))))), lambda shape,  alpha, beta, sigma, grad: tf.math.negative(tf.math.add(tf.constant(0.0, shape=shape, dtype=tf.float32), grad)), lambda shape,  alpha, beta, sigma, grad: beta")
    #evaluate_rmsprop_var_fashion(2, "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.constant(0.0, shape=shape, dtype=tf.float32), lambda shape,  alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(tf.constant(0.0, shape=shape, dtype=tf.float32), tf.constant(1.05038445e-02, shape=shape, dtype=tf.float32)), beta), tf.math.divide_no_nan(tf.math.multiply(tf.constant(1.14904229e-03, shape=shape, dtype=tf.float32), grad), tf.math.sqrt(tf.math.add(alpha, tf.constant(1.14904229e-03, shape=shape, dtype=tf.float32)))))), lambda shape,  alpha, beta, sigma, grad: sigma, lambda shape,  alpha, beta, sigma, grad: beta ")
    #evaluate_rmsprop_var_fashion(3, "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.constant(0.0, shape=shape, dtype=tf.float32), lambda shape,  alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(tf.constant(0.0, shape=shape, dtype=tf.float32), tf.constant(1.05038445e-02, shape=shape, dtype=tf.float32)), beta), tf.math.divide_no_nan(tf.math.multiply(tf.constant(1.14904229e-03, shape=shape, dtype=tf.float32), grad), tf.math.sqrt(tf.math.add(alpha, tf.constant(1.14904229e-03, shape=shape, dtype=tf.float32)))))), lambda shape,  alpha, beta, sigma, grad: tf.math.negative(tf.math.add(grad, grad)), lambda shape,  alpha, beta, sigma, grad: beta")
    evaluate_rmsprop_var_fashion(5, "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(tf.constant(0.0, shape=shape, dtype=tf.float32), tf.constant(1.07052146e-01, shape=shape, dtype=tf.float32)), alpha), tf.math.multiply(tf.constant(1.07052146e-01, shape=shape, dtype=tf.float32), tf.math.square(grad)))), lambda shape,  alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.negative(beta), tf.math.divide_no_nan(tf.math.multiply(tf.constant(1.14904229e-03, shape=shape, dtype=tf.float32), grad), tf.math.sqrt(tf.math.add(alpha, tf.constant(5.55606489e-05, shape=shape, dtype=tf.float32)))))), lambda shape,  alpha, beta, sigma, grad: tf.math.negative(tf.math.add(tf.constant(5.55606489e-05, shape=shape, dtype=tf.float32), grad)), lambda shape,  alpha, beta, sigma, grad: beta")
    #evaluate_rmsprop_fashion(0.001, 0.9)
