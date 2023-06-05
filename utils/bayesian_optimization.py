import numpy as np
import json
import os
import random
import pickle
from bayes_opt import BayesianOptimization
import os
import sys
import re
import utils.smart_phenotype as s_phenotype
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import csv
from utils.data_functions import load_fashion_mnist_training, load_cifar10_training, load_fashion_mnist_full
import tensorflow as tf
"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
"""
from tensorflow.keras.models import load_model
from tensorflow import keras
from keras import backend as K
from optimizers.custom_optimizer import CustomOptimizer

import sys

import numpy as np
import datetime

import utils.create_models as c
c.create_models()

def get_constants_and_probe(phenotype):
    smart_phenotype = s_phenotype.smart_phenotype(phenotype)
    print(smart_phenotype)
    probe = []
    constant_strings = []
    for tf_constant in re.findall("tf.constant\([0-9]\.[0-9]+e.0[0-9], shape=shape, dtype=tf.float32\)", phenotype):
        value = re.findall("[0-9]\.[0-9]+e.0[0-9]", tf_constant)
        if value[0] in smart_phenotype:
            constant_strings.append(tf_constant)
            probe.append(float(value[0]))
    return constant_strings, probe

def create_optimizer_with_params(phenotype, **kwargs):
    for key, value in kwargs.items():
        phenotype.replace(key, f"tf.constant({value}, shape=shape, dtype=tf.float32)")
    return phenotype

def create_evaluate_optimizer_function(phenotype, params, train_model):
    def evaluate_optimizer(**kwargs):
        for key, value in kwargs.items():
            phenotype.replace(key, f"tf.constant({value}, shape=shape, dtype=tf.float32)")
        fitness, other_info = train_model((phenotype, params)) 
        #print(fitness, kwargs)
        return fitness
    return evaluate_optimizer


def tune_optimizer(n_iter, init_points, phenotype, params):
    constants, probes = get_constants_and_probe(phenotype)
    f = create_evaluate_optimizer_function(phenotype, params, train_model_fmnist_full)
    pbounds = {}
    params = {}
    i = 0
    for constant, probe_value in zip(constants, probes):
        param_key = 'param_' + str(i)
        pbounds[param_key] = (0, 1)
        params[param_key] = probe_value
        phenotype.replace(constant, param_key, 1)
        i += 1

    optimizer = BayesianOptimization(
        f=f,
        pbounds=pbounds,
        verbose=2
    )

    optimizer.probe(params=params)
    
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
    )

def train_model_fmnist(phen_params):
    phen, params = phen_params
    validation_size = params['VALIDATION_SIZE']
    test_size = params['FITNESS_SIZE'] 
    batch_size = params['BATCH_SIZE']
    epochs = params['EPOCHS']
    patience = params['PATIENCE']

    dataset = load_fashion_mnist_training(validation_size=validation_size, test_size=test_size)
    model = load_model(params['MODEL'], compile=False)
    weights = model.get_weights()
    opt = CustomOptimizer(phen=phen, model=model)

    scores = []
    for x in range(15):
      model.set_weights(weights)
      
      
      model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
      early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)

      score = model.fit(dataset['x_train'], dataset['y_train'],
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
          validation_data=(dataset['x_val'], dataset['y_val']),
          validation_steps= validation_size // batch_size,
          callbacks=[
              early_stop
          ])

      K.clear_session()
      results = {}
      for metric in score.history:
          results[metric] = []
          for n in score.history[metric]:
              results[metric].append(n)
      test_score = model.evaluate(x=dataset['x_test'],y=dataset["y_test"], verbose=0, callbacks=[keras.callbacks.History()])
      scores.append(test_score[-1])
    from statistics import mean
    return mean(scores), results

def get_test_score_fminst(phen_params):
    phen, params = phen_params
    validation_size = params['VALIDATION_SIZE']
    batch_size = params['BATCH_SIZE']
    epochs = 1000
    patience = 1001

    dataset = load_fashion_mnist_full(validation_size=validation_size)
    model = load_model(params['MODEL'], compile=False)
    weights = model.get_weights()
    opt = CustomOptimizer(phen=phen, model=model)

    scores = []

    model.set_weights(weights)
    
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)

    score = model.fit(dataset['x_train'], dataset['y_train'],
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=(dataset['x_val'], dataset['y_val']),
        validation_steps= validation_size // batch_size,
        callbacks=[
            early_stop
        ])

    K.clear_session()
    results = {}
    for metric in score.history:
        results[metric] = []
        for n in score.history[metric]:
            results[metric].append(n)
    val_score = model.evaluate(x=dataset['x_val'],y=dataset["y_val"], verbose=2, callbacks=[keras.callbacks.History()])
    test_score = model.evaluate(x=dataset['x_test'],y=dataset["y_test"], verbose=2, callbacks=[keras.callbacks.History()])
    scores.append(test_score[-1])
    from statistics import mean
    return val_score[-1], test_score[-1]
def train_model_fmnist_full(phen_params):
    phen, params = phen_params
    validation_size = params['VALIDATION_SIZE']
    batch_size = params['BATCH_SIZE']
    epochs = 1000
    patience = 1001

    dataset = load_fashion_mnist_training(validation_size=validation_size, test_size=0)
    model = load_model(params['MODEL'], compile=False)
    weights = model.get_weights()
    opt = CustomOptimizer(phen=phen, model=model)

    scores = []
    for x in range(15):
      model.set_weights(weights)
      
      
      model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
      early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)

      score = model.fit(dataset['x_train'], dataset['y_train'],
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
          validation_data=(dataset['x_val'], dataset['y_val']),
          validation_steps= validation_size // batch_size,
          callbacks=[
              early_stop
          ])

      K.clear_session()
      results = {}
      for metric in score.history:
          results[metric] = []
          for n in score.history[metric]:
              results[metric].append(n)
      test_score = model.evaluate(x=dataset['x_val'],y=dataset["y_val"], verbose=0, callbacks=[keras.callbacks.History()])
      scores.append(test_score[-1])
    from statistics import mean
    return mean(scores), results

def train_model_tensorflow_cifar10(phen_params):
    phen, params = phen_params
    validation_size = params['VALIDATION_SIZE']
    fitness_size =params['FITNESS_SIZE'] 
    batch_size = params['BATCH_SIZE']
    epochs = params['EPOCHS']
    patience = params['PATIENCE']

    dataset = load_cifar10_training(validation_size=validation_size, test_size=fitness_size)
    model = load_model(params['MODEL'], compile=False)
    weights = model.get_weights()



    # optimizer is constant aslong as phen doesn't changed?
    # -> opportunity to cache opt and compiled model

    scores = []
    for x in range(15):
        model.set_weights(weights)
        opt = CustomOptimizer(phen=phen, model=model)

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)

        score = model.fit(dataset['x_train'], dataset['y_train'],
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(dataset['x_val'], dataset['y_val']),
            validation_steps= validation_size // batch_size,
            callbacks=[
                early_stop
            ])

        K.clear_session()
        results = {}
        for metric in score.history:
            results[metric] = []
            for n in score.history[metric]:
                results[metric].append(n)
        test_score = model.evaluate(x=dataset['x_test'],y=dataset["y_test"], verbose=0, callbacks=[keras.callbacks.History()])
        scores.append(test_score[-1])
    from statistics import mean
    return mean(scores), results

mnist_params = {
    "BATCH_SIZE": 1000,
    "EPOCHS": 100,
    "FITNESS_FLOOR": 0,
    "PATIENCE": 5,
    "VALIDATION_SIZE": 3500,
    "FITNESS_SIZE": 50000,
    "MODEL": 'models/mnist_model.h5',
}

cifar_params = {
    "BATCH_SIZE": 1000,
    "EPOCHS": 100,
    "FITNESS_FLOOR": 0,
    "PATIENCE": 5,
    "VALIDATION_SIZE": 3500,
    "FITNESS_SIZE": 40000,
    "MODEL": 'models/cifar_model.h5',
}
#1.3 Best Phenotype
#phenotype = "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.add(alpha, tf.constant(4.70911357e-03, shape=shape, dtype=tf.float32)), lambda shape,  alpha, beta, grad: tf.math.multiply(tf.math.add(tf.math.add(grad, grad), tf.math.add(tf.constant(9.98279874e-01, shape=shape, dtype=tf.float32), tf.math.sqrt(tf.math.square(tf.math.negative(tf.math.multiply(tf.constant(9.94242714e-01, shape=shape, dtype=tf.float32), tf.math.divide_no_nan(grad, grad))))))), grad), lambda shape,  alpha, beta, sigma, grad: tf.constant(9.99720385e-01, shape=shape, dtype=tf.float32), lambda shape,  alpha, beta, sigma, grad: tf.math.multiply(beta, alpha)"
#tune_optimizer(90, 10, phenotype, cifar_params)
#1.3 Generation 35 Best Phenotype
#phenotype = "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.add(alpha, grad), lambda shape,  alpha, beta, grad: beta, lambda shape,  alpha, beta, sigma, grad: tf.constant(3.14881358e-03, shape=shape, dtype=tf.float32), lambda shape,  alpha, beta, sigma, grad: tf.math.multiply(sigma, alpha)"
#tune_optimizer(90, 10, phenotype, cifar_params)
#print(get_constants_and_probe(phenotype))
"""
for x in range(15):
    phenotype = "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.divide_no_nan(grad, tf.constant(1.72012560e-03, shape=shape, dtype=tf.float32)), lambda shape,  alpha, beta, grad: tf.constant(8.59898661e-03, shape=shape, dtype=tf.float32), lambda shape,  alpha, beta, sigma, grad: tf.math.multiply(tf.math.add(tf.math.add(sigma, grad), grad), tf.constant(1.56514861e-02, shape=shape, dtype=tf.float32)), lambda shape,  alpha, beta, sigma, grad: tf.math.negative(sigma)"
    with open("log.txt", 'a') as f:
        val, test = get_test_score_fminst((phenotype, mnist_params))
        print(f"FM,{val},{test}", file=f)
    phenotype = "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.pow(grad, tf.constant(9.99372875e-01, shape=shape, dtype=tf.float32)), lambda shape,  alpha, beta, grad: tf.math.add(grad, grad), lambda shape,  alpha, beta, sigma, grad: tf.math.subtract(tf.math.multiply(tf.math.add(sigma, grad), tf.constant(8.92170603e-02, shape=shape, dtype=tf.float32)), tf.math.square(tf.constant(8.32200197e-05, shape=shape, dtype=tf.float32))), lambda shape,  alpha, beta, sigma, grad: tf.math.negative(sigma)"
    with open("log.txt", 'a') as f:
        val, test = get_test_score_fminst((phenotype, mnist_params))
        print(f"FMX,{val},{test}", file=f)
    phenotype = "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.negative(alpha), lambda shape,  alpha, beta, grad: tf.math.subtract(tf.math.multiply(tf.math.subtract(tf.math.add(tf.constant(1.07052146e-01, shape=shape, dtype=tf.float32), grad), tf.constant(5.75728612e-03, shape=shape, dtype=tf.float32)), beta), grad), lambda shape,  alpha, beta, sigma, grad: grad, lambda shape,  alpha, beta, sigma, grad: tf.math.multiply(beta, tf.math.add(tf.math.subtract(tf.constant(1.28252101e-02, shape=shape, dtype=tf.float32), alpha), tf.constant(2.11963334e-01, shape=shape, dtype=tf.float32)))"
    with open("log.txt", 'a') as f:
        val, test = get_test_score_fminst((phenotype, mnist_params))
        print(f"OM,{val},{test}", file=f)
    phenotype = "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.multiply(tf.math.subtract(alpha, grad), tf.constant(7.03711536e-03, shape=shape, dtype=tf.float32)), lambda shape,  alpha, beta, grad: beta, lambda shape,  alpha, beta, sigma, grad: sigma, lambda shape,  alpha, beta, sigma, grad: tf.math.add(tf.math.subtract(alpha, tf.math.divide_no_nan(beta, tf.math.add(alpha, beta))), alpha)"
    with open("log.txt", 'a') as f:
        val, test = get_test_score_fminst((phenotype, mnist_params))
        print(f"OMX,{val},{test}", file=f)
        
"""
