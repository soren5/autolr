import csv
from utils.data_functions import load_data_evolution
import tensorflow as tf
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

from tensorflow.keras.models import load_model
from tensorflow import keras
from keras import backend as K
from optimizers.custom_optimizer import CustomOptimizer

import sys

import numpy as np
import datetime
from sge.parameters import (
    params,
    set_parameters
)
set_parameters(sys.argv[1:])

experiment_time = datetime.datetime.now()



def train_model(phen):
    print(params['EPOCHS'])
    validation_size = params['VALIDATION_SIZE']
    fitness_size = params['FITNESS_SIZE'] 
    batch_size = params['BATCH_SIZE']
    epochs = params['EPOCHS']
    patience = params['PATIENCE']

    dataset = load_data_evolution(validation_size=validation_size, test_size=fitness_size, split=True, img_size=(28,28))
    model = load_model(params['MODEL'], compile=False)
    weights = model.get_weights()
    
    final_score = 1
    final_info = None

    for i in range(5):
        model.set_weights(weights)

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
        opt = CustomOptimizer(alpha=alpha_dict, alpha_func=alpha_func, beta=beta_dict, beta_func=beta_func, sigma=sigma_dict, sigma_func=sigma_func, grad_func=grad_func)
        
        
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)

        score = model.fit(dataset['x_train'][i], dataset['y_train'][i],
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
        test_score = model.evaluate(x=dataset['x_test'],y=dataset["y_test"], verbose=0, callbacks=[keras.callbacks.History()])

        print("trial ", i, ": ", test_score[-1])
        if test_score[-1] < final_score:
            final_score = test_score[-1]
            final_info = results
        
        if test_score[-1] < 0.8:
            break
    print("final fitness: ", final_score)
    return final_score, results