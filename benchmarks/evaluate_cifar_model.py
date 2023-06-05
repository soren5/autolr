
import csv
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
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import fashion_mnist, mnist, cifar10
from tensorflow.keras import backend as K
import numpy as np
import datetime
import json
import random
import time
from tensorflow.python.framework import ops
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from optimizers.evolved.ades import ADES
#from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from utils.data_functions import load_cifar10_full
import os
import pandas as pd

cwd_path = os.getcwd()


def get_metric_dictionary(score):
    #This is cursed code I did during my masters
    #I do not remember why it is like this but it does create the dictionary I want so I am not too miffed
    pain = {}
    for metric in score.history:
        #print(metric)
        pain[metric] = []
        for n in score.history[metric]:
            #print(n)
            if type(n) == np.float32:
                n = n.item()
            pain[metric].append(n)
    return pain

def evaluate_cifar_model(dataset=None, model=None, optimizer=None, batch_size=1000, epochs=100, step=100, verbose=0, experiment_name='development_results', save_best_only=True):
    assert optimizer != None

    if dataset is None:
        dataset = load_cifar10_full()

    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    validation_size = len(dataset['x_val'])

    if model == None:
        n_model = load_model('models/cifar_model.h5', compile=False)
        model = n_model

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(cwd_path, f'models/checkpoints/cifar_model_checkpoint_{experiment_name}.h5'),
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=save_best_only)

    epoch_progress = 0

    while epoch_progress < epochs:
        if epochs - epoch_progress < step:
            run_epochs = epochs - epoch_progress
        else:
            run_epochs = step
        print(epochs)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        score = model.fit(dataset['x_train'], dataset['y_train'],
            batch_size=batch_size,
            epochs=run_epochs,
            verbose=2,
            validation_data=(dataset['x_val'], dataset['y_val']),
            validation_steps= validation_size // batch_size,
            callbacks=[model_checkpoint_callback])
        K.clear_session()


        metric_dictionary = get_metric_dictionary(score)
        if save_best_only:
            model.load_weights(os.path.join(cwd_path, f'models/checkpoints/cifar_model_checkpoint_{experiment_name}.h5'))
        test_score = model.evaluate(dataset['x_test'], dataset['y_test'], batch_size=batch_size, verbose=verbose, callbacks=[keras.callbacks.History()])
        result = [test_score[-1], metric_dictionary]
        data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , experiment_name + ".csv"))
        col_names = ["epochs", "accuracy", "loss", "val_accuracy", "val_loss"] 
        print(col_names)
        for x in range(len(metric_dictionary['val_loss'])):
           col_values = [x, metric_dictionary['accuracy'][x], metric_dictionary['loss'][x], metric_dictionary['val_accuracy'][x], metric_dictionary['val_loss'][x]]
           print(col_values)
           data_frame = data_frame.append(pd.DataFrame([col_values], columns=col_names), ignore_index=True)
    
        """
        col_values = [max(result[1]['val_accuracy']), min(result[1]['val_loss']), result[0]]
        col_names = ["max_val_accuracy", "min_val_loss", "test_accuracy"] 
        data_frame = data_frame.append(pd.DataFrame([col_values], columns=col_names), ignore_index=True)
        """
        data_frame.to_csv(os.path.join(cwd_path, 'results/' , experiment_name + ".csv"), index=False)
        epoch_progress += run_epochs

     
    return result

def resume_cifar_model(dataset=None, optimizer=None, batch_size=1000, epochs=100, verbose=0):  
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    n_model = load_model('models/cifar_model.h5', compile=False)
    model = n_model 
    model.load_weights(os.path.join(cwd_path, f'models/checkpoints/cifar_model_checkpoint.h5'))
    evaluate_cifar_model(model=model, optimizer=optimizer, epochs=epochs, verbose=verbose)



if __name__ == "__main__":
    resume = False
    model = load_model('models/cifar_model.h5', compile=False)
    alpha_dict = {}
    for layer in model.layers:
        for trainable_weight in layer._trainable_weights:
            alpha_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="alpha" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
    
    optimizer = Adam(learning_rate=0.00163, beta_1=0.81344, beta_2=0.71023)
    col_names = ["epochs", "val_accuracy", "val_loss"] 
    data_frame = pd.DataFrame(columns=col_names)
    experiment_name= "adam_bo_cifar_results"
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , experiment_name + ".csv"), index=False)
    evaluate_cifar_model(optimizer=optimizer, model= model, batch_size=1000, epochs=1000, step=1000, verbose=2, experiment_name=experiment_name)

    optimizer = SGD(learning_rate=0.00907, nesterov=True, momentum=0.98433)
    col_names = ["epochs", "val_accuracy", "val_loss"] 
    data_frame = pd.DataFrame(columns=col_names)
    experiment_name= "nesterov_bo_cifar_results"
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , experiment_name + ".csv"), index=False)
    evaluate_cifar_model(optimizer=optimizer, model= model, batch_size=1000, epochs=1000, step=1000, verbose=2, experiment_name=experiment_name)

    optimizer = ADES(beta_1=0.92226, beta_2=0.69285)
    col_names = ["epochs", "val_accuracy", "val_loss"] 
    data_frame = pd.DataFrame(columns=col_names)
    experiment_name= "ades_bo_cifar_results"
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , experiment_name + ".csv"), index=False)
    evaluate_cifar_model(optimizer=optimizer, model= model, batch_size=1000, epochs=1000, step=1000, verbose=2, experiment_name=experiment_name)
