

import csv
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
import tensorflow as tf
import random
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import ops
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
#from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
import os
import pandas as pd

cwd_path = os.getcwd()

def load_cifar10(n_classes=10, validation_size=3500, test_size=3500):
    #Confirmar mnist
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                    test_size=validation_size + test_size,
                                                    stratify=y_train)



    img_rows, img_cols, channels = 32, 32, 3

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_val /= 255
    x_test /= 255

    #subraction of the mean image
    x_mean = 0
    for x in x_train:
        x_mean += x
    x_mean /= len(x_train)
    x_train -= x_mean
    x_val -= x_mean
    x_test -= x_mean


    # input image dimensions


    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)

    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_val = keras.utils.to_categorical(y_val, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    dataset = { 
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test,
            'y_test': y_test}

    return dataset

    def prot_div(left, right):
        if right == 0:
            return 0
        else:
            return left / right

    def if_func(condition, state1, state2):
        if condition:
            return state1
        else:
            return state2

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

def evaluate_cifar_model(dataset=None, model=None, optimizer=None, batch_size=1000, epochs=100, verbose=0):
    assert optimizer != None

    if dataset is None:
        dataset = load_cifar10()
    
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
        filepath=os.path.join(cwd_path, f'models/checkpoints/cifar_model_checkpoint.h5'),
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max')

    epoch_progress = 0

    while epoch_progress < epochs:
        if epochs - epoch_progress < 100:
            run_epochs = epochs - epoch_progress
        else:
            run_epochs = 100
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        score = model.fit(dataset['x_train'], dataset['y_train'],
            batch_size=batch_size,
            epochs=run_epochs,
            verbose=verbose,
            validation_data=(dataset['x_val'], dataset['y_val']),
            validation_steps= validation_size // batch_size,
            callbacks=[model_checkpoint_callback])
        K.clear_session()


        metric_dictionary = get_metric_dictionary(score)
        test_score = model.evaluate(dataset['x_test'], dataset['y_test'], batch_size=batch_size, verbose=verbose, callbacks=[keras.callbacks.History()])
        result = [test_score[-1], metric_dictionary]

        data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "development_results.csv"))
        col_values = [max(result[1]['val_accuracy']), min(result[1]['val_loss']), result[0]]
        col_names = ["max_val_accuracy", "min_val_loss", "test_accuracy"] 

        data_frame = data_frame.append(pd.DataFrame([col_values], columns=col_names), ignore_index=True)
        data_frame.to_csv(os.path.join(cwd_path, 'results/' , "development_results.csv"), index=False)

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

    resume = True
    if resume:
        resume_cifar_model(optimizer=Adam(), epochs=100000, verbose=2)
    else:
        col_names = ["epochs", "max_val_accuracy", "min_val_loss", "test_accuracy"] 
        data_frame = pd.DataFrame(columns=col_names)
        data_frame.to_csv(os.path.join(cwd_path, 'results/' , "development_results.csv"), index=False)
        evaluate_cifar_model(optimizer=Adam(), epochs=100000, verbose=2)