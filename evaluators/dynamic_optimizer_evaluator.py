import csv
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import backend as K
import numpy as np
import datetime
import json
import tensorflow as tf
import random
import time
from utils.data_functions import load_dataset
from sge.parameters import (
    params,
    set_parameters
)

experiment_time = datetime.datetime.now()

validation_size = params['VALIDATION_SIZE']
fitness_size = params['FITNESS_SIZE'] 
batch_size = params['BATCH_SIZE']
epochs = params['EPOCHS']
img_rows, img_cols = 28, 28

dataset = load_dataset(validation_size=validation_size, test_size=fitness_size, split=False, img_size = [img_rows, img_cols])

datagen_train = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen_test = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

datagen_train.fit(dataset['x_train'])
datagen_test.fit(dataset['x_train'])

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

model = load_model(params['MODEL'], compile=False)
weights = model.get_weights()

def train_model(phen):
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False), metrics=['accuracy'])
    scheduler = None
    function_string ='''
def scheduler(epoch, learning_rate):
    print('epoch: ', epoch)
    print('learning_rate: ', learning_rate)
    return ''' + phen
    exec(function_string, globals())

    lr_schedule_callback = keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    early_stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    score = model.fit_generator(datagen_train.flow(dataset['x_train'],
                                                       dataset['y_train'],
                                                       batch_size=batch_size),
                                    steps_per_epoch=(dataset['x_train'].shape[0] // batch_size),
                                    epochs=epochs,
                                    validation_data=(datagen_test.flow(dataset['x_val'], dataset['y_val'], batch_size=batch_size)),
                                    validation_steps = validation_size // batch_size,
                                    callbacks = [lr_schedule_callback, early_stop],
                                    verbose=1)
    K.clear_session()
    results = {}

    for metric in score.history:
        results[metric] = []
        for n in score.history[metric]:
            results[metric].append(n)
    test_score = model.evaluate(x=datagen_test.flow(dataset['x_test'], dataset['y_test'], batch_size=batch_size), callbacks=[keras.callbacks.History()])

    print(test_score)
    return test_score, results