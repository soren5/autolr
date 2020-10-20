import csv
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import backend as K
from utils.custom_optimizer import CustomOptimizer
from utils.data_functions import load_dataset

from sge.parameters import (
    params,
    set_parameters
)

import numpy as np
import datetime

experiment_time = datetime.datetime.now()

validation_size = params['VALIDATION_SIZE']
test_size = params['TEST_SIZE'] 
batch_size = params['BATCH_SIZE']
epochs = params['EPOCHS']
img_rows, img_cols = 28, 28

dataset = load_dataset(validation_size=validation_size, test_size=test_size, split=True, img_size = [img_rows, img_cols])

datagen_test = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

for train_data in dataset['x_train']:
    datagen_test.fit(train_data)

model = load_model(params['MODEL'], compile=False)
weights = model.get_weights()

def train_model(phen):
    input_shape = (1, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (img_rows, img_cols, 1)
    num_classes = 10

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
        early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
        score = model.fit(dataset['x_train'][i], dataset['y_train'][i],
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
        test_score = model.evaluate(x=datagen_test.flow(dataset['x_test'], dataset['y_test'], batch_size=batch_size), verbose=0, callbacks=[keras.callbacks.History()])

        print("trial ", i, ": ", test_score[-1])
        if test_score[-1] < final_score:
            final_score = test_score[-1]
            final_info = results
        
        if test_score[-1] < 0.8:
            break
    print("final fitness: ", final_score)
    return final_score, results