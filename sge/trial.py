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
import pytest
from examples.ADES import ADES
from tests.core.test_custom_optimizer import CustomOptimizer

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
validation_size = 3500
test_size = 3500


#model = load_model('examples/models/model_7_0_fashionmnist.h5')
experiment_time = datetime.datetime.now()

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


batch_size = 1000
epochs = 100
img_rows, img_cols, channels = 28, 28, 1
cifar= False
def resize_data(args):
    """
        Resize the dataset 28 x 28 datasets to 32x32

        Parameters
        ----------
        args : tuple(np.array, (int, int))
            instances, and shape of the reshaped signal

        Returns
        -------
        content : np.array
            reshaped instances
    """

    content, shape = args
    content = content.reshape(-1, 28, 28, 1)

    if shape != (28, 28):
        content = tf.image.resize(content, shape, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    content = tf.image.grayscale_to_rgb(content)
    return content

def load_dataset(n_classes=10, validation_size=validation_size, test_size=test_size, resize=False):
        #Confirmar mnist
        if cifar:
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        else:
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                          test_size=validation_size + test_size,
                                                          stratify=y_train)



        if cifar:
            img_rows, img_cols, channels = 32, 32, 3
        else:
            img_rows, img_cols, channels = 28, 28, 1

        #input scaling
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

        if resize:
            x_train = resize_data((x_train, (32,32)))
            x_val = resize_data((x_val, (32,32)))
            x_test = resize_data((x_test, (32,32)))
            

        y_train = keras.utils.to_categorical(y_train, n_classes)
        y_val = keras.utils.to_categorical(y_val, n_classes)
        y_test = keras.utils.to_categorical(y_test, n_classes)

        dataset = { 
            'x_train': x_train,
            #'x_train': x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1),
                    'y_train': y_train,
            #'x_train': x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1),
                    'x_val': x_val,
            #        'x_val': x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1), 
                   'y_val': y_val,
                   'x_test': x_test,
            #       'x_test': x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1), 
                   'y_test': y_test}

        return dataset


#UTILIZAR FASHION MNIST
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)

if cifar == False:
    model = load_model('examples/models/my_model.h5', compile=False)
    weights = model.get_weights()

def train_model(phen, use_opt=False, opt=None):
    dataset = load_dataset()

    input_shape = (1, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (img_rows, img_cols, 1)
    num_classes = 10

    model.set_weights(weights)
    alpha_dict = {}
    beta_dict = {}
    sigma_dict = {}
    for layer in model.layers:
        for trainable_weight in layer._trainable_weights:
            alpha_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="alpha" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
            beta_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="beta" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
            sigma_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="sigma" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
    #print(phen)
    #print(random_seed)
    #print(tf.__version__)
    #tf.random.set_seed()    
    #print(globals())
    if not use_opt:
        foo = {"tf": tf}
        exec(phen, foo)
        alpha_func = foo["alpha_func"]
        beta_func = foo["beta_func"]
        sigma_func = foo["sigma_func"]
        grad_func = foo["grad_func"]
        #print(globals())
        opt = CustomOptimizer(alpha=alpha_dict, alpha_func=alpha_func, beta=beta_dict, beta_func=beta_func, sigma=sigma_dict, sigma_func=sigma_func, grad_func=grad_func)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    score = model.fit(dataset['x_train'], dataset['y_train'],
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=(dataset['x_val'], dataset['y_val']),
        validation_steps= validation_size // batch_size,
        callbacks=[
        ])
    # score = model.fit_generator(datagen_train.flow(dataset['x_train'],
    #                                                    dataset['y_train'],
    #                                                    batch_size=batch_size),
    #                                 steps_per_epoch=(dataset['x_train'].shape[0] // batch_size),
    #                                 epochs=epochs,
    #                                 validation_data=(datagen_test.flow(dataset['x_val'], dataset['y_val'], batch_size=batch_size)),
    #                                 validation_steps = validation_size // batch_size,
    #                                 callbacks = [
    #                                     #early_stop
    #                                     ],
    #                                 verbose=1)
    K.clear_session()
    pain = {}
    for metric in score.history:
        #print(metric)
        pain[metric] = []
        for n in score.history[metric]:
            #print(n)
            if type(n) == np.float32:
                n = n.item()
            pain[metric].append(n)
    test_score = model.evaluate(dataset['x_test'], dataset['y_test'], batch_size=batch_size, verbose=2, callbacks=[keras.callbacks.History()])
    return test_score[-1], pain




def train_model_denser(phen, use_opt=False, opt=None):
    dataset = load_dataset(resize=True)
    model = load_model('examples/models/model_7_0.h5', compile=False)

    input_shape = (1, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (img_rows, img_cols, 1)
    num_classes = 10


    #model.load_weights('examples/models/my_model_weights.h5')
    alpha_dict = {}
    beta_dict = {}
    sigma_dict = {}
    for layer in model.layers:
        for trainable_weight in layer._trainable_weights:
            alpha_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="alpha" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
            beta_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="beta" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
            sigma_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="sigma" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
    #print(phen)
    #print(random_seed)
    #print(tf.__version__)
    #tf.random.set_seed()    

    #print(globals())
    if not use_opt:
        foo = {"tf": tf}
        exec(phen, foo)
        alpha_func = foo["alpha_func"]
        beta_func = foo["beta_func"]
        sigma_func = foo["sigma_func"]
        grad_func = foo["grad_func"]
        opt = CustomOptimizer(alpha=alpha_dict, alpha_func=alpha_func, beta=beta_dict, beta_func=beta_func, sigma=sigma_dict, sigma_func=sigma_func, grad_func=grad_func)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    score = model.fit(dataset['x_train'], dataset['y_train'],
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=(dataset['x_val'], dataset['y_val']),
        validation_steps= validation_size // batch_size,
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN()
        ])
    # score = model.fit_generator(datagen_train.flow(dataset['x_train'],
    #                                                    dataset['y_train'],
    #                                                    batch_size=batch_size),
    #                                 steps_per_epoch=(dataset['x_train'].shape[0] // batch_size),
    #                                 epochs=epochs,
    #                                 validation_data=(datagen_test.flow(dataset['x_val'], dataset['y_val'], batch_size=batch_size)),
    #                                 validation_steps = validation_size // batch_size,
    #                                 callbacks = [
    #                                     #early_stop
    #                                     ],
    #                                 verbose=1)
    K.clear_session()
    pain = {}
    for metric in score.history:
        #print(metric)
        pain[metric] = []
        for n in score.history[metric]:
            #print(n)
            if type(n) == np.float32:
                n = n.item()
            pain[metric].append(n)
    test_score = model.evaluate(dataset['x_test'], dataset['y_test'], batch_size=batch_size, verbose=0, callbacks=[keras.callbacks.History()])
    return test_score[-1], pain

def train_model_cifar(phen, use_opt=False, opt=None):
    dataset = load_dataset()

    input_shape = (1, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (img_rows, img_cols, 1)
    num_classes = 10

    n_model = load_model('examples/models/cifar_model.h5', compile=False)
    model = n_model

    #model.load_weights('examples/models/my_model_weights.h5')
    #model.set_weights(weights)
    alpha_dict = {}
    beta_dict = {}
    sigma_dict = {}
    for layer in model.layers:
        for trainable_weight in layer._trainable_weights:
            alpha_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="alpha" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
            beta_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="beta" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
            sigma_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="sigma" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
    #print(phen)
    #print(random_seed)
    #print(tf.__version__)
    #tf.random.set_seed()    
    if not use_opt:
        foo = {"tf": tf}
        exec(phen, foo)
        alpha_func = foo["alpha_func"]
        beta_func = foo["beta_func"]
        sigma_func = foo["sigma_func"]
        grad_func = foo["grad_func"]
        #print(globals())
        opt = CustomOptimizer(alpha=alpha_dict, alpha_func=alpha_func, beta=beta_dict, beta_func=beta_func, sigma=sigma_dict, sigma_func=sigma_func, grad_func=grad_func)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    score = model.fit(dataset['x_train'], dataset['y_train'],
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(dataset['x_val'], dataset['y_val']),
        validation_steps= validation_size // batch_size,
        callbacks=[
        ])
    # score = model.fit_generator(datagen_train.flow(dataset['x_train'],
    #                                                    dataset['y_train'],
    #                                                    batch_size=batch_size),
    #                                 steps_per_epoch=(dataset['x_train'].shape[0] // batch_size),
    #                                 epochs=epochs,
    #                                 validation_data=(datagen_test.flow(dataset['x_val'], dataset['y_val'], batch_size=batch_size)),
    #                                 validation_steps = validation_size // batch_size,
    #                                 callbacks = [
    #                                     #early_stop
    #                                     ],
    #                                 verbose=1)
    K.clear_session()
    pain = {}
    for metric in score.history:
        #print(metric)
        pain[metric] = []
        for n in score.history[metric]:
            #print(n)
            if type(n) == np.float32:
                n = n.item()
            pain[metric].append(n)
    test_score = model.evaluate(dataset['x_test'], dataset['y_test'], batch_size=batch_size, verbose=2, callbacks=[keras.callbacks.History()])
    return test_score[-1], pain
#from bayes_opt import BayesianOptimization

ades = 'alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.add(alpha, tf.math.multiply(grad, tf.math.add(alpha, tf.constant(7.67413430e-04, shape=shape, dtype=tf.float32)))), lambda shape,  alpha, beta, grad: tf.math.multiply(tf.constant(8.92170603e-02, shape=shape, dtype=tf.float32), tf.math.add(beta, tf.math.multiply(tf.math.add(beta, tf.constant(9.99060945e-01, shape=shape, dtype=tf.float32)), tf.math.add(beta, grad)))), lambda shape,  alpha, beta, sigma, grad: tf.math.add(sigma, grad), lambda shape,  alpha, beta, sigma, grad: tf.math.negative(beta)'
sign = 'alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.pow(tf.math.add(alpha, alpha), tf.math.add(alpha, grad)), lambda shape,  alpha, beta, grad: tf.math.add(beta, grad), lambda shape,  alpha, beta, sigma, grad: sigma, lambda shape,  alpha, beta, sigma, grad: tf.math.multiply(tf.math.square(tf.constant(2.47663801e-01, shape=shape, dtype=tf.float32)), tf.math.divide_no_nan(tf.math.sqrt(tf.constant(1.86692945e-04, shape=shape, dtype=tf.float32)), tf.math.divide_no_nan(tf.math.negative(beta), tf.math.sqrt(tf.math.square(beta)))))'

for number in range(30):
    test_score, val_score = train_model('',
    use_opt=True,
    opt=SGD(momentum=0.9, nesterov=True))
    f = open("record.txt", "a")
    f.write('nesterov ' + str(number) + '\n')
    f.write('val test' + '\n')
    f.write(str( max(val_score['val_accuracy'])) + ', ' + str(test_score) + '\n')
    f.close()
# pbounds = {'lr': (0, 0.1), 'beta1': (0.9, 1), 'beta2': (0.9, 1)}

# def foo(lr, beta1, beta2):

# bae = BayesianOptimization(f=foo, pbounds = pbounds, verbose=0, random_state=1)
# bae.maximize(init_points=1, n_iter=1)
# print(bae.max)
# pbounds = {'lr': (0, 0.1), 'beta1': (0.9, 1), 'beta2': (0.9, 1)}

# def foo(lr, beta1, beta2):
#     return train_model_cifar('',
#     use_opt=True,
#     opt=Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2))[0]

# bae = BayesianOptimization(f=foo, pbounds = pbounds, verbose=0, random_state=1)
# bae.maximize(init_points=100, n_iter=5)
# print(bae.max)

# pbounds = {'lr': (0, 0.1), 'momentum': (0.9, 1)}
# def foo(lr, momentum):
#     return train_model_cifar('',
#     use_opt=True,
#     opt=SGD(learning_rate=lr, momentum=momentum, nesterov=True))[0]

# bae = BayesianOptimization(f=foo, pbounds = pbounds, verbose=0, random_state=1)
# bae.maximize(init_points=100, n_iter=5)
# print(bae.max)

# pbounds = {'lr': (0, 0.1), 'rho': (0.9, 1)}
# def foo(lr, rho):
#     return train_model_cifar('',
#     use_opt=True,
#     opt=RMSprop(learning_rate=lr, rho=rho))[0]

# bae = BayesianOptimization(f=foo, pbounds = pbounds, verbose=0, random_state=1)
# bae.maximize(init_points=100, n_iter=5)
# print(bae.max)

# pbounds = {'beta1': (0.9, 1), 'beta2': (0.9, 1)}
# def foo(beta1, beta2):
#     return train_model_cifar('',
#     use_opt=True,
#     opt=ADES(beta1 = beta1, beta2=beta2))[0]

# bae = BayesianOptimization(f=foo, pbounds = pbounds, verbose=2, random_state=1)
# bae.maximize(init_points=100, n_iter=5)
# print(bae.max)
