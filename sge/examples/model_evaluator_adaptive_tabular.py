import csv
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import backend as K
from tests.core.test_custom_optimizer import CustomOptimizer
import numpy as np
import datetime
import json
import tensorflow as tf
import random
import time
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
validation_size = 3500
test_size = 3500


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

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
img_rows, img_cols = 28, 28

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

def load_dataset(n_classes=10, validation_size=validation_size, test_size=test_size):
        #Confirmar mnist
        #(x_train, y_train), (x_test, y_test) = converted_data
        (x_train, y_train), (_, _) = fashion_mnist.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                          test_size=validation_size + test_size,
                                                          stratify=y_train)

        x_val, x_test, y_val, y_test = train_test_split(x_val, y_val,
                                                          test_size=test_size,
                                                          stratify=y_val)




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

        #x_train = resize_data((x_train, (32,32)))
        #x_val = resize_data((x_val, (32,32)))
        #x_test = resize_data((x_test, (32,32)))
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        y_train = keras.utils.to_categorical(y_train, n_classes)
        y_val = keras.utils.to_categorical(y_val, n_classes)
        y_test = keras.utils.to_categorical(y_test, n_classes)

        train_size = len(x_train)
        x_train_list = []
        y_train_list = []

        for i in range(6):
            x_train, foo, y_train, bar = train_test_split(x_train, y_train,
                                                            test_size=train_size // 7,
                                                            stratify=y_train)
            x_train_list.append(foo)
            y_train_list.append(bar)

        x_train_list.append(x_train)
        y_train_list.append(y_train)

        dataset = { 
            'x_train': x_train_list,
            #'x_train': x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1),
                    'y_train': y_train_list,
            #'x_train': x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1),
                    'x_val': x_val,
            #        'x_val': x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1), 
                   'y_val': y_val,
                   'x_test': x_test,
            #       'x_test': x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1), 
                   'y_test': y_test}

        return dataset


#Generator from keras example
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


#UTILIZAR FASHION MNIST
dataset = load_dataset()
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)

# datagen_train.fit(dataset['x_train'])
for train_data in dataset['x_train']:
    datagen_test.fit(train_data)



def fake_train_model(phen):
    training_epoch = int(random.uniform(0, 100))
    pain = {
        'val_loss': [random.uniform(0, 20) for i in range(training_epoch)],
        'val_acc': [random.uniform(0, 1) for i in range(training_epoch)]
    }
    test_score = random.uniform(0,1)
    return test_score, pain

def train_model_lite(phen):
    model = load_model('examples/models/model_7_0.h5')
    alpha_dict = {}
    beta_dict = {}
    sigma_dict = {}
    for layer in model.layers:
        for trainable_weight in layer._trainable_weights:
            alpha_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="alpha" + trainable_weight.name, shape=trainable_weight.shape, dtype=tf.float32)
            beta_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="beta" + trainable_weight.name, shape=trainable_weight.shape, dtype=tf.float32)
            sigma_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="sigma" + trainable_weight.name, shape=trainable_weight.shape, dtype=tf.float32)
    #print(tf.executing_eagerly())
    #print(phen)
    exec(phen, globals())
    opt = CustomOptimizer(alpha=alpha_dict, alpha_func=alpha_func, beta=beta_dict, beta_func=beta_func, sigma=sigma_dict, sigma_func=sigma_func, grad_func=grad_func)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    score = model.fit_generator(datagen_train.flow(dataset['x_train'],
                                                       dataset['y_train'],
                                                       batch_size=batch_size),
                                    steps_per_epoch=(dataset['x_train'].shape[0] // batch_size),
                                    epochs=5,
                                    validation_data=(datagen_test.flow(dataset['x_val'], dataset['y_val'], batch_size=batch_size)),
                                    validation_steps = validation_size // batch_size,
                                    callbacks = [early_stop],
                                    verbose=1)
    K.clear_session()
    pain = {}
    for metric in score.history:
        print(metric)
        pain[metric] = []
        for n in score.history[metric]:
            print(n)
            if type(n) == np.float32:
                n = n.item()
            pain[metric].append(n)
    #test_score = model.evaluate(x=datagen_test.flow(dataset['x_test'], dataset['y_test'], batch_size=batch_size), callbacks=[keras.callbacks.History()])
    #print(test_score[-1], pain)


model = load_model('examples/models/my_model.h5', compile=False)
#model.save_weights('examples/models/my_model_weights.h5')
weights = model.get_weights()
def train_model(phen):

    input_shape = (1, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (img_rows, img_cols, 1)
    num_classes = 10

    final_score = 1
    final_info = None

    for i in range(5):

        #model.load_weights('examples/models/my_model_weights.h5')
        model.set_weights(weights)
        alpha_dict = {}
        beta_dict = {}
        sigma_dict = {}
        for layer in model.layers:
            for trainable_weight in layer._trainable_weights:
                alpha_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="alpha" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
                beta_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="beta" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
                sigma_dict[trainable_weight.name] = tf.Variable(np.zeros(trainable_weight.shape) , name="sigma" + trainable_weight.name[:-2], shape=trainable_weight.shape, dtype=tf.float32)
        #print(tf.executing_eagerly())
        #print(phen)
        #print(random_seed)
        #print(tf.__version__)
        #tf.random.set_seed()    
        foo = {"tf": tf}
        exec(phen, foo)
       # print(phen)
        alpha_func = foo["alpha_func"]
        beta_func = foo["beta_func"]
        sigma_func = foo["sigma_func"]
        grad_func = foo["grad_func"]
        #print(globals())
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
        test_score = model.evaluate(x=datagen_test.flow(dataset['x_test'], dataset['y_test'], batch_size=batch_size), verbose=0, callbacks=[keras.callbacks.History()])

        #print("trial ", i, ": ", test_score[-1])
        if test_score[-1] < final_score:
            final_score = test_score[-1]
            final_info = pain
        
        if test_score[-1] < 0.8:
            break
    #print("final fitness: ", final_score)
    return final_score, pain
