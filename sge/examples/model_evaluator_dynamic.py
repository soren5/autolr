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

# Create a new input layer to replace the (None,None,None,3) input layer :
##print(model.summary())
#model.save("reshaped_model.h5")
#coreml_model = coremltools.converters.keras.convert('reshaped_model.h5')
#coreml_model.save('MyPredictor.mlmodel')

batch_size = 1000
epochs = 100

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
    session = tf.compat.v1.Session()
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


        x_train = resize_data((x_train, (32,32)))
        x_val = resize_data((x_val, (32,32)))
        x_test = resize_data((x_test, (32,32)))

        y_train = keras.utils.to_categorical(y_train, n_classes)
        y_val = keras.utils.to_categorical(y_val, n_classes)
        y_test = keras.utils.to_categorical(y_test, n_classes)

        dataset = { 'x_train': x_train,
            #'x_train': x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1),
                    'y_train': y_train,
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

datagen_train.fit(dataset['x_train'])
datagen_test.fit(dataset['x_train'])



def fake_train_model(phen):
    training_epoch = int(random.uniform(0, 100))
    pain = {
        'val_loss': [random.uniform(0, 20) for i in range(training_epoch)],
        'val_acc': [random.uniform(0, 1) for i in range(training_epoch)]
    }
    test_score = random.uniform(0,1)
    return test_score, pain


def train_model(phen):
    model = load_model('examples/models/model_7_0.h5')
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False), metrics=['accuracy'])
    function_string ='''
def scheduler(epoch, learning_rate):
    print('epoch: ', epoch)
    print('learning_rate: ', learning_rate)
    return ''' + phen
    exec(function_string, globals())
    #print(function_string)
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
    pain = {}
    for metric in score.history:
        print(metric)
        pain[metric] = []
        for n in score.history[metric]:
            print(n)
            if type(n) == np.float32:
                n = n.item()
            pain[metric].append(n)
    test_score = model.evaluate(x=datagen_test.flow(dataset['x_test'], dataset['y_test'], batch_size=batch_size), callbacks=[keras.callbacks.History()])
    print(test_score)
    return test_score, pain
    
if __name__ == "__main__":
    function_1 ='''
def scheduler(epoch):
    import math
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate'''
    function_2 = '''
def scheduler(x,y):
    return (prot_div(0.014227272727272727, 0.04752727272727273)*prot_div(0.014227272727272727, 0.04752727272727273))'''
    jingle()
    score1 = train_model(0, 0, function_1)
    jingle()
    score2 = train_model(0, 0, function_2)
    #print(score1)
    #print(score2)
    jingle()
