import csv
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.datasets import fashion_mnist
from keras import backend as K
import numpy as np
import datetime
import json
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
session = tf.compat.v1.Session(config=config)
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
        content = tf.image.resize_images(content, shape, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    content = tf.image.grayscale_to_rgb(content)
    print(content)
    content = content.eval(session=session)
    session.close()
    tf.reset_default_graph()
    tf.keras.backend.clear_session()
    
    return content

def load_dataset(n_classes=10, validation_size=1500):
        #Confirmar mnist
        #(x_train, y_train), (x_test, y_test) = converted_data
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                          test_size=validation_size,
                                                          stratify=y_train)


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

        dataset = { 'x_train': x_train[:10000],
            #'x_train': x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1),
                    'y_train': y_train[:10000],
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

#UTILIZAR FASHION MNIST
dataset = load_dataset()

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)

datagen_train.fit(dataset['x_train'])



def train_model(phen):
    model = load_model('examples/models/model_7_0.h5')
    opt = model.optimizer
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False), metrics=['accuracy'])
    #K.tensorflow_backend._get_available_gpus()
    function_string ='''
def scheduler(epoch, learning_rate):
    print('epoch: ', epoch)
    print('learning_rate: ', learning_rate)
    return ''' + phen
    exec(function_string, globals())
    #print(function_string)
    lr_schedule_callback = keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    early_stop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    score = model.fit_generator(datagen_train.flow(dataset['x_train'],
                                                       dataset['y_train'],
                                                       batch_size=batch_size),
                                    steps_per_epoch=(dataset['x_train'].shape[0] // batch_size),
                                    epochs=epochs,
                                    validation_data=(dataset['x_val'], dataset['y_val']),
                                    callbacks = [lr_schedule_callback, early_stop],
                                    verbose=1)
    K.clear_session()
    pain = {}
    for metric in score.history:
        pain[metric] = []
        for n in score.history[metric]:
            print(n)
            if type(n) == np.float32:
                n = n.item()
            pain[metric].append(n)
    return max(pain['val_accuracy']), pain
    
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
