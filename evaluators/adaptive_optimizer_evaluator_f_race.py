from copy import deepcopy
import csv
from pickle import NONE
from utils.data_functions import load_fashion_mnist_training, load_cifar10_training, load_mnist_training, select_fashion_mnist_training
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

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

from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import backend as K
from optimizers.custom_optimizer import CustomOptimizerArch
import datetime
experiment_time = datetime.datetime.now()

cached_dataset = None
cached_model = None
cached_weights = None

def train_model_tensorflow_cifar10(phen_params):
    phen, params, validation_size, fitness_size, batch_size, epochs, patience = find_params(phen_params)


    # Note that globals are borderline -- consider an object or a closure 
    # deliberately using globals() to make it ugly...
    if globals()['cached_dataset'] == None:
        globals()['cached_dataset'] = load_cifar10_training(validation_size=validation_size, test_size=fitness_size)
    
    cache_model(params)
        
    return evaluate_model(phen, validation_size, batch_size, epochs, patience)

def train_model_tensorflow_fmnist(phen_params):
    phen, params, validation_size, fitness_size, batch_size, epochs, patience = find_params(phen_params)

    # Note that globals are borderline -- consider an object or a closure 
    # deliberately using globals() to make it ugly...
    if globals()['cached_dataset'] == None:
        globals()['cached_dataset'] = load_fashion_mnist_training(validation_size=validation_size, test_size=fitness_size)
    
    cache_model(params)
       
    return evaluate_model(phen, validation_size, batch_size, epochs, patience)


def train_model_tensorflow_mnist(phen_params):
    phen, params, validation_size, fitness_size, batch_size, epochs, patience = find_params(phen_params)

    # Note that globals are borderline -- consider an object or a closure 
    # deliberately using globals() to make it ugly...
    if globals()['cached_dataset'] == None:
        globals()['cached_dataset'] = load_mnist_training(validation_size=validation_size, test_size=fitness_size)
    
    cache_model(params)
        
    # we assume validation and test sets are deterministic
    return evaluate_model(phen, validation_size, batch_size, epochs, patience)

def cache_model(params):
    if globals()['cached_model'] == None:
        globals()['cached_model'] = load_model(params['MODEL'], compile=False)
        globals()['cached_weights'] = globals()['cached_model'].get_weights()

def create_train_model(model_, data, weights):
    def custom_evaluate_model(phen, batch_size=32, epochs=100, patience=5):
        dataset = data 
        model = model_
        model.set_weights(weights)
        #for layer in model.layers:
        #    for trainable_weight in layer._trainable_weights:
        #        print(trainable_weight.name)
        # optimizer is constant aslong as phen doesn't changed?
        # -> opportunity to cache opt and compiled model


        print(f"Running custom: {phen}")
        opt = CustomOptimizerArch(phen=phen, model=model)
        #opt = Adam()
        #opt = SGD()
        print("Running SGD")
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        score = model.fit(dataset['x_train'], dataset['y_train'],
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=(dataset['x_val'], dataset['y_val']),
            validation_steps= len(dataset['x_val']) // batch_size,
            callbacks=[])

        K.clear_session()
        results = {}
        for metric in score.history:
            results[metric] = []
            for n in score.history[metric]:
                results[metric].append(n)
        test_score = model.evaluate(x=dataset['x_test'],y=dataset["y_test"], verbose=0, callbacks=[keras.callbacks.History()])
        return test_score[-1], results

    return custom_evaluate_model

def evaluate_model(phen, validation_size, batch_size, epochs, patience):
    dataset = globals()['cached_dataset'] 
    model = tf.keras.models.clone_model(globals()['cached_model'])
    import numpy as np
    l = []
    for p in model.trainable_weights:
        l.append(K.count_params(p))
    #non_trainable_count = int(
    #    np.sum([K.count_params(p.ref()) for p in set(model.non_trainable_weights)]))

    #print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(np.sum(l)))
    #print('Non-trainable params: {:,}'.format(non_trainable_count))
    
    # optimizer is constant aslong as phen doesn't changed?
    # -> opportunity to cache opt and compiled model
    opt = CustomOptimizer(phen=phen, model=model)
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
    test_score = model.evaluate(x=dataset['x_test'],y=dataset["y_test"], verbose=0, callbacks=[keras.callbacks.History()])
    return test_score[-1], results

def find_params(phen_params):
    phen, params = phen_params
    print(params['EPOCHS'])
    validation_size = params['VALIDATION_SIZE']
    fitness_size =params['FITNESS_SIZE'] 
    batch_size = params['BATCH_SIZE']
    epochs = params['EPOCHS']
    patience = params['PATIENCE']
    return phen,params,validation_size,fitness_size,batch_size,epochs,patience
