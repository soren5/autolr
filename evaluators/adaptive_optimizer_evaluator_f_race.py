import csv
from functools import cached_property
from utils.data_functions import load_fashion_mnist_training, load_cifar10_training, load_mnist_training
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

from keras.models import load_model
from tensorflow import keras
from keras import backend as K
from optimizers.custom_optimizer import CustomOptimizer
import datetime
experiment_time = datetime.datetime.now()

cached_dataset = None
cached_model = None

def train_model_tensorflow_cifar10(phen_params):
    phen, params = phen_params
    validation_size = params['VALIDATION_SIZE']
    fitness_size =params['FITNESS_SIZE'] 
    batch_size = params['BATCH_SIZE']
    epochs = params['EPOCHS']
    patience = params['PATIENCE']

    global cached_dataset
    if cached_dataset == None:
        cached_dataset = load_cifar10_training(validation_size= validation_size, test_size=fitness_size)  
    dataset = cached_dataset
    
    global cached_model
    if cached_model == None:
        if 'MODEL' in params:
            cached_model = load_model(params['MODEL'], compile=False)
        else:
            cached_model = load_model('models/mnist_model.h5')
    model = cached_model
    
    weights = model.get_weights()
    model.set_weights(weights)

    print(len(dataset['x_train']))
    
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

def train_model_tensorflow_fmnist(phen_params):
    phen, params = phen_params
    validation_size = params['VALIDATION_SIZE']
    fitness_size = params['FITNESS_SIZE']
    batch_size = params['BATCH_SIZE']
    epochs = params['EPOCHS']
    patience = params['PATIENCE']
    # print(params['EPOCHS'])
   
    global cached_dataset
    if cached_dataset == None:
        cached_dataset = load_fashion_mnist_training(validation_size= validation_size, test_size=fitness_size)  
    dataset = cached_dataset
    
    global cached_model
    if cached_model == None:
        if 'MODEL' in params:
            cached_model = load_model(params['MODEL'], compile=False)
        else:
            cached_model = load_model('models/mnist_model.h5')
    model = cached_model

    weights = model.get_weights()
    model.set_weights(weights)

    print(len(dataset['x_train']))

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

def train_model_tensorflow_mnist(phen_params):
    phen, params = phen_params
    print(params['EPOCHS'])
    validation_size = params['VALIDATION_SIZE']
    fitness_size =params['FITNESS_SIZE'] 
    batch_size = params['BATCH_SIZE']
    epochs = params['EPOCHS']
    patience = params['PATIENCE']

    global cached_dataset
    if cached_dataset == None:
        cached_dataset = load_fashion_mnist_training(validation_size= validation_size, test_size=fitness_size)  
    dataset = cached_dataset
    
    global cached_model
    if cached_model == None:
        if 'MODEL' in params:
            cached_model = load_model(params['MODEL'], compile=False)
        else:
            cached_model = load_model('models/mnist_model.h5')
    model = cached_model

    weights = model.get_weights()
    print(len(dataset['x_train']))

    model.set_weights(weights)
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