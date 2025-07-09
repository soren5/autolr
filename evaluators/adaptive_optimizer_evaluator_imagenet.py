from copy import deepcopy
import csv
from pickle import NONE
from utils.data_functions import load_fashion_mnist_training, load_cifar10_training, load_mnist_training, select_fashion_mnist_training, load_imagenet_training, load_tiny_imagenet
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from utils.smart_phenotype import readable_phenotype, smart_phenotype
from optimizers.custom_optimizer import CustomOptimizerArchV2, CustomOptimizerArch
import datetime
from models.keras_model_adapter import VGG16_Interface

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
cached_dataset = None
cached_model = None
cached_weights = None

def train_model_tensorflow_imagenet(phen_params):
    phen, params, validation_size, fitness_size, batch_size, epochs, patience = find_params(phen_params)
    if globals()['cached_dataset'] == None:
        globals()['cached_dataset'] = load_tiny_imagenet(validation_size=validation_size, test_size=fitness_size, batch_size=batch_size)

    #cache_resnet_model(params)
    cache_vgg16_model(params)
        
    return evaluate_model_imagenet(phen, validation_size, batch_size, epochs, patience)

def evaluate_model_imagenet(phen, validation_size, batch_size, epochs, patience):
    dataset = globals()['cached_dataset'] 
    dataset_2 = load_fashion_mnist_training(training_size=5000, validation_size=100, normalize=False, subtract_mean=False)
    model = tf.keras.models.clone_model(globals()['cached_model'])
    import numpy as np
    l = []
    for p in model.trainable_weights:
        l.append(K.count_params(p))
    print('Trainable params: {:,}'.format(np.sum(l)))
    
    # optimizer is constant aslong as phen doesn't changed?
    # -> opportunity to cache opt and compiled model
    #opt = CustomOptimizerArch(phen=phen, model=model)
    opt = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)
    
    score = model.fit(dataset['x_train'], dataset['y_train'],
        batch_size=100,
        epochs=100,
        verbose=2,
        validation_data=(dataset['x_val'], dataset['y_val']),
        validation_steps= validation_size // batch_size,
        callbacks=[
            #early_stop
        ])


    K.clear_session()
    results = {}
    for metric in score.history:
        results[metric] = []
        for n in score.history[metric]:
            results[metric].append(n)
    test_score = model.evaluate(dataset['x_val'], verbose=0, callbacks=[keras.callbacks.History()])
    return test_score[-1], results

def cache_model(params):
    if globals()['cached_model'] == None:
        globals()['cached_model'] = load_model(params['MODEL'], compile=False)
        globals()['cached_weights'] = globals()['cached_model'].get_weights()

def cache_resnet_model(params):
    if globals()['cached_model'] == None:
        globals()['cached_model'] = tf.keras.applications.ResNet50(weights=None, include_top=True)
        globals()['cached_weights'] = globals()['cached_model'].get_weights()

def cache_vgg16_model(params):
    if globals()['cached_model'] == None:
        vgg16 = VGG16_Interface(incoming_data_shape=(64,64,3))
        globals()['cached_model'] = vgg16.get_model()
        globals()['cached_weights'] = globals()['cached_model'].get_weights()

def create_train_model(model_, data, weights):
    def custom_evaluate_model(phen, batch_size=32, epochs=100, patience=5):
        dataset = data 
        model = model_
        model.set_weights(weights)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)
        term = tf.keras.callbacks.TerminateOnNaN()

        #for layer in model.layers:
        #    for trainable_weight in layer._trainable_weights:
        #        print(trainable_weight.name)
        # optimizer is constant aslong as phen doesn't changed?
        # -> opportunity to cache opt and compiled model


        print(f"\tRunning custom optimizer\n\t\tsmart: {smart_phenotype(phen)}\n\t\treadable:{readable_phenotype(phen)}\n\t\t{str(datetime.datetime.today())}")
        with open('log.log', 'a') as f:
            print(f"\tRunning custom optimizer\n\t\tsmart: {smart_phenotype(phen)}\n\t\treadable:{readable_phenotype(phen)}\n\t\t{str(datetime.datetime.today())}", file=f)
        opt = CustomOptimizerArchV2(phen=phen, model=model)
        #opt = Adam()
        #opt = SGD()
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        attempts = 5
        test_score = [0.1]
        while attempts > 0 and test_score[-1] < 0.5: 
            print(f"\t\tAttempt {attempts}, {test_score}")
            score = model.fit(dataset['x_train'], dataset['y_train'],
                batch_size=batch_size,
                epochs=epochs,
                verbose=0,
                validation_data=(dataset['x_val'], dataset['y_val']),
                validation_steps= len(dataset['x_val']) // batch_size,
                callbacks=[early_stop, term])
            K.clear_session()
            test_score = model.evaluate(x=dataset['x_test'],y=dataset["y_test"], verbose=0, callbacks=[keras.callbacks.History()])
            attempts -= 1
        results = {}
        for metric in score.history:
            results[metric] = []
            for n in score.history[metric]:
                results[metric].append(n)
        print(f"\t\t{str(datetime.datetime.today())} . Final Score: {test_score[-1]}")
        with open('log.log', 'a') as f:
            print(f"\t\t{str(datetime.datetime.today())} . Final Score: {test_score[-1]}", file=f)

        return test_score[-1], results

    return custom_evaluate_model

def evaluate_model(phen, validation_size, batch_size, epochs, patience):
    dataset = globals()['cached_dataset'] 
    model = tf.keras.models.clone_model(globals()['cached_model'])
    import numpy as np
    l = []
    for p in model.trainable_weights:
        l.append(K.count_params(p))
    print('Trainable params: {:,}'.format(np.sum(l)))
    
    # optimizer is constant aslong as phen doesn't changed?
    # -> opportunity to cache opt and compiled model
    #opt = CustomOptimizerArchV2(phen=phen, model=model)
    opt = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
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
