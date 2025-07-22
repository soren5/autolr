import tensorflow as tf
import numpy as np
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import backend as K
from utils.data_functions import load_tiny_imagenet
from models.keras_model_adapters.vgg16_adapter import VGG16_Interface
from models.keras_model_adapters.inceptionv3_adapter import InceptionV3_Interface
from models.keras_model_adapters.resnet_adapter import ResNet_Interface
from optimizers.custom_optimizer import CustomOptimizerArch, CustomOptimizerArchV2
from optimizers.evolved.ades import ADES
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import time
import json

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

cached_dataset = None
cached_model = None
cached_weights = None

def train_model_tensorflow_imagenet(phen_params, optimizer):
    phen, params, validation_size, fitness_size, batch_size, epochs, patience = find_params(phen_params)
    if globals()['cached_dataset'] == None:
        globals()['cached_dataset'] = load_tiny_imagenet(validation_size=validation_size, test_size=fitness_size, batch_size=batch_size)

    if params['MODEL'] == 'resnet':
        cache_resnet_model(params)
    elif params['MODEL'] == 'vgg16':
        cache_vgg16_model(params)
    elif params['MODEL'] == 'inceptionv3':
        cache_inceptionv3_model(params)
    else:
        raise Exception('Invalid model for validation')
        
    return evaluate_model_imagenet(phen, validation_size, batch_size, epochs, patience, optimizer=optimizer)

def evaluate_model_imagenet(phen, validation_size, batch_size, epochs, patience, optimizer=None):
    dataset = globals()['cached_dataset'] 
    adapter = globals()['cached_model']
    model = tf.keras.models.clone_model(adapter.get_model())
    data_process = adapter.pre_process


    if params['OPTIMIZER'] == "ADES":
        opt = ADES(model=model)
    elif params['OPTIMIZER'] == 'SGD':
        opt = SGD()
    elif params['OPTIMIZER'] == 'RMSprop':
        opt = RMSprop()
    elif params['OPTIMIZER'] == 'Adam':
        opt = Adam()
    opt.__name__ = params['OPTIMIZER']

    experiment_name = params['MODEL'] + '_' + opt.__name__ + '_' + str(time.time())

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)
    
    score = model.fit(data_process(dataset['x_train']), dataset['y_train'],
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=(data_process(dataset['x_val']), dataset['y_val']),
        validation_steps= validation_size // batch_size,
        callbacks=[
            early_stop
        ])


    #K.clear_session()
    results = {}
    for metric in score.history:
        results[metric] = []
        for n in score.history[metric]:
            results[metric].append(n)
            
    test_score = model.evaluate(data_process(dataset['x_test']), dataset['y_test'], verbose=2)
    results['test_score'] = test_score
    print(f"TEST SCORE: {test_score}")

    with open(experiment_name + ".json", 'w+') as f:
        json.dump(results, f)
    return test_score[-1], results


def cache_model(params):
    if globals()['cached_model'] == None:
        globals()['cached_model'] = load_model(params['MODEL'], compile=False)
        globals()['cached_weights'] = globals()['cached_model'].get_weights()

def cache_resnet_model(params):
    model = ResNet_Interface(incoming_data_shape=(64,64,3))
    globals()['cached_model'] = model
    globals()['cached_weights'] = globals()['cached_model'].get_model().get_weights()

def cache_vgg16_model(params):
    model = VGG16_Interface(incoming_data_shape=(64,64,3))
    globals()['cached_model'] = model
    globals()['cached_weights'] = globals()['cached_model'].get_model().get_weights()

def cache_inceptionv3_model(params):
    model = InceptionV3_Interface(incoming_data_shape=(64,64,3))
    globals()['cached_model'] = model
    globals()['cached_weights'] = globals()['cached_model'].get_model().get_weights()

def find_params(phen_params):
    phen, params = phen_params
    validation_size = params['VALIDATION_SIZE']
    fitness_size =params['FITNESS_SIZE'] 
    batch_size = params['BATCH_SIZE']
    epochs = params['EPOCHS']
    patience = params['PATIENCE']
    return phen,params,validation_size,fitness_size,batch_size,epochs,patience

parameter_file = 'parameters/imagenet.yml'
from sge.parameters import load_parameters, params
phen_params = (None, params)
load_parameters(parameter_file)

for _ in range(30):
    params['OPTIMIZER'] = 'Adam'

    params['MODEL'] = 'resnet'
    train_model_tensorflow_imagenet(phen_params, None)

    #params['MODEL'] = 'vgg16'
    #train_model_tensorflow_imagenet(phen_params, None)

    #params['MODEL'] = 'inceptionv3'
    #train_model_tensorflow_imagenet(phen_params, None)

