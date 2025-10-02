from copy import deepcopy
import csv
from pickle import NONE
from utils.data_functions import load_tiny_imagenet
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from utils.smart_phenotype import readable_phenotype, smart_phenotype
from optimizers.custom_optimizer import CustomOptimizerLayerVar, CustomOptimizerArch, CustomOptimizerAggregates
import datetime
from models.keras_model_adapters.vgg16_adapter import VGG16_Interface
from models.keras_model_adapters.inceptionv3_adapter import InceptionV3_Interface
from models.keras_model_adapters.resnet_adapter import ResNet_Interface
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import backend as K

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

def train_model_tensorflow_imagenet(phen_params):
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
    
    return evaluate_model_imagenet(phen, validation_size, batch_size, epochs, patience)

def evaluate_model_imagenet(phen, validation_size, batch_size, epochs, patience, optimizer=None):
    dataset = globals()['cached_dataset'] 
    model = globals()['cached_model']
    model = tf.keras.models.clone_model(model)
    data_process = globals()['pre_process']

    if 'momentum' in phen:
        opt = CustomOptimizerAggregates(model=model, phen=phen)
    else:
        opt = CustomOptimizerLayerVar(model=model, phen=phen)  

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)
    
    score = model.fit(data_process(dataset['x_train']), dataset['y_train'],
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=(data_process(dataset['x_val']), dataset['y_val']),
        callbacks=[
            early_stop
        ])

    K.clear_session()
    results = {}
    for metric in score.history:
        results[metric] = []
        for n in score.history[metric]:
            results[metric].append(n)
            
    test_score = model.evaluate(data_process(dataset['x_test']), dataset['y_test'], verbose=2)
    results['test_score'] = test_score
    print('Test score:', test_score[0])

    return test_score[-1], results

def cache_model(params):
    if globals()['cached_model'] == None:
        if params['MODEL'] == 'VGG16':
            cache_vgg16_model(params)
        elif params['MODEL'] == 'ResNet50':
            cache_resnet_model(params)
        elif params['MODEL'] == 'InceptionV3':
            cache_inceptionv3_model(params)
        else:
            globals()['cached_model'] = load_model(params['MODEL'], compile=False)
        globals()['cached_weights'] = globals()['cached_model'].get_weights()

def cache_resnet_model(params):
    if globals()['cached_model'] == None:
        interface = ResNet_Interface(incoming_data_shape=(64,64,3))
        globals()['cached_model'] = interface.get_model()
        globals()['cached_weights'] = globals()['cached_model'].get_weights()
        globals()['pre_process'] = interface.pre_process

def cache_vgg16_model(params):
    if globals()['cached_model'] == None:
        interface = VGG16_Interface(incoming_data_shape=(64,64,3))
        globals()['cached_model'] = interface.get_model()
        globals()['cached_weights'] = globals()['cached_model'].get_weights()
        globals()['pre_process'] = interface.pre_process

def cache_inceptionv3_model(params):
    if globals()['cached_model'] == None:
        interface = InceptionV3_Interface(incoming_data_shape=(64,64,3))
        globals()['cached_model'] = interface.get_model()
        globals()['cached_weights'] = globals()['cached_model'].get_weights()
        globals()['pre_process'] = interface.pre_process

def find_params(phen_params):
    phen, params = phen_params
    #print(params['EPOCHS'])
    validation_size = params['VALIDATION_SIZE']
    fitness_size =params['FITNESS_SIZE'] 
    batch_size = params['BATCH_SIZE']
    epochs = params['EPOCHS']
    patience = params['PATIENCE']
    return phen,params,validation_size,fitness_size,batch_size,epochs,patience
