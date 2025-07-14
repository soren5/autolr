import models.keras_model_adapter as kma
import tensorflow as tf
import numpy as np
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import backend as K
from utils.data_functions import load_tiny_imagenet
from models.keras_model_adapters.vgg16_interface import VGG16_Interface
from models.keras_model_adapters.inceptionv3_interface import InceptionV3_Interface
from models.keras_model_adapters.resnet_interface import ResNet_Interface
from optimizers.custom_optimizer import CustomOptimizerArch, CustomOptimizerArchV2

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

    #cache_resnet_model(params)
    cache_vgg16_model(params)
        
    return evaluate_model_imagenet(phen, validation_size, batch_size, epochs, patience, optimizer=optimizer)

def evaluate_model_imagenet(phen, validation_size, batch_size, epochs, patience, optimizer=None):
    dataset = globals()['cached_dataset'] 
    model = tf.keras.models.clone_model(globals()['cached_model'])

    if phen == None:
        opt = optimizer
    else:
        opt = CustomOptimizerArch(phen=phen, model=model)
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
        model = ResNet_Interface(incoming_data_shape=(64,64,3))
        globals()['cached_model'] = model.get_model()
        globals()['cached_weights'] = globals()['cached_model'].get_weights()

def cache_vgg16_model(params):
    if globals()['cached_model'] == None:
        model = VGG16_Interface(incoming_data_shape=(64,64,3))
        globals()['cached_model'] = model.get_model()
        globals()['cached_weights'] = globals()['cached_model'].get_weights()

def cache_inceptionv3_model(params):
    if globals()['cached_model'] == None:
        model = InceptionV3_Interface(incoming_data_shape=(64,64,3))
        globals()['cached_model'] = model.get_model()
        globals()['cached_weights'] = globals()['cached_model'].get_weights()

parameter_file = ''
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
from sge.parameters import load_parameters, params
phen_params = (None, params)
load_parameters(parameter_file)
train_model_tensorflow_imagenet(phen_params, optimizer)

def find_params(phen_params):
    phen, params = phen_params
    print(params['EPOCHS'])
    validation_size = params['VALIDATION_SIZE']
    fitness_size =params['FITNESS_SIZE'] 
    batch_size = params['BATCH_SIZE']
    epochs = params['EPOCHS']
    patience = params['PATIENCE']
    return phen,params,validation_size,fitness_size,batch_size,epochs,patience
