import models.keras_model_adapter as kma
import tensorflow as tf
import numpy as np
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import backend as K
from utils.data_functions import load_tiny_imagenet
from models.vgg16_interface import VGG16_Interface

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
        
    return evaluate_model_imagenet(phen, validation_size, batch_size, epochs, patience)

def evaluate_model_imagenet(phen, validation_size, batch_size, epochs, patience):
    dataset = globals()['cached_dataset'] 
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

parameter_file = ''
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
from sge.parameters import load_parameters, params
load_parameters(parameter_file)
train_model_tensorflow_imagenet(params, optimizer)

def find_params(phen_params):
    phen, params = phen_params
    print(params['EPOCHS'])
    validation_size = params['VALIDATION_SIZE']
    fitness_size =params['FITNESS_SIZE'] 
    batch_size = params['BATCH_SIZE']
    epochs = params['EPOCHS']
    patience = params['PATIENCE']
    return phen,params,validation_size,fitness_size,batch_size,epochs,patience
