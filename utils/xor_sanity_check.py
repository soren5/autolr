import gc
import math

import numpy as np
import tensorflow as tf
from keras.layers.core import Activation, Dense, Dropout
from keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

from optimizers.custom_optimizer import (CustomOptimizerAggregates,
                                         CustomOptimizerArch,
                                         CustomOptimizerLayerVar)
from sge.grammar import grammar
from sge.parameters import params
from utils.smart_phenotype import readable_phenotype


def xor_check(phen):
    model = Sequential()
    model.add(Dense(8, input_dim=2))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    print('[XOR CHECK] START')
    x = np.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]], dtype=np.float32)

    y = np.array([[0],
                [1],
                [1],
                [0]], dtype=np.float32)


    class My_Callback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs={}):
            self.epoch = epoch

        def on_batch_end(self, batch, logs={}):
            #if self.epoch % 500 == 0:
            #    print(f'[{self.epoch} {batch}]{logs}')
            if math.isnan(logs['loss']):
                print(f"NAN loss at Epoch {self.epoch}, Batch {batch}")
                self.model.stop_training = True
            if logs['binary_accuracy'] == 1.0:
                print(f"[XOR CHECK] Solved at Epoch {self.epoch}, Batch {batch}")
                self.model.stop_training = True

    if 'momentum' in phen:
        opt = CustomOptimizerAggregates(model=model, phen=phen)
    else:
        opt = CustomOptimizerLayerVar(model=model, phen=phen)
        
    model.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=['mse', 'binary_accuracy'])
    history = model.fit(x, y, batch_size=4, epochs=5000, verbose=0, callbacks=[My_Callback()])
    predictions = model.predict_on_batch(x)
    model.load_weights('models/xor_model.h5')


    try:
        binary_predictions = np.array([[round(pred[0])] for pred in predictions], dtype=np.float32)
        clear = (binary_predictions.astype(int) == y.astype(int)).all()
        if clear:
            print("[XOR CHECK] PASS")
        else:
            print("[XOR CHECK] FAIL")
            clear = params["CLEAR_XOR_CHECK"] if "CLEAR_XOR_CHECK" in params else False
    except:
        print("[XOR CHECK] EXCEPTION")
        clear = False 

    K.clear_session()
    gc.collect()
    return clear
