import tensorflow as tf
import numpy as np
from utils.smart_phenotype import readable_phenotype
from optimizers.custom_optimizer import CustomOptimizerArch, CustomOptimizerArchV2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import math
from sge.parameters import params
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

model = Sequential()
model.add(Dense(8, input_dim=2))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.save_weights('models/xor_model.h5')

def xor_check(phen):
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


    opt = CustomOptimizerArchV2(model=model, phen=phen)
    #opt = Adam()

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
    return clear
