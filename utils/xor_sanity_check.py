import tensorflow as tf
import numpy as np
from utils.smart_phenotype import readable_phenotype
from optimizers.custom_optimizer import CustomOptimizerArch

def xor_check(phen):
    #print(f"Running custom: {readable_phenotype(phen)}")
    x = np.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]], dtype=np.float32)

    y = np.array([[0],
                [1],
                [1],
                [0]], dtype=np.float32)


    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(2,)))
    model.add(tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid, kernel_initializer=tf.initializers.Constant(0.5)))
    model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))
    opt = CustomOptimizerArch(phen=phen, model=model)

    model.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=['mse', 'binary_accuracy'])
    
    history = model.fit(x, y, batch_size=1, epochs=500, verbose=0)

    predictions = model.predict_on_batch(x)
    try:
        binary_predictions = np.array([[round(pred[0])] for pred in predictions], dtype=np.float32)
        clear = (binary_predictions.astype(int) == y.astype(int)).all()
        if clear:
            print("\tPASS xor check")
        else:
            print("\tFAIL xor check")
            clear = True
    except:
        print("\t EXCEPTION xor check")
        clear = False 
    return clear

