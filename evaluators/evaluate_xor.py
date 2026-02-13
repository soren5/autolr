import keras
import tensorflow as tf
from dataset_loaders.mnist import MNIST_Dataset
from evaluators.evaluator_utils import Evaluator
import numpy as np
import random
from utils.smart_phenotype import smart_phenotype, readable_phenotype

from optimizers.custom_optimizer import CustomOptimizer


class XOR_Evaluator(Evaluator):
    def __init__(self, params, configuration_file=None, task_name=None):
        super().__init__(params, configuration_file=configuration_file, task_name=task_name)

    def _init_dataset_(self, validation_size, fitness_size, run):
        class XORDataset():
            def __init__(self):
                self.x = np.array([[0, 0],
                                [0, 1],
                                [1, 0],
                                [1, 1]], dtype=np.float32)

                self.y = np.array([[0],
                                [1],
                                [1],
                                [0]], dtype=np.float32)
        self.dataset = XORDataset()
    
    def _init_model_(self, params):
        from keras.models import Sequential
        from keras.layers.core import Activation, Dense, Dropout
        model = Sequential()
        model.add(Dense(8, input_dim=2))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        self.model = model
        self.model_initial_weights = self.model.get_weights()

    def evaluate(self, phen):
        # Open (or create) logs/mnist_log.log to track the evaluation
        with open(f"logs/run_{self.run}_{self.task_name}_log.log", "a") as f:
            f.write(f"[{self.task_name} evaluate start]: Running optimizer with key {smart_phenotype(phen)}\n Full phenotype:\n {readable_phenotype(phen)}\n")
        
        # Open training log file for the evaluator, add a line with the phenotype being evaluated
        csv_log_file = f"logs/run_{self.run}_{self.task_name}_training_log.csv"
        with open(csv_log_file, "a") as f:
            f.write(f"phenotype: {smart_phenotype(phen)}\n")

        result = self.train_model(phen)
        return result

    def train_model(self, phen, fake=False, optimizer=None): 
        model = tf.keras.models.clone_model(self.model)
        optimizer = CustomOptimizer(phen=phen, model=model)
        if optimizer is not None:
            if phen != "":
                print("WARNING: Both phenotype and optimizer provided, using provided optimizer")
        else:
            optimizer = CustomOptimizer(phen=phen, model=model)
        

        model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=['mse', 'binary_accuracy'])
        
        early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=self.patience, restore_best_weights=True)
        terminate_on_nan = keras.callbacks.TerminateOnNaN()
        csv_logger = keras.callbacks.CSVLogger(self.csv_log_file, append=True)

        if fake:
            print("WARNING FAKE FITNESS IS ON " * 10)
            return random.random(), {}
        
        history = model.fit(self.dataset.x, 
                            self.dataset.y, 
                            batch_size=4, 
                            epochs=5000, 
                            verbose=0, 
                            callbacks=[
                                early_stop,
                                terminate_on_nan,
                                csv_logger
                            ])
        
        predictions = model.predict_on_batch(self.dataset.x)
        #model.load_weights('models/xor_model.h5')

        try:
            binary_predictions = np.array([[round(pred[0])] for pred in predictions], dtype=np.float32)
            clear = (binary_predictions.astype(int) == self.dataset.y.astype(int)).all()
        except:
            clear = False 
        return clear
        