import math
import tensorflow as tf
from optimizers.custom_optimizer import CustomOptimizer
from tensorflow import keras
from tensorflow.keras import backend as K
from utils.smart_phenotype import smart_phenotype, readable_phenotype
import random 
import os
import datetime
from keras.models import load_model
import gc
from tensorflow.python.client import device_lib

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

class Evaluator():
    def __init__(self, params, configuration_file=None, task_name=None):
        if params['MULTI_TASK']:
            # For multi-task, we need a separate configuration for each task, check if it exists
            if configuration_file not in params:
                raise Exception(f'MULTI_TASK is set to True, but no {configuration_file} is provided in parameters')
            
            # We have the config file, everything is ok let's load it
            config_path = params[configuration_file]
        else:
            # We are in single task, no config file is needed
            config_path = None
            
        self.configuration_file = configuration_file
        self.task_name = task_name
        
        # Get the parameters from find_params, which will handle both single and multi-task cases
        validation_size, fitness_size, batch_size, epochs, patience, model_file, normalize, subtract_mean = self.find_params(config_path, params)
        
        # Store the parameters in the object for later use
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_size = validation_size

        self.run = params['RUN']
        self.fake_fitness = params['FAKE_FITNESS']



        self.log_path = os.path.join(params['LOGS_DIR'], params['EXPERIMENT_NAME']) #By default, this is autolr/logs, but it will check for an environment variable to override it, this is useful for running on the cluster to account for nfs

        self._init_dataset_(validation_size, fitness_size, self.run, normalize, subtract_mean)
        self._init_model_(os.path.join(params['MODELS_DIR'], model_file))
        self._init_logs_(params)

    def _init_model_(self, model_path):
        # This will pass if the model path is valid and the model can be loaded

        self.model = load_model(model_path, compile=False)
        self.model_initial_weights = self.model.get_weights()
 
    def _init_dataset_(self, params):
        raise NotImplementedError("This method should be implemented in the subclass")

    def _init_logs_(self, params):
        # Logs are written to two folders, self.log_path/logs (for .log files) and self.log_path/csv (for .csv files)
        # Check if these directories exist and if not, create them
        print(f"Initializing logs for {self.task_name} evaluator, log path: {self.log_path}")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(f"{self.log_path}/csv"):
            os.makedirs(f"{self.log_path}/csv")
        if not os.path.exists(f"{self.log_path}/logs"):
            os.makedirs(f"{self.log_path}/logs")
        print(f"Created log directories: {self.log_path}/logs and {self.log_path}/csv")

        # Check if log file already exists, if it does, make a copy of it with a timestamp to avoid overwriting previous logs, add a z to the name so it is clear that this log is a backup and not the current log
        if os.path.exists(f"{self.log_path}/logs/run_{self.run}_{self.task_name}_log.log"):
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            os.rename(f"{self.log_path}/logs/run_{self.run}_{self.task_name}_log.log", f"{self.log_path}/logs/z_run_{self.run}_{self.task_name}_log_{timestamp}.log")
            print(f"Backed up existing log file to: {self.log_path}/logs/z_run_{self.run}_{self.task_name}_log_{timestamp}.log")
        # Do the same for the training log file, but with a different name to avoid confusion
        if os.path.exists(f"{self.log_path}/csv/run_{self.run}_{self.task_name}_training_log.csv"):
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            os.rename(f"{self.log_path}/csv/run_{self.run}_{self.task_name}_training_log.csv", f"{self.log_path}/csv/z_run_{self.run}_{self.task_name}_training_log_{timestamp}.csv")
            print(f"Backed up existing training log file to: {self.log_path}/csv/z_run_{self.run}_{self.task_name}_training_log_{timestamp}.csv")

        with open(f"{self.log_path}/logs/run_{self.run}_{self.task_name}_log.log", "a") as f:
            f.write(f"[{self.task_name} evaluator init]: Running with parameters: {params}\n") 

    def evaluate(self, phen):

        # Open (or create) logs/mnist_log.log to track the evaluation
        with open(f"{self.log_path}/logs/run_{self.run}_{self.task_name}_log.log", "a") as f:
            f.write(f"[{self.task_name} evaluate start]: Running optimizer with key {smart_phenotype(phen)}\n Full phenotype:\n {readable_phenotype(phen)}\n")
            # Get gpu memory info and log it, this is useful to check if the gpu memory is being used correctly, and to debug out of memory errors on the cluster
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            f.write(f"[{self.task_name} evaluate start]: Memory usage (current/peak): {mem_info['current'] / 1e9:.2f} GB / {mem_info['peak'] / 1e9:.2f} GB\n")
        
        # Open training log file for the evaluator, add a line with the phenotype being evaluated
        self.csv_log_file = f"{self.log_path}/csv/run_{self.run}_{self.task_name}_training_log.csv"
        with open(self.csv_log_file, "a") as f:
            f.write(f"phenotype: {smart_phenotype(phen)}\n")

        # Run evaluation with the phenotype and collect results
        fitness, results = self.train_model(phen, 
            fake=self.fake_fitness)
        
        # Log the fitness result and return it
        with open(f"{self.log_path}/logs/run_{self.run}_{self.task_name}_log.log", "a") as f:
            f.write(f"[{self.task_name} evaluate end]: Fitness: {fitness}\n")
            # Get gpu memory info and log it, this is useful to check if the gpu memory is being used correctly, and to debug out of memory errors on the cluster
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            f.write(f"[{self.task_name} evaluate end]: Memory usage (current/peak): {mem_info['current'] / 1e9:.2f} GB / {mem_info['peak'] / 1e9:.2f} GB\n")
            f.write(f"[{self.task_name} evaluate end]: Starting cleanup to free memory for next evaluation\n")
            # Clear TensorFlow's internal state
            tf.keras.backend.clear_session()

            # Force Python garbage collection
            gc.collect()

            # Reset the GPU memory allocator
            device_lib.list_local_devices() 

            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            f.write(f"[{self.task_name} evaluate end]: Cleanup complete. Memory usage after cleanup (current/peak): {mem_info['current'] / 1e9:.2f} GB / {mem_info['peak'] / 1e9:.2f} GB\n\n")
        return fitness, results
    
    def evaluate_optimizer(self, optimizer):
        # Open (or create) logs/mnist_log.log to track the evaluation
        with open(f"{self.log_path}/logs/run_{self.run}_{self.task_name}_log.log", "a") as f:
            f.write(f"[{self.task_name} evaluate start]: Running optimizer {optimizer.name}\n")
        
        # Open training log file for the evaluator, add a line with the phenotype being evaluated
        self.csv_log_file = f"{self.log_path}/csv/run_{self.run}_{self.task_name}_training_log.csv"
        with open(self.csv_log_file, "a") as f:
            f.write(f"optimizer: {optimizer.name}\n")

        # Run evaluation with the phenotype and collect results
        fitness, results = self.train_model("",
            fake=self.fake_fitness,
            optimizer=optimizer)
        
        # Log the fitness result and return it
        with open(f"{self.log_path}/logs/run_{self.run}_{self.task_name}_log.log", "a") as f:
            f.write(f"[{self.task_name} evaluate end]: Fitness: {fitness}\n\n\n")
            
        return fitness, results

    def find_params(self, config_file, params):
        if config_file is not None:
            json_path = config_file
            with open(json_path, 'r') as f:
                import json
                print(f"Loading parameters for {self.task_name} evaluator from config file: {json_path}")
                fmnist_params = json.load(f)
                validation_size = fmnist_params['VALIDATION_SIZE']
                fitness_size = fmnist_params['FITNESS_SIZE']
                batch_size = fmnist_params['BATCH_SIZE']
                epochs = fmnist_params['EPOCHS']
                patience = fmnist_params['PATIENCE']
                model_path = fmnist_params['MODEL']
                normalize = fmnist_params['NORMALIZE']
                subtract_mean = fmnist_params['SUBTRACT_MEAN']
        else:
            print(f"Loading parameters for {self.task_name} evaluator from default parameters")
            validation_size = params['VALIDATION_SIZE']
            fitness_size =params['FITNESS_SIZE'] 
            batch_size = params['BATCH_SIZE']
            epochs = params['EPOCHS']
            patience = params['PATIENCE']
            model_path = params['MODEL']
            normalize = params['NORMALIZE']
            subtract_mean = params['SUBTRACT_MEAN']
    
            
        return validation_size, fitness_size, batch_size, epochs, patience, model_path, normalize, subtract_mean

    def collect_results(self, score, test_score):
        results = {}
        for metric in score.history:
            results[metric] = []
            for n in score.history[metric]:
                results[metric].append(n)
        results['test_score'] = test_score[-1]
        return results

    def train_model(self, phen, fake=False, optimizer=None):
        model = tf.keras.models.clone_model(self.model)

        if optimizer is not None:
            if phen != "":
                print("WARNING: Both phenotype and optimizer provided, using provided optimizer")
        else:
            optimizer = CustomOptimizer(phen=phen, model=model)
    
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=self.patience, restore_best_weights=True)
        terminate_on_nan = keras.callbacks.TerminateOnNaN()
        csv_logger = keras.callbacks.CSVLogger(self.csv_log_file, append=True)

        if fake:
            print("WARNING FAKE FITNESS IS ON " * 10)
            return random.random(), {}
        
        score = model.fit(self.dataset.x_train, self.dataset.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=0,
            validation_data=(self.dataset.x_val, self.dataset.y_val),
            validation_steps= self.validation_size // self.batch_size,
            callbacks=[
                early_stop,
                terminate_on_nan,
                csv_logger
            ])

        fitness_score = model.evaluate(x=self.dataset.x_fit,y=self.dataset.y_fit, verbose=0, callbacks=[])
        
        results = self.collect_results(score, fitness_score)
        return results['test_score'], results