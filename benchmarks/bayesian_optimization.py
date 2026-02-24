from benchmarks.evaluate_{model_name}_model import evaluate_{model_name}_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import random
import os
import pandas as pd
from bayes_opt import BayesianOptimization
from optimizers.evolved.ades import ADES
import math
from utils.bayesian_optimization import *
from optimizers.custom_optimizer import CustomOptimizerArch
from sge.parameters import params
cwd_path = os.getcwd()
models_dir = params.get('MODELS_DIR', 'models')

def optimize_generic(phenotype, name, n_iter, init_points):
    constants, probes = get_constants_and_probe(phenotype)
    pbounds = {}
    pparams = {}
    i=0

    for constant, probe_value in zip(constants, probes):
        param_key = 'param_' + str(i)
        pbounds[param_key] = (0, 1)
        pparams[param_key] = probe_value
        phenotype.replace(constant, param_key, 1)
        i += 1
    f = create_evaluate_generic(phenotype, name)

    bayesian_optimizer = BayesianOptimization(f=f, pbounds=pbounds, verbose=2)
    bayesian_optimizer.probe(params=pparams)
    bayesian_optimizer.maximize(init_points=init_points, n_iter=n_iter)

def create_evaluate_generic(phenotype, optimizer_name, model_name, evaluate_model_function, optimizer_class):
    def evaluate_generic(**kwargs):
        from tensorflow.keras.models import load_model
        import tensorflow as tf
        import numpy as np
        from os.path import exists
        if not exists(os.path.join(cwd_path, 'results/' , f"{optimizer_name}_bo_{model_name}_results.csv")):
            with open(os.path.join(cwd_path, 'results/' , f"{optimizer_name}_bo_{model_name}_results.csv"), 'a') as f:
                header = "epochs,max_val_accuracy,min_val_loss,test_accuracy"
                i=0
                for key, value in kwargs.items():
                    phenotype.replace(key, f"tf.constant({value}, shape=shape, dtype=tf.float32)")
                    header += f',param_{i}'
                    i += 1
                f.write(header) 
        else:
            for key, value in kwargs.items():
                phenotype.replace(key, f"tf.constant({value}, shape=shape, dtype=tf.float32)")


        model = load_model(os.path.join(models_dir, f"{model_name}.h5"), compile=False)
        optimizer = optimizer_class(phen=phenotype, model=model)
        print("Going to evaluate")
        result = evaluate_model_function(optimizer=optimizer, model=model, verbose=0, epochs=1000, experiment_name=f'{optimizer_name}_bo_{model_name}_results')

        data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , f"{optimizer_name}_bo_{model_name}_results.csv"))
        if len(data_frame) > 1:
            total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
        else:
            total_epochs = 0

        col_values = [total_epochs + 100]
        col_names = ["epochs"] 
        for key, value in kwargs.items():
            col_names.append(key)
            col_values.append(value)
        
        data_frame.loc[len(data_frame) - 1, col_names] = col_values
        data_frame.to_csv(os.path.join(cwd_path, 'results/' , f"{optimizer_name}_bo_{model_name}_results.csv"), index=False)

        return max(result[1]['val_accuracy'])
    return evaluate_generic
