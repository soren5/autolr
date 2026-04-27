import optuna

from fitness_functions.fitness_functions import Optimizer_Evaluator_FMNIST_CIFAR10_TIN
import pandas as pd
import os
from evaluators.evaluate_rastringin import Rastringin_Evaluator
from evaluators.evaluate_fmnist import FMNIST_Evaluator
import numpy as np

from sge.operators.mutation import mutate_one
from sge.parameters import reset_parameters, manual_load_parameters, params
import yaml
from sge.grammar import grammar
from utils.smart_phenotype import readable_phenotype, smart_phenotype, advanced_readable_phenotype
import seaborn as sns
import matplotlib.pyplot as plt
import ast
# Load the archive from a pickle file if it exists
import os
import pickle
from tensorflow.keras.optimizers import Adam

reset_parameters()
grammar._reset_grammar()
with open("parameters/multi_task.yml", 'r') as ymlfile:
    parameters = yaml.load(ymlfile, Loader=yaml.FullLoader)
    parameters['EXPERIMENT_NAME'] = "adam_optimizer_multi_task_test"
manual_load_parameters(parameters=parameters)


fitness_function = Optimizer_Evaluator_FMNIST_CIFAR10_TIN(params)

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    beta_1 = trial.suggest_float("beta_1", 0.8, 0.9999, log=True)
    beta_2 = trial.suggest_float("beta_2", 0.8, 0.9999, log=True)
    epsilon = trial.suggest_float("epsilon", 1e-8, 1e-1, log=True)
    # Use adam with parameters for resnet on imagenet as an example
    optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    optimizer.name = "Adam"
    fitness = fitness_function.evaluate(None, params, opt=optimizer)
    print(f"Fitness: {fitness}, Learning Rate: {learning_rate}, Beta 1: {beta_1}, Beta 2: {beta_2}, Epsilon: {epsilon}")
    return fitness

study = optuna.create_study(direction='minimize')

study.optimize(objective, n_trials=100)
