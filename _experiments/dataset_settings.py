import logging
from itertools import product

import yaml
from evaluators.evaluate_fmnist import FMNIST_Evaluator
from evaluators.evaluate_cifar import CIFAR10_Evaluator
from evaluators.evaluate_tiny_imagenet import TINY_IMAGENET_Evaluator
from sge.parameters import reset_parameters, manual_load_parameters, params
from tensorflow.keras.optimizers import Adam


configurations = [
    {'task': 'FMNIST', "all_false": True},
    {'task': 'FMNIST', "all_false": False},

    {'task': 'Tiny-Imagenet', "all_false": True},
    {'task': 'Tiny-Imagenet', "all_false": False},

    {'task': 'CIFAR10', "all_false": True},
    {'task': 'CIFAR10', "all_false": False},

]
num_runs = 30


def main():
    with open("parameters/multi_task.yml", 'r') as ymlfile:
        parameters = yaml.load(ymlfile, Loader=yaml.FullLoader)
        reset_parameters()
        manual_load_parameters(parameters)
        params['FAKE_FITNESS'] = False
        for config in configurations:
            if config['task'] == 'Tiny-Imagenet':
                evaluator = TINY_IMAGENET_Evaluator(params, task_name='tiny_imagenet')
            elif config['task'] == 'CIFAR10':
                evaluator = CIFAR10_Evaluator(params, task_name='cifar10')
            elif config['task'] == 'FMNIST':
                evaluator = FMNIST_Evaluator(params, task_name='fmnist')
            for run in range(num_runs):
                optimizer = Adam()
                optimizer.name = f'Adam + {config["task"]} + {config["all_false"]}'

                evaluator.evaluate_optimizer(optimizer)

if __name__ == '__main__':
    main()