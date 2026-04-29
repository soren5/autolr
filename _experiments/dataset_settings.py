import logging
from itertools import product

import yaml
from evaluators.evaluate_fmnist import FMNIST_Evaluator
from evaluators.evaluate_cifar10 import CIFAR10_Evaluator
from evaluators.evaluate_tiny_imagenet import TINY_IMAGENET_Evaluator
from evaluators.evaluate_cifar100 import CIFAR100_Evaluator
from sge.parameters import reset_parameters, manual_load_parameters, params
from tensorflow.keras.optimizers import Adam

configurations = [
    #{'task': 'FMNIST', "pre_process": True},
    #{'task': 'FMNIST', "pre_process": False},

    #{'task': 'CIFAR10', "pre_process": True},
    #{'task': 'CIFAR10', "pre_process": False},

    {'task': 'CIFAR100', 'pre_process': True},
    #{'task': 'CIFAR100', 'pre_process': False},

    #{'task': 'Tiny-Imagenet', "pre_process": True},
    #{'task': 'Tiny-Imagenet', "pre_process": False},
]
num_repetitions = 15


def main(i):
        params['FAKE_FITNESS'] = False
        for run in range(i, i + num_repetitions):
            for config in configurations:
                with open("parameters/multi_task.yml", 'r') as ymlfile:
                    parameters = yaml.load(ymlfile, Loader=yaml.FullLoader)
                    parameters['EXPERIMENT_NAME'] = "adam_optimizer_cifar100_test_using_dataset_settings"
                    reset_parameters()
                    manual_load_parameters(parameters)
                if config['pre_process']:
                    params['FMNIST_CONFIG'] = 'parameters/dataset_parameters/FMNIST_CONFIG.json'
                    params['CIFAR10_CONFIG'] = 'parameters/dataset_parameters/CIFAR10_CONFIG.json'
                    params['CIFAR10_CONFIG'] = 'parameters/dataset_parameters/CIFAR100_CONFIG.json'
                    params['TINY_IMAGENET_CONFIG'] = 'parameters/dataset_parameters/TINY_IMAGENET_CONFIG.json'
                else:
                    params['FMNIST_CONFIG'] = 'parameters/dataset_parameters/FMNIST_CONFIG_all_false.json'
                    params['CIFAR10_CONFIG'] = 'parameters/dataset_parameters/CIFAR10_CONFIG_all_false.json'
                    params['TINY_IMAGENET_CONFIG'] = 'parameters/dataset_parameters/TINY_IMAGENET_CONFIG_all_false.json'

                print(params)
                if config['task'] == 'Tiny-Imagenet':
                    evaluator = TINY_IMAGENET_Evaluator(params, task_name=f"{config['task'].lower()}_{config['pre_process']}")
                elif config['task'] == 'CIFAR1000':
                    evaluator = CIFAR100_Evaluator(params, task_name=f"{config['task'].lower()}_{config['pre_process']}")
                elif config['task'] == 'CIFAR10':
                    evaluator = CIFAR10_Evaluator(params, task_name=f"{config['task'].lower()}_{config['pre_process']}")
                elif config['task'] == 'FMNIST':
                    evaluator = FMNIST_Evaluator(params, task_name=f"{config['task'].lower()}_{config['pre_process']}")
                optimizer = Adam()
                optimizer.name = f'Adam + {config["task"]} + {config["pre_process"]}'

                evaluator.run = run
                evaluator.evaluate_optimizer(optimizer)

if __name__ == '__main__':
    # Get the first argument passed on to the script and use it as the starting number for runs
    import sys
    if len(sys.argv) > 1:
        i = int(sys.argv[1])
    else:        
        i = 0
    main(i) 