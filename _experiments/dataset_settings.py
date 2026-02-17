import logging
from itertools import product

import yaml
from evaluators.evaluate_fmnist import FMNIST_Evaluator
from evaluators.evaluate_cifar import CIFAR10_Evaluator
from evaluators.evaluate_tiny_imagenet import TINY_IMAGENET_Evaluator
from sge.parameters import reset_parameters, manual_load_parameters, params
from tensorflow.keras.optimizers import Adam


configurations = [
    #{'task': 'FMNIST', "all_false": True},
    #{'task': 'FMNIST', "all_false": False},

    #{'task': 'CIFAR10', "all_false": True},
    #{'task': 'CIFAR10', "all_false": False},

    {'task': 'Tiny-Imagenet', "all_false": True},
    {'task': 'Tiny-Imagenet', "all_false": False},
]
num_repetitions = 10


def main(i):
    with open("parameters/multi_task.yml", 'r') as ymlfile:
        parameters = yaml.load(ymlfile, Loader=yaml.FullLoader)
        reset_parameters()
        manual_load_parameters(parameters)
        params['FAKE_FITNESS'] = False
        for run in range(i, i + num_repetitions):
            for config in configurations:
                if config['task'] == 'Tiny-Imagenet':
                    evaluator = TINY_IMAGENET_Evaluator(params, task_name=f"{config['task'].lower()}_{config['all_false']}")
                elif config['task'] == 'CIFAR10':
                    evaluator = CIFAR10_Evaluator(params, task_name=f"{config['task'].lower()}_{config['all_false']}")
                elif config['task'] == 'FMNIST':
                    evaluator = FMNIST_Evaluator(params, task_name=f"{config['task'].lower()}_{config['all_false']}")
                    optimizer = Adam()
                    optimizer.name = f'Adam + {config["task"]} + {config["all_false"]}'

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