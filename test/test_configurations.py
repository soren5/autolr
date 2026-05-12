import utils.utilities as ut
import pytest
import yaml
from utils import create_models

@pytest.fixture
def base_fixture():
    from sge.parameters import reset_parameters
    from sge.grammar import grammar 

    reset_parameters()
    grammar._reset_grammar()

def test_multi_task(base_fixture):
    # python -m main --parameters=parameters/multi_task.yml --fake=true
    create_models.create_models()

    with open("parameters/multi_task.yml", 'r') as ymlfile:
        parameters = yaml.load(ymlfile, Loader=yaml.FullLoader)
    parameters['GENERATIONS'] = 3
    parameters['POPSIZE'] = 3
    parameters['FAKE_FITNESS'] = True
    run_parameters_multi_task(parameters)

def test_deep_architecture_optimizer_with_aggregators(base_fixture):
    # python -m main --parameters=parameters/deep_architecture_optimizer_with_aggregators.yml 
    # --fake=true 
    create_models.create_models()

    with open("parameters/deep_architecture_optimizer_with_aggregators.yml", 'r') as ymlfile:
        parameters = yaml.load(ymlfile, Loader=yaml.FullLoader)
    parameters['GENERATIONS'] = 3
    parameters['POPSIZE'] = 3
    parameters['FAKE_FITNESS'] = True
    run_parameters(parameters)


def test_deep_architecture_optimizer(base_fixture):
    # python -m main --parameters=parameters/deep_architecture_optimizer.yml 
    # --fake=true 
    create_models.create_models()

    with open("parameters/deep_architecture_optimizer.yml", 'r') as ymlfile:
        parameters = yaml.load(ymlfile, Loader=yaml.FullLoader)
    parameters['GENERATIONS'] = 3
    parameters['POPSIZE'] = 3
    parameters['FAKE_FITNESS'] = True
    run_parameters(parameters)

def test_basic_architecture_optimizer(base_fixture):
    # python -m main --parameters=parameters/basic_architecture_optimizer.yml 
    # --fake=true 
    create_models.create_models()

    with open("parameters/basic_architecture_optimizer.yml", 'r') as ymlfile:
        parameters = yaml.load(ymlfile, Loader=yaml.FullLoader)
    parameters['GENERATIONS'] = 3
    parameters['POPSIZE'] = 3
    parameters['FAKE_FITNESS'] = True
    run_parameters(parameters)

def test_base(base_fixture):
    # python -m main --parameters=parameters/base.yml 
    # --fake=true 
    create_models.create_models()

    with open("parameters/base.yml", 'r') as ymlfile:
        parameters = yaml.load(ymlfile, Loader=yaml.FullLoader)
    parameters['GENERATIONS'] = 3
    parameters['POPSIZE'] = 5
    parameters['FAKE_FITNESS'] = True
    run_parameters(parameters)

def run_parameters(parameters):
    from sge.parameters import manual_load_parameters, params
    from fitness_functions.fitness_functions import Optimizer_Evaluator_Tensorflow
    import sge

    manual_load_parameters(parameters=parameters)

    evaluation_function = Optimizer_Evaluator_Tensorflow(params)

    sge.evolutionary_algorithm(parameters=parameters, evaluation_function=evaluation_function)

def run_parameters_multi_task(parameters):
    from sge.parameters import manual_load_parameters, params
    from fitness_functions.fitness_functions import Optimizer_Evaluator_FMNIST_CIFAR10_TIN
    import sge

    manual_load_parameters(parameters=parameters)

    evaluation_function = Optimizer_Evaluator_FMNIST_CIFAR10_TIN(params)

    sge.evolutionary_algorithm(parameters=parameters, evaluation_function=evaluation_function)

if __name__ == '__main__':
    #test_multi_task(base_fixture)
    #test_deep_architecture_optimizer_with_aggregators(base_fixture)
    #test_deep_architecture_optimizer(base_fixture)
    #test_basic_architecture_optimizer(base_fixture)
    test_base(base_fixture)