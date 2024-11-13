import utils.utilities as ut
def test_short_run():
    import sge.grammar as grammar
    import sge
    from main import Optimizer_Evaluator_Tensorflow
    from utils import create_models
    create_models.create_models()
    parameters = {
        "SELECTION_TYPE": "tournament",
        "POPSIZE": 2,
        "GENERATIONS": 2,
        "ELITISM": 0,   
        "SEED": 0,                
        "PROB_CROSSOVER": 0.0,
        "PROB_MUTATION": 0.15,
        "TSIZE": 2,
        "GRAMMAR": 'grammars/adaptive_autolr_grammar_architecture_mutate_level.txt',
        "EXPERIMENT_NAME": 'dumps/test_engine',
        "RUN": 1,
        "INCLUDE_GENOTYPE": True,
        "SAVE_STEP": 1,
        "VERBOSE": True,
        "EPOCHS": 2,
        "MODEL": 'models/mnist_model.h5',
        "VALIDATION_SIZE": 10,
        "FITNESS_SIZE": 59980,
        "BATCH_SIZE": 5,
        "MIN_TREE_DEPTH": 6,
        "MAX_TREE_DEPTH": 17,
        "FITNESS_FLOOR": 0,
        "PREPOPULATE": False,
        "PATIENCE": 0,
    }
    evaluation_function = Optimizer_Evaluator_Tensorflow()
    sge.evolutionary_algorithm(parameters=parameters, evaluation_function=evaluation_function)
    ut.delete_directory(parameters['EXPERIMENT_NAME'], "run_1")

def test_short_run_cifar10():
    import sge.grammar as grammar
    import sge
    from main import Optimizer_Evaluator_Tensorflow
    from utils import create_models
    create_models.create_models()
    parameters = {
        "SELECTION_TYPE": "tournament",
        "POPSIZE": 2,
        "GENERATIONS": 2,
        "ELITISM": 0,   
        "SEED": 0,                
        "PROB_CROSSOVER": 0.0,
        "PROB_MUTATION": {
        0: 0.0, 
        1: 0.01, 
        2: 0.01, 
        3: 0.01, 
        4: 0.05, 
        5: 0.15, 
        6: 0.01, 
        7: 0.01, 
        8: 0.01, 
        9: 0.05, 
        10: 0.15, 
        11: 0.01, 
        12: 0.01, 
        13: 0.01, 
        14: 0.05, 
        15: 0.15, 
        16: 0.01, 
        17: 0.01, 
        18: 0.05, 
        19: 0.15},
        "TSIZE": 2,
        "GRAMMAR": 'grammars/adaptive_autolr_grammar_architecture_mutate_level.txt',
        "EXPERIMENT_NAME": 'dumps/test_engine',
        "MODEL": 'models/cifar_model.h5',
        "RUN": 1,
        "INCLUDE_GENOTYPE": True,
        "SAVE_STEP": 1,
        "VERBOSE": True,
        "EPOCHS": 2,
        "VALIDATION_SIZE": 10,
        "FITNESS_SIZE": 49980,
        "BATCH_SIZE": 5,
        "MIN_TREE_DEPTH": 6,
        "MAX_TREE_DEPTH": 17,
        "FITNESS_FLOOR": 0,
        "PREPOPULATE": False,
        "PATIENCE": 0,
    }
    from evaluators.adaptive_optimizer_evaluator_f_race import train_model_tensorflow_cifar10
    evaluation_function = Optimizer_Evaluator_Tensorflow(train_model=train_model_tensorflow_cifar10)
    sge.evolutionary_algorithm(parameters=parameters, evaluation_function=evaluation_function)

def test_short_run_mnist():
    import sge.grammar as grammar
    import sge
    from main import Optimizer_Evaluator_Tensorflow
    from utils import create_models
    create_models.create_models()
    parameters = {
        "SELECTION_TYPE": "tournament",
        "POPSIZE": 2,
        "GENERATIONS": 2,
        "ELITISM": 0,   
        "SEED": 0,                
        "PROB_CROSSOVER": 0.0,
        "PROB_MUTATION": {
        0: 0.0, 
        1: 0.01, 
        2: 0.01, 
        3: 0.01, 
        4: 0.05, 
        5: 0.15, 
        6: 0.01, 
        7: 0.01, 
        8: 0.01, 
        9: 0.05, 
        10: 0.15, 
        11: 0.01, 
        12: 0.01, 
        13: 0.01, 
        14: 0.05, 
        15: 0.15, 
        16: 0.01, 
        17: 0.01, 
        18: 0.05, 
        19: 0.15},
        "TSIZE": 2,
        "GRAMMAR": 'grammars/adaptive_autolr_grammar_architecture_mutate_level.txt',
        "MODEL": 'models/mnist_model.h5',
        "EXPERIMENT_NAME": 'dumps/test_engine',
        "RUN": 1,
        "INCLUDE_GENOTYPE": True,
        "SAVE_STEP": 1,
        "VERBOSE": True,
        "EPOCHS": 2,
        "VALIDATION_SIZE": 10,
        "FITNESS_SIZE": 59980,
        "BATCH_SIZE": 5,
        "MIN_TREE_DEPTH": 6,
        "MAX_TREE_DEPTH": 17,
        "FITNESS_FLOOR": 0,
        "PREPOPULATE": False,
        "PATIENCE": 0,
    }
    from evaluators.adaptive_optimizer_evaluator_f_race import train_model_tensorflow_mnist
    evaluation_function = Optimizer_Evaluator_Tensorflow(train_model=train_model_tensorflow_mnist)
    sge.evolutionary_algorithm(parameters=parameters, evaluation_function=evaluation_function)

