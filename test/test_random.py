import utils.utilities as ut
def test_native_random():
    import random 
    random_state = random.getstate()
    random_1 = random.random()
    random.setstate(random_state)
    random_2 = random.random()
    assert random_1 == random_2

def test_numpy_random():
    import numpy as np
    random_state = np.random.get_state()
    random_1 = np.random.rand()
    np.random.set_state(random_state)
    random_2 = np.random.rand()
    assert random_1 == random_2

def test_engine_random():
    import sge
    params = {
        "POPSIZE": 10,
        "GENERATIONS": 10,
        "ELITISM": 0,   
        "SEED": 0,                
        "PROB_CROSSOVER": 0.0,
        "PROB_MUTATION": 0.1,
        "TSIZE": 3,
        "GRAMMAR": 'grammars/adaptive_autolr_grammar_mutate_level.txt',
        "EXPERIMENT_NAME": 'dumps/test_engine_random',
        "RUN": 1,
        "INCLUDE_GENOTYPE": True,
        "SAVE_STEP": 1,
        "VERBOSE": True,
        "EPOCHS": 2,
        "MODEL": 'models/mnist_model.h5',
        "VALIDATION_SIZE": 10,
        "FITNESS_SIZE": 59590,
        "BATCH_SIZE": 5,
        "MIN_TREE_DEPTH": 6,
        "MAX_TREE_DEPTH": 17,
        "FITNESS_FLOOR": 0,
        "PREPOPULATE": False,
        "FAKE_FITNESS": True,
    }
    pop1 = sge.evolutionary_algorithm(parameters=params, evaluation_function=None)
    params["RUN"] = 2
    pop2 = sge.evolutionary_algorithm(parameters=params, evaluation_function=None)
    ut.delete_directory(params['EXPERIMENT_NAME'], ['run_1', 'run_2'])
    assert pop1 == pop2


def test_engine_resume():
    import sge
    params = {
        "POPSIZE": 10,
        "GENERATIONS": 3,
        "ELITISM": 0,   
        "PROB_CROSSOVER": 0.8,
        "PROB_MUTATION": 0.8,
        "TSIZE": 3,
        "GRAMMAR": 'grammars/adaptive_autolr_grammar_mutate_level.txt',
        "EXPERIMENT_NAME": 'dumps/test_engine_resume',
        "RUN": 1,
        "SAVE_STEP": 1,
        "VERBOSE": True,
        "MIN_TREE_DEPTH": 6,
        "MAX_TREE_DEPTH": 17,
        "FITNESS_FLOOR": 0,
        "FAKE_FITNESS": True,
        "SEED": 0,
    }
    pop1 = sge.evolutionary_algorithm(parameters=params, evaluation_function=None)
    params['RESUME'] = 1
    params['LOAD_ARCHIVE'] = True
    pop2 = sge.evolutionary_algorithm(parameters=params, evaluation_function=None)
    ut.delete_directory(params['EXPERIMENT_NAME'], 'run_1')
    assert pop1 == pop2    
    

#if __name__ == "__main__":
#    test_engine_resume()

