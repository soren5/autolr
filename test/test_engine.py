def test_engine():
    import sge.grammar as grammar
    import sge
    params = {
       "POPSIZE": 50,
        "GENERATIONS": 50,
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
        "GRAMMAR": 'grammars/adaptive_autolr_grammar_mutate_level.txt',
        "EXPERIMENT_NAME": 'dumps/test_engine',
        "RUN": 0,
        "INCLUDE_GENOTYPE": True,
        "SAVE_STEP": 1,
        "VERBOSE": True,
        "MIN_TREE_DEPTH": 6,
        "MAX_TREE_DEPTH": 17,
        "PREPOPULATE": False,
        "FAKE_FITNESS": True
    }
    sge.evolutionary_algorithm(parameters=params)

def test_short_run():
    import sge.grammar as grammar
    import sge
    from main import LROptimizer
    params = {
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
        "GRAMMAR": 'grammars/adaptive_autolr_grammar_mutate_level.txt',
        "EXPERIMENT_NAME": 'dumps/test_engine',
        "RUN": 1,
        "INCLUDE_GENOTYPE": True,
        "SAVE_STEP": 1,
        "VERBOSE": True,
        "EPOCHS": 2,
        "MODEL": 'models/mnist_model.h5',
        "VALIDATION_SIZE": 10,
        "TEST_SIZE": 59590,
        "BATCH_SIZE": 5,
        "MIN_TREE_DEPTH": 6,
        "MAX_TREE_DEPTH": 17,
        "FITNESS_FLOOR": 0,
        "PREPOPULATE": False,
    }
    evaluation_function = LROptimizer()
    sge.evolutionary_algorithm(parameters=params, evaluation_function=evaluation_function)

test_short_run()