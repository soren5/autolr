def test_pytorch():
    import sge
    from main import Optimizer_Evaluator_Torch
    evaluation_function = Optimizer_Evaluator_Torch()
    parameters = {
        "SELECTION_TYPE": "tournament",
        "POPSIZE": 2,
        "GENERATIONS": 2,
        "ELITISM": 0,   
        "SEED": 0,                
        "PROB_CROSSOVER": 0.0,
        "PROB_MUTATION": 0.15,
        "TSIZE": 2,
        "GRAMMAR": 'grammars/adaptive_autolr_grammar_torch.txt',
        "EXPERIMENT_NAME": 'dumps/test_torch',
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
    sge.evolutionary_algorithm(parameters=parameters, evaluation_function=evaluation_function)
