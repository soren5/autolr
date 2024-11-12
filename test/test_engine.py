import utils.utilities as ut

class TensorflowFitnessGenerator:
    def __init__(self) -> None:
        self.fitness ={}
        self.populations = {}
        self.initial_populations = {}
        self.random_states = {}
        from utils.smart_phenotype import smart_phenotype
        self.smart_phenotype = smart_phenotype
        pass
    def evaluate(self, phen, parameters):
        if self.smart_phenotype(phen) in self.fitness:
            fit = self.fitness[self.smart_phenotype(phen)]
        else:
            import tensorflow 
            fit = -tensorflow.random.uniform(shape=[1])[0]
            self.fitness[self.smart_phenotype(phen)] = fit
        return float(fit), {}
    def init_net(self, parameters):
        pass
    def init_data(self, parameters):
        pass
    def init_evaluation(self, parameters):
        pass

def test_engine():

    import sge.grammar as grammar
    import sge
    parameters = {
        "SELECTION_TYPE": "tournament",       
        "POPSIZE": 10,
        "GENERATIONS": 10,
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
        "RUN": 1,
        "INCLUDE_GENOTYPE": True,
        "SAVE_STEP": 1,
        "VERBOSE": True,
        "MIN_TREE_DEPTH": 6,
        "MAX_TREE_DEPTH": 17,
        "FAKE_FITNESS": True,
        "FITNESS_FLOOR": 0,
    }
    sge.evolutionary_algorithm(parameters=parameters)
    ut.delete_directory(parameters['EXPERIMENT_NAME'], "run_1")

def test_default_parameters():
    import sge.grammar as grammar
    import sge    
    from main import Optimizer_Evaluator_Tensorflow
    from utils import create_models
    create_models.create_models()
    evaluation_function = Optimizer_Evaluator_Tensorflow()

    sge.evolutionary_algorithm(evaluation_function=evaluation_function)

def test_mutation_errors():
    import sge.grammar as grammar
    import sge
    import yaml
    from main import Optimizer_Evaluator_Tensorflow
    from utils import create_models

    create_models.create_models()
    evaluation_function = Optimizer_Evaluator_Tensorflow()

    with open("parameters/adaptive_autolr.yml", 'r') as ymlfile:
        parameters = yaml.load(ymlfile, Loader=yaml.FullLoader)
    parameters['PROB_MUTATION'] = {0: 0.2, 1:0.2}
    parameters['FAKE_FITNESS'] = True
    try:
        sge.evolutionary_algorithm(parameters=parameters, evaluation_function=evaluation_function)
    except AssertionError:
        print("Caught Invalid Size Error successfully")
    else:
        raise AssertionError("Failed to catch Invalid Size Error successfully")
    parameters['PROB_MUTATION'] = [1.0, 1.0]
    try:
        sge.evolutionary_algorithm(parameters=parameters, evaluation_function=evaluation_function)
    except Exception:
        print("Caught Invalid Mutation Type Error successfully")
    else:
        raise AssertionError("Failed to catch invalid Mutation Type Error successfully")
    ut.delete_directory(parameters['EXPERIMENT_NAME'], "run_1")

def test_parameters():
    import sge, os
    parameters = {
        "SELECTION_TYPE": "tournament",
        "POPSIZE": 10,
        "GENERATIONS": 1,
        "ELITISM": 0,   
        "PROB_CROSSOVER": 0.0,
        "PROB_MUTATION": 0.1,
        "TSIZE": 3,
        "GRAMMAR": 'grammars/adaptive_autolr_grammar_architecture_mutate_level.txt',
        "EXPERIMENT_NAME": 'test_parameter',
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
    sge.evolutionary_algorithm(parameters=parameters, evaluation_function=None)
    ut.delete_directory(parameters['EXPERIMENT_NAME'], "run_1")
    

def test_archive():
    """I devised this test to discover if there are reproducility problems with the archive.
    The only problem is if we take an archive from the future and use it in an earlier generation.
    This will not yield the same result as fitness evaluation burns random seed numbers (to map the genotype)."""
    import sge
    import tensorflow as tf
    parameters = {
        "SELECTION_TYPE": "tournament",
        "POPSIZE": 10,
        "GENERATIONS": 3,
        "ELITISM": 0,   
        "PROB_CROSSOVER": 0.0,
        "PROB_MUTATION": 0.9,
        "TSIZE": 3,
        "GRAMMAR": 'grammars/adaptive_autolr_grammar_architecture_mutate_level.txt',
        "EXPERIMENT_NAME": 'dumps/test_archive',
        "RUN": 1,
        "INCLUDE_GENOTYPE": True,
        "SAVE_STEP": 1,
        "VERBOSE": True,
        "MIN_TREE_DEPTH": 2,
        "MAX_TREE_DEPTH": 4,
        "FITNESS_FLOOR": 0,
        "SEED": 4,
    }
    from utils.smart_phenotype import smart_phenotype

    fitness = TensorflowFitnessGenerator()
    pop1 = sge.evolutionary_algorithm(parameters=parameters, evaluation_function=fitness)
    parameters['RESUME'] = 1
    parameters['LOAD_ARCHIVE'] = True
    import os
    old_path = os.path.join(parameters['EXPERIMENT_NAME'], "run_1", "z-archive_3.json")
    new_path = os.path.join(parameters['EXPERIMENT_NAME'], "run_1", "z-archive_1.json")
    os.remove(new_path)
    os.rename(old_path, new_path)
    pop2 = sge.evolutionary_algorithm(parameters=parameters, evaluation_function=fitness)
    pop3 = sge.evolutionary_algorithm(parameters=parameters, evaluation_function=fitness)
    ut.delete_directory(parameters['EXPERIMENT_NAME'], "run_1")
    assert pop3 == pop2    

def test_archive_id():
    import sge
    import tensorflow as tf
    from sge.parameters import manual_load_parameters
    parameters = {
        "SELECTION_TYPE": "tournament",
        "POPSIZE": 10,
        "GENERATIONS": 20,
        "ELITISM": 9,   
        "PROB_CROSSOVER": 0.0,
        "PROB_MUTATION": 0.001,
        "TSIZE": 3,
        "GRAMMAR": 'grammars/adaptive_autolr_grammar_architecture_mutate_level.txt',
        "EXPERIMENT_NAME": 'dumps/test_archive',
        "RUN": 1,
        "INCLUDE_GENOTYPE": True,
        "SAVE_STEP": 1,
        "VERBOSE": True,
        "MIN_TREE_DEPTH": 2,
        "MAX_TREE_DEPTH": 4,
        "FITNESS_FLOOR": 0,
        "SEED": 4,
    }

    fitness = TensorflowFitnessGenerator()

    manual_load_parameters(parameters, reset=True)
    pop1 = sge.evolutionary_algorithm(parameters=parameters, evaluation_function=fitness)

    parameters['RESUME'] = 1
    parameters['LOAD_ARCHIVE'] = True
    parameters['PARENT_EXPERIMENT'] = parameters['EXPERIMENT_NAME']
    #manual_load_parameters(parameters)

    pop2 = sge.evolutionary_algorithm(parameters=parameters, evaluation_function=fitness)
    ut.delete_directory(parameters['EXPERIMENT_NAME'], "run_1")
    pop1 = ut.prune_population(pop1)
    pop2 = ut.prune_population(pop2)

    with open("log.log", 'w') as f:
        print(ut._unidiff_output(str(pop1), str(pop2)), file=f)
    with open("log1.log", 'w') as f:
        print(str(pop1), file=f)   
    with open("log2.log", 'w') as f:
        print(str(pop2), file=f)
    #print(f"\n\n\n{len(pop1)}, {len(pop2)}")

    assert pop1 == pop2


def test_reevaluation():
    import yaml
    from sge.parameters import load_parameters
    import copy

    class FitnessEvaluator:
        def __init__(self) -> None:
            pass
        def evaluate(self, phen, parameters):
            return 1, {}
    
    evaluator = FitnessEvaluator()
    load_parameters("parameters/adaptive_autolr.yml")
    #params['CURRENT_GEN'] = 0
    indiv = {'key': 'grad', 'genotype': [[0], [1], [], [0], [1], [], [1], [], [0], [0], [8], [0, 1, 1], [6], [0, 1], [1], [], [0, 1, 0, 1, 0, 1, 0], [1, 4, 4, 9], [1, 1, 2], [43]], 'fitness': 1, 'tree_depth': 9, 'operation': 'initialization', 'id': 0, 'phenotype': 'alpha_func, beta_func, sigma_func, grad_func = lambda shape, alpha, grad: alpha, lambda shape, alpha, beta, grad: tf.constant(2.28478855e-04, shape=shape, dtype=tf.float32), lambda shape, alpha, beta, sigma, grad: tf.math.divide_no_nan(alpha, grad), lambda shape, alpha, beta, sigma, grad: grad', 'mapping_values': [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 3, 1, 2, 1, 0, 7, 4, 3, 1], 'smart_phenotype': 'subtract(alpha, pow(alpha, pow(constant(2.28478855e-04), constant(2.11963334e-01))))', 'other_info': {}}
    indiv_2 = copy.deepcopy(indiv)
    evaluator.evaluate(indiv_2, FitnessEvaluator())

    indiv.pop('other_info')
    indiv_2.pop('other_info')

    print(indiv)
    print(indiv_2)
    
    print(indiv == indiv_2)
    assert indiv == indiv_2


#if __name__ == "__main__":
#    test_archive_id()