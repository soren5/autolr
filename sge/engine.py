from lib2to3.pgen2 import driver
from math import isclose
from operator import inv
import random
import sys
from xml.etree.ElementTree import tostring
from sge.grammar import grammar
import copy
from datetime import datetime
from sge.logger import find_last_generation_to_load
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate_level, mutate
from sge.operators.selection import tournament, universal_stochastic_sampling
import time
import statistics
from scipy import stats
import numpy as np
from sge.parameters import (
    params,
    set_parameters,
    manual_load_parameters
)
from utils.genotypes import *
from utils.smart_phenotype import smart_phenotype, single_task_key, readable_phenotype


def generate_random_individual():
    genotype = [[] for key in grammar.get_non_terminals()]
    tree_depth = grammar.recursive_individual_creation(genotype, grammar.get_start_rule()[0], 0)
    return {'genotype': genotype, 'fitness': None, 'tree_depth' : tree_depth, 'operation': "initialization"}

def make_initial_population():
    for i in range(params['POPSIZE']):
        yield generate_random_individual()

def initialize_population(solutions=[]):
    population = list(make_initial_population())
    for i in range(len(solutions)):
        population[i] = {"genotype": solutions[i], "fitness": None, "parent": "X", 'operation': "initialization"}
    for i in range(len(population)):
        population[i]['id'] = i
    return population

def start_population_from_scratch():
    population = initialize_population()
    archive = {}
    for indiv in population:
        indiv["evaluations"] = [] 
        mapping_values = [0 for i in indiv['genotype']]
        phen, tree_depth = grammar.mapping(indiv['genotype'], mapping_values)
        indiv['phenotype'] = phen
        indiv['mapping_values'] = mapping_values
    id = len(population)
    counter = id - 1
    it = 0
    return population, archive, counter, it 

def evaluate(ind, eval_func):
    start = time.time()
    if 'phenotype' not in ind:
        mapping_values = [0 for i in ind['genotype']]
        phen, tree_depth = grammar.mapping(ind['genotype'], mapping_values)
    else:
        mapping_values = ind['mapping_values']
        phen = ind['phenotype']
        tree_depth = ind['tree_depth']
    other_info = {}
    #print(f"Registering {smart_phenotype(phen)}")
    if 'grad' in smart_phenotype(phen):
        if "FAKE_FITNESS" in params and params['FAKE_FITNESS']:
            print(f"USING FAKE FITNESS")
            import numpy as np
            import tensorflow as tf
            quality = -(random.random() + np.random.random())/2
            other_info = {'source': 'evaluation'}
        else:
            quality, other_info = eval_func.evaluate(phen, params)
    else:
        print('\t[GRAD CHECK] FAIL')
        quality = params['FITNESS_FLOOR']
        other_info = {'source': 'invalid detection'}


    end = time.time()
    if "FAKE_FITNESS" in params and params['FAKE_FITNESS']:
        duration = random.random()
    else:
        duration = end - start
    ind['phenotype'] = phen 
    ind['fitness'] = quality
    ind['other_info'] = other_info
    ind['other_info']['duration'] = duration
    ind['mapping_values'] = mapping_values
    ind['tree_depth'] = tree_depth
    ind['key'] = single_task_key(ind['phenotype'], params['CURRENT_GEN'])



def setup(evaluation_function=None, parameters=None, logger=None):
    if parameters is None:
        set_parameters(sys.argv[1:])
    else:
        manual_load_parameters(parameters)
    print(params)

    #print(params)
    if 'SEED' not in params:
        params['SEED'] = int(datetime.now().microsecond)
    if logger is None:
        import sge.logger as logger
        print("Using Native Logger")
    if 'RESUME' in params and type(params['RESUME']) == str and params["RESUME"].isdecimal():
        params["RESUME"] = int(params["RESUME"])
    logger.params = params 
    logger.prepare_dumps()
    
    random.seed(params['SEED'])
    np.random.seed(params['SEED'])
    
    grammar.set_path(params['GRAMMAR'])
    grammar.read_grammar()
    grammar.set_max_tree_depth(params['MAX_TREE_DEPTH'])
    grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])
    
    if evaluation_function != None and not ("FAKE_FITNESS" in params and params['FAKE_FITNESS']):
        evaluation_function.init_net(params)
        evaluation_function.init_data(params)
        evaluation_function.init_evaluation(params)




def evolutionary_algorithm(evaluation_function=None, parameters=None, logger_module=None):
    import os
    
    logger = read_params(parameters, logger_module)
    setup(evaluation_function, parameters, logger_module)

    #check_google_colab(params, logger)


    population, archive, counter, it = initialize_pop(logger)

    print(params)
    
    return run_evolution(evaluation_function, logger, population, archive, counter, it)

def run_evolution(evaluation_function, logger, population, archive, counter, it):
    
    start_time = time.time()
    
    while simulation_is_running(it, start_time):
        
        print(f"{it}")
        params["CURRENT_GEN"] = it
        evaluation_function, population, archive, it = update_archive_and_fitness(evaluation_function, population, archive, it)
        
        save_data(logger, population, it)


        logger, population, archive, counter, it = reproduction_and_elitism(logger, population, archive, counter, it)

    return population

def update_archive_and_fitness(evaluation_function, population, archive, it):
    for indiv in population:
        evaluation_function, archive, indiv = update_archive(evaluation_function, archive, indiv, it)
 
    population, archive = update_best_fitness(population, archive)
       
    for indiv in population:
        archive, indiv = update_key_and_fitness_based_on_archive(archive, indiv)

    population, it  = sort_pop_and_print_best_fit(population, it)
    return evaluation_function, population, archive, it

def simulation_is_over(it):
    return it == params['GENERATIONS']

def simulation_is_running(it, start_time):
    return it < params['GENERATIONS'] and (True if 'TIME_STOP' not in params else (True if time.time() - start_time < params['TIME_STOP'] else False))

def reproduction_and_elitism(logger, population, archive, counter, it):
    new_population, population = reproduce_via_elitism(population)
    logger, population, archive, counter, it, new_population = reproduction(logger, population, archive, counter, it, new_population)
    return logger, population, archive, counter, it

def save_data(logger, population, it):
    logger.evolution_progress(it, population)
    logger.elicit_progress(it, population)

def sort_pop_and_print_best_fit(population, it):
    population = sort_pop_based_on_fitness(population)
    print("\ngeneration: " + str(it) + "; best fit so far: " + str(population[0]['fitness']) + "\n")
    return population, it

def read_params(parameters, logger_module):
    if logger_module != None:
        logger = logger_module
    else:
        import sge.logger as logger
    return logger

def check_google_colab(params, logger):
    if "COLAB" in params and params["COLAB"]:
        from google.colab import drive
        drive.mount('/content/drive')
    if 'RESUME' in params:
        population = logger.load_population(params['RESUME'])
        if params['LOAD_ARCHIVE']:
            archive = logger.load_archive(params['RESUME'])
        else:
            archive = {}
        logger.load_random_state(params['RESUME'])
        it = params['RESUME']
        counter = int(np.max([archive[x]['id'] for x in archive]))

def reproduction(logger, population, archive, counter, it, new_population):
    while len(new_population) < params['POPSIZE']:
        new_indiv = selection(population)
        new_indiv_2 = selection(population) 
        #print(new_indiv)
        new_indiv = crossover(new_indiv, new_indiv_2, params['PROB_CROSSOVER'])
        new_indiv = mutation(new_indiv)
        new_indiv = map_phenotype(new_indiv)
        archive, counter, new_population, new_indiv = update_archive_with_new_indiv(archive, counter, new_population, new_indiv)
    it, population = go_to_next_generation(it, new_population)
    save_data_new_pop(logger, population, archive, it)
    use_google_colab_in_reproduction()
    return logger, population, archive, counter, it, new_population

def selection(population):
    if params['SELECTION_TYPE'] == 'tournament':
        new_indiv = tournament_selection(population)
    elif params['SELECTION_TYPE'] == 'stochastic':
        new_indiv = universal_stochastic_sampling(population)
    return new_indiv

def go_to_next_generation(it, new_population):
    population = new_population
    it += 1
    return it, population

def use_google_colab_in_reproduction():
    if "COLAB" in params and params["COLAB"]:
        driver.flush_and_unmount()
        driver.Driver.mount('/content/drive')
        import os
        print(os.listdir(f"{params['EXPERIMENT_NAME']}/run_{params['RUN']}"))

def save_data_new_pop(logger, population, archive, it):
    logger.save_archive(it, archive)
    logger.save_population(it, population)
    logger.save_random_state(it)

def update_archive_with_new_indiv(archive, counter, new_population, new_indiv):
    key = single_task_key(new_indiv['phenotype'], params['CURRENT_GEN'])
    if key in archive:
        new_indiv['id'] = archive[key]['id']
    else:
        counter += 1
        new_indiv['id'] = counter
    new_population.append(new_indiv)
    return archive, counter, new_population, new_indiv

def map_phenotype(new_indiv):
    mapping_values = [0 for i in new_indiv['genotype']]
    phen, tree_depth = grammar.mapping(new_indiv['genotype'], mapping_values)
    new_indiv['phenotype'] = phen
    new_indiv['smart_phenotype'] = smart_phenotype(phen)
    return new_indiv

def mutation(new_indiv):
    if type(params['PROB_MUTATION']) == float:
        new_indiv = mutate(new_indiv, params['PROB_MUTATION'])
    elif type(params['PROB_MUTATION']) == dict:
        assert len(params['PROB_MUTATION']) == len(new_indiv['genotype'])
        new_indiv = mutate_level(new_indiv, params['PROB_MUTATION'])
    else:
        raise Exception("Invalid mutation type")
    return new_indiv

def tournament_selection(population):
    #if random.random() < params['PROB_CROSSOVER']:
    #    p1 = tournament(population, params['TSIZE'])
    #    p2 = tournament(population, params['TSIZE'])
    #    new_indiv = crossover(p1, p2)
    #else:
    new_indiv = tournament(population, params['TSIZE'])
    return new_indiv

def reproduce_via_elitism(population):
    behaviors_added = []
    new_population = []
    print("[ELITE] Adding following individuals to new population:")
    for indiv in population:
        if indiv['smart_phenotype'] not in behaviors_added and len(behaviors_added) < params['ELITISM']:
            behaviors_added.append(indiv['smart_phenotype'])
            new_population.append(indiv)
            print(readable_phenotype(indiv['phenotype']))
    while len(new_population) < params['ELITISM']:
        print("[ELITE] Could not fill with unique individuals, adding best individual again.")
        new_population.append(population[0])

    for indiv in new_population:
        indiv['operation'] = 'elitism'
        indiv['other_info']['source'] = 'elitism'
    return new_population, population

def sort_pop_based_on_fitness(population):
    population.sort(key=lambda x: x['fitness'])
    return population

def update_key_and_fitness_based_on_archive(archive, indiv):
    key = update_key(indiv)
    update_fitness_based_on_archive(archive, indiv, key)
    return archive, indiv

def update_fitness_based_on_archive(archive, indiv, key):
    indiv['fitness'] = archive[key]['fitness']
    if 'other_info' not in indiv:
        indiv['other_info'] = {}
    if 'source' not in indiv['other_info']:
        indiv['other_info']['source'] = 'archive'


def update_key(indiv):
    key = single_task_key(indiv['phenotype'], params['CURRENT_GEN'])
    return key

def update_best_fitness(population, archive):
    best_fit = params['FITNESS_FLOOR'] + 1
    for indiv in population:
        key = single_task_key(indiv['phenotype'], params['CURRENT_GEN'])
        #key = indiv['smart_phenotype']
        if archive[key]['fitness'] < best_fit:
                # best = archive[key]
            best_fit = archive[key]['fitness'] 
        return population, archive

def update_archive(evaluation_function, archive, indiv, it):
    indiv['smart_phenotype'] = smart_phenotype(indiv['phenotype'])
    key = single_task_key(indiv['phenotype'], params['CURRENT_GEN'])
    if key in archive and 'fitness' not in archive[key]:
        raise Exception('Incomplete archive entry')
    if key not in archive:
        archive[key] = {'evaluations': []}
        archive[key]['id'] = indiv['id']
        evaluate(indiv, evaluation_function)
        archive[key]['evaluations'].append(indiv['fitness'])
        archive[key]['fitness'] = statistics.mean(archive[key]['evaluations'])

    return evaluation_function, archive, indiv

def initialize_pop(logger):
    if 'RESUME' in params:
        if params["RESUME"] == "Last":
            last_gen = find_last_generation_to_load()
        elif type(params["RESUME"]) == int:
            if params["RESUME"] != 0: 
                last_gen = params['RESUME']
            else: 
                last_gen = None
        else:
            raise Exception("Invalid RESUME")

        if last_gen != None:           
            population = logger.load_population(last_gen)
            logger.load_random_state(last_gen)
            if 'LOAD_ARCHIVE' in params and params['LOAD_ARCHIVE'] == True:     
                archive = logger.load_archive(last_gen)
            else:
                archive = {}
            counter = int(np.max([x['id'] for x in population]))
            it = last_gen
        else:
            population, archive, counter, it = start_population_from_scratch()
    
    else:
        if 'PREPOPULATE' in params and params['PREPOPULATE']:
            genes_dict={
                'all': [get_adam_genotype(), get_momentum_genotype(), get_rmsprop_genotype()],
                'adam': [get_adam_genotype()],
                'rmsprop': [get_rmsprop_genotype()],
                'momentum': [get_momentum_genotype()],
            }
            population = initialize_population(genes_dict[params["GENES"]])
        else:
            population, archive, counter, it = start_population_from_scratch()  
            for indiv in population:
                mapping_values = [0 for i in indiv['genotype']]
                phen, tree_depth = grammar.mapping(indiv['genotype'], mapping_values)
                indiv['phenotype'] = phen
                indiv['mapping_values'] = mapping_values
    return population, archive, counter, it