from math import isclose
from operator import inv
import random
import sys
from xml.etree.ElementTree import tostring
import sge.grammar as grammar
import copy
from datetime import datetime
from sge.logger import find_generation_to_load
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate_level, mutate
from sge.operators.selection import tournament
import time
import statistics
from scipy import stats
import numpy as np
from sge.parameters import (
    params,
    set_parameters
)
from utils.genotypes import *
from utils.smart_phenotype import smart_phenotype


def generate_random_individual():
    genotype = [[] for key in grammar.get_non_terminals()]
    tree_depth = grammar.recursive_individual_creation(genotype, grammar.start_rule()[0], 0)
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
    mapping_values = [0 for i in ind['genotype']]
    phen, tree_depth = grammar.mapping(ind['genotype'], mapping_values)
    other_info = {}
    if "FAKE_FITNESS" in params and params['FAKE_FITNESS']:
        import numpy as np
        import tensorflow as tf
        quality = -(random.random() + np.random.random())/2
    else:
        if 'grad' in smart_phenotype(phen):
            quality, other_info = eval_func.evaluate(phen, params)
        else:
            quality = params['FITNESS_FLOOR']
    ind['phenotype'] = phen 
    ind['fitness'] = quality
    ind['other_info'] = other_info
    ind['mapping_values'] = mapping_values
    ind['tree_depth'] = tree_depth


def setup(parameters=None, logger=None):
    if parameters is None:
        set_parameters(sys.argv[1:])
    else:
        global params
        params = parameters
    #print(params)
    if 'SEED' not in params:
        params['SEED'] = int(datetime.now().microsecond)
    if logger is None:
        import sge.logger as logger
        print("Using Native Logger")
    logger.prepare_dumps()
    random.seed(params['SEED'])
    np.random.seed(params['SEED'])
    grammar.set_path(params['GRAMMAR'])
    grammar.read_grammar()
    grammar.set_max_tree_depth(params['MAX_TREE_DEPTH'])
    grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])


def evolutionary_algorithm(evaluation_function=None, resume_generation=-1, parameters=None, logger_module=None):
    import os
    if logger_module != None:
        logger = logger_module
    else:
        import sge.logger as logger
    setup(parameters, logger_module)
    
    if "COLAB" in params and params["COLAB"]:
        from google.colab import drive
        drive.mount('/content/drive')
    
    if 'RESUME' in params and params["RESUME"] != False:
        
        if type(params["RESUME"]) == float: 
            last_gen = params['RESUME']
            experiment_name = params["EXPERIMENT_NAME"]
        elif params["RESUME"] == "Last":
            last_gen, experiment_name = find_generation_to_load()
        
        if last_gen != None:           
            population = logger.load_population(last_gen, experiment_name)
            logger.load_random_state(last_gen, experiment_name)
            if 'LOAD_ARCHIVE' in params and params['LOAD_ARCHIVE'] == True:     
                archive = logger.load_archive(last_gen, experiment_name)
                counter = int(np.max([archive[x]['id'] for x in archive]))
            else:
                archive = {}
                id = len(population)
                counter = id - 1
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

    while it <= params['GENERATIONS']:
        print("GENERATION: " + str(it))
        # evaluation_indices = list(range(len(population)))
        for indiv in population:
            indiv['smart_phenotype'] = smart_phenotype(indiv['phenotype'])
            key = indiv['smart_phenotype']
            if key not in archive or 'fitness' not in archive[key] or 'fitness' in archive[key] == None:
                archive[key] = {'evaluations': []}
                archive[key]['id'] = indiv['id']
                # evaluate seems to be deterministic. 
                # Btw., if not, the caching of key|fitness pairs wouldn't be 100% correct
                evaluate(indiv, evaluation_function)
                archive[key]['evaluations'].append(indiv['fitness'])
                archive[key]['fitness'] = statistics.mean(archive[key]['evaluations'])

        """
                # if in doubt (you should;), test:
                for _ in range(5):                     
                    evaluate(indiv, evaluation_function)
                    archive[key]['evaluations'].append(indiv['fitness'])
                    archive[key]['fitness'] = statistics.mean(archive[key]['evaluations'])
                deterministic = archive[key]['evaluations'][0]
                for x in archive[key]['evaluations']:
                    if not isclose(x, deterministic):
                        raise "wrong assumption!"
                

        # `works` without:
        try:
            stat, p_value_kruskal = stats.kruskal(*[archive[population[x]['smart_phenotype']]['evaluations'] for x in evaluation_indices])
        except ValueError as e:
            p_value_kruskal = 1
            
        while p_value_kruskal < 0.05 and len(evaluation_indices) > 1:
        """ 
        best_fit = params['FITNESS_FLOOR'] + 1
        for indiv in population:
            key = indiv['smart_phenotype']
            if archive[key]['fitness'] < best_fit:
                # best = archive[key]
                best_fit = archive[key]['fitness']
        """    
            to_remove = []
            for eval_index in evaluation_indices:
                eval_count = 0
                # Iterate all the individuals, if they are statiscally different from the best, remove them from the next iteration of the cycle.
                # If they are similar to best, re-evaluate
                indiv = population[eval_index]
                key = indiv['smart_phenotype']

                if indiv['id'] != best['id']:
                    try:
                        stat, p_value = stats.mannwhitneyu(best['evaluations'], archive[indiv['smart_phenotype']]['evaluations'])
                    except ValueError as e:
                        p_value = 1
                    if p_value < 0.05:
                        to_remove.append(eval_index)
                    else:
                        # There is no statistical difference, re-evaluate
                        key = indiv['smart_phenotype']  
                        os.system("clear")    
                        print(f"[{it}]-Reval: indiv {eval_count}/{len(evaluation_indices)} eval #{len(archive[key]['evaluations']) + 1}  {key}")
                        evaluate(indiv, evaluation_function)
                        archive[key]['evaluations'].append(indiv['fitness'])
                        archive[key]['fitness'] = statistics.mean(archive[key]['evaluations']) 
            for remove_index in to_remove:
                evaluation_indices.remove(remove_index)
            #ids_left = [population[x]["id"] for x in evaluation_indices]
            if len(evaluation_indices) > 1:
                try:
                    stat, p_value_kruskal = stats.kruskal(*[archive[population[x]['smart_phenotype']]['evaluations'] for x in evaluation_indices])
                except ValueError as e:
                    p_value_kruskal = 1
        """

        for indiv in population:
            key = indiv['smart_phenotype']
            indiv['fitness'] = archive[key]['fitness']
        #print(f"{[x['id'] for x in population]}")

        population.sort(key=lambda x: x['fitness'])
        logger.evolution_progress(it, population)
        logger.elicit_progress(it, population)

        print("\ngeneration: " + str(it) + "; best fit so far: " + str(population[0]['fitness']) + "\n")
        new_population = population[:params['ELITISM']]
        for indiv in new_population:
            indiv['operation'] = 'elitism'
        while len(new_population) < params['POPSIZE']:
            if random.random() < params['PROB_CROSSOVER']:
                p1 = tournament(population, params['TSIZE'])
                p2 = tournament(population, params['TSIZE'])
                new_indiv = crossover(p1, p2)
            else:
                new_indiv = tournament(population, params['TSIZE'])
            print(type(params['PROB_MUTATION']))
            if type(params['PROB_MUTATION']) == float:
                new_indiv = mutate(new_indiv, params['PROB_MUTATION'])
            elif type(params['PROB_MUTATION']) == dict:
                assert len(params['PROB_MUTATION']) == len(new_indiv['genotype'])
                new_indiv = mutate_level(new_indiv, params['PROB_MUTATION'])
            mapping_values = [0 for i in new_indiv['genotype']]
            phen, tree_depth = grammar.mapping(new_indiv['genotype'], mapping_values)
            new_indiv['phenotype'] = phen
            new_indiv['smart_phenotype'] = smart_phenotype(phen)
            if new_indiv['smart_phenotype'] in archive:
                new_indiv['id'] = archive[new_indiv['smart_phenotype']]['id']
            else:
                counter += 1
                new_indiv['id'] = counter
                archive[new_indiv['smart_phenotype']] = {'evaluations': [], 'id': new_indiv['id']}
            new_population.append(new_indiv)
        population = new_population
        it += 1
        logger.save_archive(it, archive)
        logger.save_population(it, population)
        logger.save_random_state(it)
        #why are we reloading what we just saved? does it change in the meantime?
        logger.load_random_state(it, params["EXPERIMENT_NAME"])
        #print(population)
        if "COLAB" in params and params["COLAB"]:
            drive.flush_and_unmount()
            drive.mount('/content/drive')
            import os
            print(os.listdir(f"{params['EXPERIMENT_NAME']}/run_{params['RUN']}"))
    return population
