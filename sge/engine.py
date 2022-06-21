from operator import inv
import random
import sys
import sge.grammar as grammar
import copy
from datetime import datetime
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate_level as mutate
from sge.operators.selection import tournament
import time
import statistics
from scipy import stats
import numpy as np
from sge.parameters import (
    params,
    set_parameters
)
from genotypes import *
from utils.smart_phenotype import smart_phenotype


def generate_random_individual():
    genotype = [[] for key in grammar.get_non_terminals()]
    tree_depth = grammar.recursive_individual_creation(genotype, grammar.start_rule()[0], 0)
    return {'genotype': genotype, 'fitness': None, 'tree_depth' : tree_depth}


def make_initial_population():
    for i in range(params['POPSIZE']):
        yield generate_random_individual()

def initialize_population(solutions=[]):
    population = list(make_initial_population())
    for i in range(len(solutions)):
        population[i] = {"genotype": solutions[i], "fitness": None, "parent": "X"}
    for i in range(len(population)):
        population[i]['id'] = i
    return population
    



def evaluate(ind, eval_func):
    mapping_values = [0 for i in ind['genotype']]
    phen, tree_depth = grammar.mapping(ind['genotype'], mapping_values)
    if 'grad' in smart_phenotype(phen):
        import numpy as np
        import tensorflow as tf
        quality, other_info = eval_func.evaluate(phen, params)
        #quality = -(random.random() + np.random.random() + tf.random.Generator.from_seed(random.randint(0, 100)).normal([]).numpy())/3
        #quality = -(random.random() + np.random.random())/2
        other_info = {}
    else:
        quality = 100000
        other_info = {}
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
    if params['SEED'] is None:
        params['SEED'] = int(datetime.now().microsecond)
    if logger is None:
        import sge.logger as logger
        print("Using Native Logger")
    logger.prepare_dumps()
    random.seed(params['SEED'])
    grammar.set_path(params['GRAMMAR'])
    grammar.read_grammar()
    grammar.set_max_tree_depth(params['MAX_TREE_DEPTH'])
    grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])


def evolutionary_algorithm(evaluation_function=None, resume_generation=-1, parameters=None, logger_module=None):
    if logger_module != None:
        logger = logger_module
    else:
        import sge.logger as logger
    setup(parameters, logger_module)
    if "COLAB" in params and params["COLAB"]:
        from google.colab import drive
        drive.mount('/content/drive')
    #print(sys.argv)
    if params['RESUME'] > -1:
        #population = logger.load_population(params['RESUME'])
        #archive = logger.load_archive(params['RESUME'])
        #logger.load_random_state()
        it = params['RESUME']
        counter = len(archive) 
    else:
        print(params['EPOCHS'])
        if params['PREPOPULATE']:
            genes_dict={
                'all': [get_adam_genotype(), get_momentum_genotype(), get_rmsprop_genotype()],
                'adam': [get_adam_genotype()],
                'rmsprop': [get_rmsprop_genotype()],
                'momentum': [get_momentum_genotype()],
            }
            population = initialize_population(genes_dict[params["GENES"]])
        else:
            population = initialize_population()
        archive = {}
        for indiv in population:
            indiv["evaluations"] = [] 
            mapping_values = [0 for i in indiv['genotype']]
            phen, tree_depth = grammar.mapping(indiv['genotype'], mapping_values)
            indiv['phenotype'] = phen
            indiv['mapping_values'] = mapping_values
        id = len(population)
        counter = id
        it = 0
    
    
    while it <= params['GENERATIONS']:
        evaluation_indices = list(range(len(population)))
        for indiv in population:
            indiv['smart_phenotype'] = smart_phenotype(indiv['phenotype'])
            if indiv['smart_phenotype'] not in archive:
                key = indiv['smart_phenotype']
                archive[key] = {'evaluations': []}
                archive[key]['id'] = indiv['id']
                for _ in range(5):
                    evaluate(indiv, evaluation_function)
                    archive[key]['evaluations'].append(indiv['fitness'])
                    archive[key]['fitness'] = statistics.mean(archive[key]['evaluations'])
                    with open("log.txt", 'a') as f:
                        print(f"[{it}][{indiv['id']}] new_fitness { archive[key]['fitness']}, n_evaluations {len( archive[key]['evaluations'])}, smart phenotype {key}", file=f)
        try:
            stat, p_value_kruskal = stats.kruskal(*[archive[population[x]['smart_phenotype']]['evaluations'] for x in evaluation_indices])
        except ValueError as e:
            with open("log.txt", 'a') as f:
                print(f"[{it}] Value error: {e}. Next Iteration.", file=f)
            p_value_kruskal = 1
        while p_value_kruskal < 0.05 and len(evaluation_indices) > 1:
            with open("log.txt", 'a') as f:
                print(f"[{it}] Running iteration...",file=f)
            #population.sort(key=lambda x: x['fitness'])
            #best = population[0]
            best_fit = 100000
            for indiv in population:
                key = indiv['smart_phenotype']
                if archive[key]['fitness'] < best_fit:
                    best = archive[key]
                    best_fit = archive[key]['fitness']
            to_remove = []
            for eval_index in evaluation_indices:
                indiv = population[eval_index]
                with open("log.txt", 'a') as f:
                    print(f"[{it}] Testing {indiv['id']}",  file=f)
                if indiv['id'] != best['id']:
                    try:
                        stat, p_value = stats.mannwhitneyu(best['evaluations'], archive[indiv['smart_phenotype']]['evaluations'])
                    except ValueError as e:
                        with open("log.txt", 'a') as f:
                            print(f"[{it}] [{indiv['id']}] is equal to best [{best['id']}. Value error: {e}", file=f) 
                        p_value = 1
                    if p_value < 0.05:
                        with open("log.txt", 'a') as f:
                            print(f"[{it}] Removing individual [{indiv['id']}], different from best [{best['id']}].",  file=f)
                        to_remove.append(eval_index)
                    else:
                        with open("log.txt", 'a') as f:
                            print(f"[{it}] Evaluating individual [{indiv['id']}], similar to best [{best['id']}].",  file=f)
                        key = indiv['smart_phenotype']             
                        evaluate(indiv, evaluation_function)
                        archive[key]['evaluations'].append(indiv['fitness'])
                        archive[key]['fitness'] = statistics.mean(archive[key]['evaluations']) 
                        with open("log.txt", 'a') as f:
                            print(f"[{it}][{indiv['id']}] new_fitness {archive[key]['fitness']}, n_evaluations {len(archive[key]['evaluations'])}, smart phenotype {key}", file=f)
            for remove_index in to_remove:
                evaluation_indices.remove(remove_index)
            ids_left = [population[x]["id"] for x in evaluation_indices]
            if len(evaluation_indices) > 1:
                try:
                    stat, p_value_kruskal = stats.kruskal(*[archive[population[x]['smart_phenotype']]['evaluations'] for x in evaluation_indices])
                except ValueError as e:
                    with open("log.txt", 'a') as f:
                        print(f"[{it}] Entire population is equal. Concluding iteration. Value error: {e}. ", file=f)
                    p_value_kruskal = 1
                if p_value_kruskal < 0.05:
                    with open("log.txt", 'a') as f:
                        print(f"[{it}] {p_value_kruskal} - There is a significant difference in the population, {len(ids_left)} indivs left: {ids_left}", file=f)
                else:
                    with open("log.txt", 'a') as f:
                        print(f"{p_value_kruskal} - There is no significant difference in the population, concluding iteration.", file=f)
            else:
                with open("log.txt", 'a') as f:
                    print(f"[{it}] Only one individual left, concluding iteration.", file=f)
        for indiv in population:
            key = indiv['smart_phenotype']
            indiv['fitness'] = archive[key]['fitness']
        print(f"{[x['id'] for x in population]}")
        population.sort(key=lambda x: x['fitness'])
        logger.evolution_progress(it, population)
        new_population = population[:params['ELITISM']]
        while len(new_population) < params['POPSIZE']:
            if random.random() < params['PROB_CROSSOVER']:
                p1 = tournament(population, params['TSIZE'])
                p2 = tournament(population, params['TSIZE'])
                new_indiv = crossover(p1, p2)
                new_indiv["parent"] = [p1['id'], p2['id']]
            else:
                new_indiv = tournament(population, params['TSIZE'])
                new_indiv["parent"] = [new_indiv['id']]

            new_indiv = mutate(new_indiv, params['PROB_MUTATION'])

            mapping_values = [0 for i in new_indiv['genotype']]
            phen, tree_depth = grammar.mapping(new_indiv['genotype'], mapping_values)
            new_indiv['phenotype'] = phen
            new_indiv['smart_phenotype'] = smart_phenotype(phen)
            if new_indiv['smart_phenotype'] in archive:
                with open("log.txt", 'a') as f:
                    print(f"[{it}][{len(new_population)}/{params['POPSIZE']}] New individual found in archive with id {archive[new_indiv['smart_phenotype']]['id']}", file=f)
                new_indiv['id'] = archive[new_indiv['smart_phenotype']]['id']
            else:
                with open("log.txt", 'a') as f:
                    print(f"[{it}][{len(new_population)}/{params['POPSIZE']}] New individual with new behaviour, assigned id {counter + 1}", file=f)
                counter += 1
                new_indiv['id'] = counter
            new_population.append(new_indiv)
        population = new_population
        it += 1
        print(population)
        #logger.save_archive(it, archive)
        #logger.save_random_state()
        if "COLAB" in params and params["COLAB"]:
            drive.flush_and_unmount()
            drive.mount('/content/drive')
            import os
            print(os.listdir(f"{params['EXPERIMENT_NAME']}/run_{params['RUN']}"))
    return population

