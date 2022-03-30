from operator import inv
import random
import sys
import sge.grammar as grammar
import sge.logger as logger
import copy
from datetime import datetime
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate
from sge.operators.selection import tournament
import time
import statistics
from scipy import stats
from sge.parameters import (
    params,
    set_parameters
)
from genotypes import *


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
        population[i] = {"genotype": solutions[i], "fitness": None}
    for i in range(len(population)):
        population[i]['id'] = i
    return population
    



def evaluate(ind, eval_func):
    mapping_values = [0 for i in ind['genotype']]
    phen, tree_depth = grammar.mapping(ind['genotype'], mapping_values)
    quality, other_info = eval_func.evaluate(phen, params)
    ind['phenotype'] = phen
    ind['fitness'] = quality
    ind['other_info'] = other_info
    ind['mapping_values'] = mapping_values
    ind['tree_depth'] = tree_depth


def setup():
    set_parameters(sys.argv[1:])
    #print(params)
    if params['SEED'] is None:
        params['SEED'] = int(datetime.now().microsecond)
    logger.prepare_dumps()
    random.seed(params['SEED'])
    grammar.set_path(params['GRAMMAR'])
    grammar.read_grammar()
    grammar.set_max_tree_depth(params['MAX_TREE_DEPTH'])
    grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])


def evolutionary_algorithm(evaluation_function=None, resume_generation=-1):
    setup()
    #print(sys.argv)
    if params['RESUME'] > -1:
        population = logger.load_population(params['RESUME'])
        logger.load_random_state()
        it = params['RESUME']
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
        it = 0
    history = {}
    start_time = time.time()
    for i in population:
        i["evaluations"] = [] 
    id = len(population)
    counter = id
    while it <= params['GENERATIONS']:
        evaluation_indices = list(range(len(population)))
        for i in population:
            for _ in range(5):
                evaluate(i, evaluation_function)
                i['evaluations'].append(i['fitness'])
                i['fitness'] = statistics.mean(i['evaluations'])
                with open("log.txt", 'a') as f:
                    print(f"[{i['id']}] new_fitness {i['fitness']}, n_evaluations {len(i['evaluations'])}, evaluations {i['evaluations']}", file=f)
        stat, p_value_kruskal = stats.kruskal(*[population[x]['evaluations'] for x in evaluation_indices])
        while p_value_kruskal < 0.05 and len(evaluation_indices) > 1:
            with open("log.txt", 'a') as f:
                print(f"Running iteration {it}...",file=f)
            #population.sort(key=lambda x: x['fitness'])
            #best = population[0]
            best_fit = 0
            for indiv in population:
                if indiv['fitness'] < best_fit:
                    best = indiv
                    best_fit = indiv['fitness']
            to_remove = []
            for eval_index in evaluation_indices:
                indv = population[eval_index]
                with open("log.txt", 'a') as f:
                    print(f"Testing {indv['id']}",  file=f)
                try:
                    stat, p_value = stats.mannwhitneyu(best['evaluations'], indv['evaluations'])
                except ValueError as e:
                    with open("log.txt", 'a') as f:
                        print(f"Value error: {e}. Continuing search", file=f)
                    p_value = 1
                #with open("log.txt", 'a') as f:
                    #print(f"best_fitness: {best['fitness']}, indiv_fitness: {indv['fitness']}, mannwhitneyu pvalue: {p_value}", file=f)
                #print(best["evaluations"], indv['evaluations'])
                #if p_value < (0.05 / len(population)):
                if p_value < (0.05):
                    with open("log.txt", 'a') as f:
                        print(f"Removing individual [{indv['id']}].",  file=f)
                    to_remove.append(eval_index)
                else:
                    with open("log.txt", 'a') as f:
                        print(f"Evaluating individual [{indv['id']}].",  file=f)
                    evaluate(indv, evaluation_function)
                    indv['evaluations'].append(indv['fitness'])
                    indv['fitness'] = statistics.mean(indv['evaluations']) 
                    with open("log.txt", 'a') as f:
                        print(f"[{indv['id']}] new_fitness {indv['fitness']}, n_evaluations {len(indv['evaluations'])}, evaluations {indv['evaluations']}", file=f)
            for remove_index in to_remove:
                evaluation_indices.remove(remove_index)
            ids_left = [population[x]["id"] for x in evaluation_indices]
            if len(evaluation_indices) > 1:
                stat, p_value_kruskal = stats.kruskal(*[population[x]['evaluations'] for x in evaluation_indices])
                if p_value_kruskal < 0.05:
                    with open("log.txt", 'a') as f:
                        print(f"{p_value_kruskal} - There is a significant difference in the population, indivs left: {ids_left}", file=f)
                else:
                    with open("log.txt", 'a') as f:
                        print(f"{p_value_kruskal} - There is no significant difference in the population, concluding iteration {it}", file=f)
            else:
                with open("log.txt", 'a') as f:
                    print(f"Only one individual left, concluding iteration {it}", file=f)
            
            
                    
        population.sort(key=lambda x: len(x["evaluations"]))
        logger.evolution_progress(it, population)
        new_population = population[:params['ELITISM']]
        while len(new_population) < params['POPSIZE']:
            if random.random() < params['PROB_CROSSOVER']:
                p1 = tournament(population, params['TSIZE'])
                p2 = tournament(population, params['TSIZE'])
                ni = crossover(p1, p2)
            else:
                ni = tournament(population, params['TSIZE'])
            ni = mutate(ni, params['PROB_MUTATION'])
            counter += 1
            ni['id'] = counter
            ni['evaluations'] = []
            new_population.append(ni)
        population = new_population
        it += 1
        print(population)
        #logger.bee_report(it, population, start_time)
        logger.save_random_state()
    return population

