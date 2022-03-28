import random
import sys
import sge.grammar as grammar
import sge.logger as logger
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
        for x in range(5):
            evaluate(i, evaluation_function)
            with open("log.txt", 'a') as f:
                print(f"[{i['id']}] new_fitness {i['fitness']}, evaluations {i['evaluations']}", file=f)
            i['evaluations'].append(i['fitness'])
            i['fitness'] = statistics.mean(i['evaluations'])

    id = len(population)
    while it <= params['GENERATIONS'] * params['POPSIZE']:
        i = population[0]
        evaluate(i, evaluation_function)
        with open("log.txt", 'a') as f:
            print(f"[{i['id']}] new_fitness {i['fitness']}, evaluations {i['evaluations']}", file=f)
        i['evaluations'].append(i['fitness'])
        i['fitness'] = statistics.mean(i['evaluations'])
        stat, p_value = stats.kruskal(*[x['evaluations'] for x in population])
        with open("log.txt", 'a') as f:
            print("kruskall wallis pvalue ", p_value, file=f)
        if p_value < 0.05:
            with open("log.txt", 'a') as f:
                print("There is a significant difference in the population", file=f)
            population.sort(key=lambda x: x['fitness'])
            best = population[0]
            for indv in population:
                stat, p_value = stats.mannwhitneyu(best['evaluations'], indv['evaluations'])
                #with open("log.txt", 'a') as f:
                    #print(f"best_fitness: {best['fitness']}, indiv_fitness: {indv['fitness']}, mannwhitneyu pvalue: {p_value}", file=f)
                #print(best["evaluations"], indv['evaluations'])
                #if p_value < (0.05 / len(population)):
                if p_value < (0.05 / 1):
                    print("Creating new individual.")
                    population.remove(indv)
                    if random.random() < params['PROB_CROSSOVER']:
                        p1 = tournament(population, params['TSIZE'])
                        p2 = tournament(population, params['TSIZE'])
                        ni = crossover(p1, p2)
                    else:
                        ni = tournament(population, params['TSIZE'])
                    ni = mutate(ni, params['PROB_MUTATION'])
                    ni["evaluations"] = [] 
                    ni['id'] = id 
                    id += 1
                    for x in range(5):
                        evaluate(ni, evaluation_function)
                        with open("log.txt", 'a') as f:
                            print(f"[{ni['id']}] new_fitness {ni['fitness']}, evaluations {ni['evaluations']}", file=f)
                        ni['evaluations'].append(ni['fitness'])
                        ni['fitness'] = statistics.mean(ni['evaluations'])
                    population.append(ni)
        population.sort(key=lambda x: len(x["evaluations"]))
        logger.evolution_progress(it, population)
        #logger.bee_report(it, population, start_time)
        logger.save_random_state()
        it += 1
    return population

