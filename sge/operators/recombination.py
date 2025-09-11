import random
import copy
from sge.grammar import grammar


def crossover(p1, p2, pcrossover):
    p = copy.deepcopy(p1)
    p['fitness'] = None
    if random.random() < pcrossover:
        xover_p_value = 0.5
        gen_size = len(p1['genotype'])
        mask = [random.random() for i in range(gen_size)]
        genotype = []
        for index, prob in enumerate(mask):
            if prob < xover_p_value:
                genotype.append(p1['genotype'][index][:])
            else:
                genotype.append(p2['genotype'][index][:])
        mapping_values = [0] * gen_size
        # compute nem individual
        _, tree_depth = grammar.mapping(genotype, mapping_values)
        p = {'genotype': genotype, 'fitness': None, 'mapping_values': mapping_values, 'tree_depth': tree_depth, 'operation': 'crossover', 'parent': [p1['parent'][0], p2['parent'][0]]}
    return p