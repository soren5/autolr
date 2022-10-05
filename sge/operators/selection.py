import random
import copy


def tournament(population, tsize=3):
    pool = random.sample(population, tsize)
    if any(ind['fitness'] == None for ind in pool):
        raise "Some individuals have no fitness at the moment of selection"
    pool.sort(key=lambda i: i['fitness'])
    indiv = copy.deepcopy(pool[0])
    indiv["operation"] = "copy" 
    indiv["parent"] = [indiv['id']]
    return indiv

def universal_stochastic_sampling(population):
    fits = [indiv['fitness'] for indiv in population]
    max = sum(fits)
    pick = random.uniform(0, max)
    current = 0
    for fitness, indiv in zip(fits, population):
        current += fitness
        if current > pick:
            indiv = copy.deepcopy(indiv)
            indiv["operation"] = "copy" 
            indiv["parent"] = [indiv['id']]
            return indiv



