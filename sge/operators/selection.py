import random
import copy


def tournament(population, tsize=3):
    pool = random.sample(population, tsize)
    pool.sort(key=lambda i: i['fitness'])
    indiv = copy.deepcopy(pool[0])
    indiv["operation"] = "copy" 
    indiv["parent"] = [indiv['id']]
    return indiv
