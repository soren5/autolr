import numpy as np
from sge.parameters import params
import json
import os
import tensorflow as tf
import random
import pickle
import glob
import re

def evolution_progress(generation, pop):
    fitness_samples = [i['fitness'] for i in pop]
    data = '%4d\t%.6e\t%.6e\t%.6e' % (generation, np.min(fitness_samples), np.mean(fitness_samples), np.std(fitness_samples))
    if params['VERBOSE']:
        print(data)
    save_progress_to_file(data)
    if generation % params['SAVE_STEP'] == 0:
        save_step(generation, pop)

def elicit_progress(generation, pop):
    def translate_operation_to_elicit(operation):
        if operation == "initialization":
            return -1
        elif operation == "copy":
            return 0
        elif operation == "crossover":
            return 1
        elif operation == "mutation":
            return 2
        elif operation == "elitism":
            return 3
        elif operation == "crossover+mutation":
            return 4
        else:
            pass
    data = ""
    for indiv in pop:
        parent_1 = -1
        parent_2 = -1
        if 'parent' in indiv and len(indiv['parent']) >= 1:
            parent_1 = indiv['parent'][0]
            if len(indiv['parent']) >= 2:
                parent_2 = indiv['parent'][1]
        data += f"{generation} {indiv['id']} {translate_operation_to_elicit(indiv['operation'])} {parent_1} {parent_2} {indiv['fitness'] * -1}\n"
    with open('%s/run_%d/elicit_report.txt' % (params['EXPERIMENT_NAME'], params['RUN']), 'a') as f:
        f.write(data)  

def save_random_state(it):
    import sys
    builtin_state = random.getstate()
    #tf_seed = random.randint(0, sys.maxsize)
    #tf.random.set_seed(tf_seed)
    numpy_state = np.random.get_state()
    with open(str(params['EXPERIMENT_NAME']) + '/run_' + str(params['RUN']) + f'/builtinstate_{it}', 'wb') as f:
        pickle.dump(builtin_state, f)
    with open(str(params['EXPERIMENT_NAME']) + '/run_' + str(params['RUN']) + f'/numpystate_{it}', 'wb') as f:
        pickle.dump(numpy_state, f)
    #np.random.set_state(numpy_state)
    #random.setstate(builtin_state)
    

def load_random_state(it):
    import sys
    with open(str(params['EXPERIMENT_NAME']) + '/run_' + str(params['RUN']) + f'/builtinstate_{it}', 'rb') as f:
        builtin_state = pickle.load(f)
    with open(str(params['EXPERIMENT_NAME']) + '/run_' + str(params['RUN']) + f'/numpystate_{it}', 'rb') as f:
        numpy_state = pickle.load(f)
    np.random.set_state(numpy_state)
    random.setstate(builtin_state)
    #tf_seed = random.randint(0, sys.maxsize)
    #tf.random.set_seed(tf_seed)


def save_progress_to_file(data):
    with open('%s/run_%d/_progress_report.csv' % (params['EXPERIMENT_NAME'], params['RUN']), 'a') as f:
        f.write(data + '\n')


def save_step(generation, population):
    with open('%s/run_%d/iteration_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation), 'w') as f:
        json.dump(population, f)

def save_population(generation, population):
    with open('%s/run_%d/population_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation), 'w') as f:
        json.dump(population, f)

def load_population(generation):
    with open('%s/run_%d/population_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation), 'r') as f:
        population = json.load(f)
    return population

def save_archive(generation, archive):
    with open('%s/run_%d/z-archive_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation), 'w') as f:
        json.dump(archive, f)


def load_archive(generation):
    with open('%s/run_%d/z-archive_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation), 'r') as f:
        archive = json.load(f)
    return archive

def save_parameters():
    params_lower = dict((k.lower(), v) for k, v in params.items())
    c = json.dumps(params_lower)
    open('%s/run_%d/_parameters.json' % (params['EXPERIMENT_NAME'], params['RUN']), 'a').write(c)


def prepare_dumps():
    try:
        os.makedirs('%s/run_%d' % (params['EXPERIMENT_NAME'], params['RUN']))
    except FileExistsError as e:
        pass
    save_parameters()

#finds the latest generation recorded in previous running on the simualtion based on the last population recorded
def find_last_gen_recorded_in_files():
    pop_pattern = re.compile('population_\d{1,3}')
    filename = max(glob.glob(('%s/run_%d/population_?*') % (params['EXPERIMENT_NAME'], params['RUN'])))
    last_gen_filename = pop_pattern.findall(filename)
    gen_pattern = re.compile('\d{1,3}')
    last_gen = gen_pattern.findall(last_gen_filename[0])[0]
    return int(last_gen)

