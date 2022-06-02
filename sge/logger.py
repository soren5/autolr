import numpy as np
from sge.parameters import params
import json
import os
import tensorflow as tf
import random
import pickle



def evolution_progress(generation, pop):
    fitness_samples = [i['fitness'] for i in pop]
    data = '%4d\t%.6e\t%.6e\t%.6e' % (generation, np.min(fitness_samples), np.mean(fitness_samples), np.std(fitness_samples))
    if params['VERBOSE']:
        print(data)
    save_progress_to_file(data)
    if generation % params['SAVE_STEP'] == 0:
        save_step(generation, pop)



def save_random_state():
    builtin_state = random.getstate()
    with open(str(params['EXPERIMENT_NAME']) + '/run_' + str(params['RUN']) + '/builtinstate', 'wb') as f:
        pickle.dump(builtin_state, f)
    np.random.seed(random.randint(0, 1000000))
    #pickle.dump(numpy_state, str(params['EXPERIMENT_NAME']) + '/run_' + str(params['RUN']) + '/numpystate')
    tf.random.set_seed(random.randint(0, 1000000))
    

def load_random_state():
    with open(str(params['EXPERIMENT_NAME']) + '/run_' + str(params['RUN']) + '/builtinstate', 'rb') as f:
        builtin_state = pickle.load(f)
    #numpy_state = pickle.load(str(params['EXPERIMENT_NAME']) + '/run_' + str(params['RUN']) + '/numpystate')
    random.setstate(builtin_state)
    np.random.seed(random.randint(0, 1000000))
    tf.random.set_seed(random.randint(0, 1000000))


def save_progress_to_file(data):
    with open('%s/run_%d/_progress_report.csv' % (params['EXPERIMENT_NAME'], params['RUN']), 'a') as f:
        f.write(data + '\n')


def save_step(generation, population):
    with open('%s/run_%d/iteration_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation), 'w') as f:
        json.dump(population, f)

def load_population(generation):
    with open('%s/run_%d/iteration_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation), 'r') as f:
        population = json.load(f)
    return population

def save_parameters():
    params_lower = dict((k.lower(), v) for k, v in params.items())
    c = json.dumps(params_lower)
    open('%s/run_%d/parameters.json' % (params['EXPERIMENT_NAME'], params['RUN']), 'a').write(c)


def prepare_dumps():
    try:
        os.makedirs('%s/run_%d' % (params['EXPERIMENT_NAME'], params['RUN']))
    except FileExistsError as e:
        pass
    save_parameters()