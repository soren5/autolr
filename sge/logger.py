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
    import sys
    builtin_state = random.getstate()
    tf_seed = random.randint(0, sys.maxsize)
    tf.random.set_seed(tf_seed)
    numpy_state = np.random.get_state()
    with open(str(params['EXPERIMENT_NAME']) + '/run_' + str(params['RUN']) + '/builtinstate', 'wb') as f:
        pickle.dump(builtin_state, f)
    with open(str(params['EXPERIMENT_NAME']) + '/run_' + str(params['RUN']) + '/numpystate', 'wb') as f:
        pickle.dump(numpy_state, f)
    

def load_random_state():
    import sys
    with open(str(params['EXPERIMENT_NAME']) + '/run_' + str(params['RUN']) + '/builtinstate', 'rb') as f:
        builtin_state = pickle.load(f)
    with open(str(params['EXPERIMENT_NAME']) + '/run_' + str(params['RUN']) + '/numpystate', 'rb') as f:
        numpy_state = pickle.load(f)
    np.random.set_state(numpy_state)
    random.setstate(builtin_state)
    tf_seed = random.randint(0, sys.maxsize)
    tf.random.set_seed(tf_seed)


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
    open('%s/run_%d/parameters.json' % (params['EXPERIMENT_NAME'], params['RUN']), 'a').write(c)


def prepare_dumps():
    try:
        os.makedirs('%s/run_%d' % (params['EXPERIMENT_NAME'], params['RUN']))
    except FileExistsError as e:
        pass
    save_parameters()