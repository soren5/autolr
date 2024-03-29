import numpy as np
from sge.parameters import params
import json
import os
import tensorflow as tf
import random
import pickle
import glob
import re
import os.path
from os import path

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

    #find files in correct folder
    f = str(params['EXPERIMENT_NAME']) + '/run_' + str(params['RUN']) + f'/builtinstate_{it}'
    g = str(params['EXPERIMENT_NAME']) + '/run_' + str(params['RUN']) + f'/numpystate_{it}'
    if(not path.isfile(f)) and not path.isfile(g):
        f = str(params['PARENT_EXPERIMENT']) + '/run_' + str(params['RUN']) + f'/builtinstate_{it}'
        g = str(params['PARENT_EXPERIMENT']) + '/run_' + str(params['RUN']) + f'/numpystate_{it}'  
    
    #load files
    if(path.isfile(f) and path.isfile(g)):
        builtin_state = pickle.load(open(f, 'rb'))
        numpy_state = pickle.load(open(g, 'rb'))
        np.random.set_state(numpy_state)
        random.setstate(builtin_state)
    else:
        raise Exception("Cannot open random state file of gen: ",  it, '\n',
        "in folder: ", params["PARENT_EXPERIMENT"],'\\', params['RUN'])



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
    #find file in correct folder
    f = '%s/run_%d/population_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation)
    if(not path.isfile(f)):
        f = '%s/run_%d/population_%d.json' % (params["PARENT_EXPERIMENT"], params['RUN'], generation)
    
    #laod file
    if(path.isfile(f)):
        population = json.load(open(f,'r'))
    else:
        raise Exception("Cannot open population file  of gen: ",  generation, '\n',
        "in folder: ", params["PARENT_EXPERIMENT"],'\\', params['RUN'])
    return population

def save_archive(generation, archive):
    with open('%s/run_%d/z-archive_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation), 'w') as f:
        json.dump(archive, f)


def load_archive(generation):
    #find file in correct folder
    f ='%s/run_%d/z-archive_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation)
    if(not path.isfile(f)):
        f = '%s/run_%d/z-archive_%d.json' % (params["PARENT_EXPERIMENT"], params['RUN'], generation)
    
    #load file
    if(path.isfile(f)):
        archive = json.load(open(f,'r'))
    else:
        raise Exception("Cannot open archive file of gen: ", generation, '\n'
        "in folder: " , params["PARENT_EXPERIMENT"],'\\', params['RUN'])   
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

#extract number from file
def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)

#finds the latest generation recorded in previous running on the simualtion based on the last population recorded
def find_last_gen_recorded_in_folder(folder):

    filenames = glob.glob(('%s/run_%d/builtinstate_?*') % (folder, params['RUN']))
    if(len(filenames) == 0): return None

    last_gen_name = max(filenames, key=extract_number)
    last_gen = extract_number(last_gen_name)
    return int(last_gen[0])

#Will find the most recent generation to load in case there are already some generations in the current experiment
#else it will look for a parent experiment from which to load the data and change the experiment name so that now the folder of the parent expeirment is used to load the data 
#if there is no data to load in any case it will return none

def find_last_generation_to_load():
    last_gen = find_last_gen_recorded_in_folder(params["EXPERIMENT_NAME"])
    if last_gen == None and 'PARENT_EXPERIMENT' in params and params["PARENT_EXPERIMENT"] != False: 
        last_gen = find_last_gen_recorded_in_folder(params["PARENT_EXPERIMENT"])
    return last_gen