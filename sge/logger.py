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
    s_phen = [i['smart_phenotype'] for i in pop]
    header = f'gen \tbest \tmean \tdev  \tdiver%'
    percent_diversity = len(list(set(s_phen))) / len(pop)
    data = '%4d\t%.6e\t%.6e\t%.6e\t%.6e' % (generation, np.min(fitness_samples), np.mean(fitness_samples), np.std(fitness_samples), percent_diversity)
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
        path_to_save = os.path.join(params['DUMPS_DIR'], params['EXPERIMENT_NAME'], f"run_{params['RUN']}")
        with open(os.path.join(path_to_save, 'elicit_report.txt'), 'a') as f:
            f.write(data)  

def save_random_state(it):
    import sys
    builtin_state = random.getstate()
    #tf_seed = random.randint(0, sys.maxsize)
    #tf.random.set_seed(tf_seed)
    numpy_state = np.random.get_state()
    path_to_save = os.path.join(params['DUMPS_DIR'], params['EXPERIMENT_NAME'], f"run_{params['RUN']}")
    with open(os.path.join(path_to_save, f'builtinstate_{it}'), 'wb') as f:
        pickle.dump(builtin_state, f)
    with open(os.path.join(path_to_save, f'numpystate_{it}'), 'wb') as f:
        pickle.dump(numpy_state, f)
    #np.random.set_state(numpy_state)
    #random.setstate(builtin_state)
    

def load_random_state(it):
    import sys

    #find files in correct folder
    path_to_save = os.path.join(params['DUMPS_DIR'], params['EXPERIMENT_NAME'], f"run_{params['RUN']}")
    path_to_parent_experiment = os.path.join(params['DUMPS_DIR'], params['PARENT_EXPERIMENT'], f"run_{params['RUN']}") if 'PARENT_EXPERIMENT' in params and params['PARENT_EXPERIMENT'] != False else None
    
    f = os.path.join(path_to_save, f'builtinstate_{it}')
    g = os.path.join(path_to_save, f'numpystate_{it}')

    # if files are not found in current experiment folder, look for them in parent experiment folder
    if(not path.isfile(f)) and not path.isfile(g) and path_to_parent_experiment is not None:
        f = os.path.join(path_to_parent_experiment, f'builtinstate_{it}')
        g = os.path.join(path_to_parent_experiment, f'numpystate_{it}')  
    
    #load files
    if(path.isfile(f) and path.isfile(g)):
        builtin_state = pickle.load(open(f, 'rb'))
        numpy_state = pickle.load(open(g, 'rb'))
        np.random.set_state(numpy_state)
        random.setstate(builtin_state)
    else:
        raise Exception("Cannot open random state file of gen: ",  it, '\n',
        "in folder: ", path_to_parent_experiment if path_to_parent_experiment is not None else path_to_save)



def save_progress_to_file(data):
    path_to_save = os.path.join(params['DUMPS_DIR'], params['EXPERIMENT_NAME'], f"run_{params['RUN']}")
    with open(os.path.join(path_to_save, '_progress_report.csv'), 'a') as f:
        f.write(data + '\n')


def save_step(generation, population):
    path_to_save = os.path.join(params['DUMPS_DIR'], params['EXPERIMENT_NAME'], f"run_{params['RUN']}")
    with open(os.path.join(path_to_save, f'iteration_{generation}.json'), 'w') as f:
        json.dump(population, f)

def save_population(generation, population):
    path_to_save = os.path.join(params['DUMPS_DIR'], params['EXPERIMENT_NAME'], f"run_{params['RUN']}")
    with open(os.path.join(path_to_save, f'population_{generation}.json'), 'w') as f:
        json.dump(population, f)

def load_population(generation):
    path_to_save = os.path.join(params['DUMPS_DIR'], params['EXPERIMENT_NAME'], f"run_{params['RUN']}")
    path_to_parent_experiment = os.path.join(params['DUMPS_DIR'], params['PARENT_EXPERIMENT'], f"run_{params['RUN']}") if 'PARENT_EXPERIMENT' in params and params['PARENT_EXPERIMENT'] != False else None
    
    #find file in correct folder
    f = os.path.join(path_to_save, f'population_{generation}.json')
    if(not path.isfile(f)):
        f = os.path.join(path_to_parent_experiment, f'population_{generation}.json') if path_to_parent_experiment is not None else f
    
    #laod file
    if(path.isfile(f)):
        population = json.load(open(f,'r'))
    else:
        raise Exception("Cannot open population file  of gen: ",  generation, '\n',
        "in folder: ", path_to_parent_experiment if path_to_parent_experiment is not None else path_to_save)
    return population

def save_archive(generation, archive):
    path_to_save = os.path.join(params['DUMPS_DIR'], params['EXPERIMENT_NAME'], f"run_{params['RUN']}")
    with open(os.path.join(path_to_save, f'z-archive_{generation}.json'), 'w') as f:
        json.dump(archive, f)


def load_archive(generation):
    path_to_save = os.path.join(params['DUMPS_DIR'], params['EXPERIMENT_NAME'], f"run_{params['RUN']}")
    path_to_parent_experiment = os.path.join(params['DUMPS_DIR'], params['PARENT_EXPERIMENT'], f"run_{params['RUN']}") if 'PARENT_EXPERIMENT' in params and params['PARENT_EXPERIMENT'] != False else None
    #find file in correct folder
    f = os.path.join(path_to_save, f'z-archive_{generation}.json')
    if(not path.isfile(f)):
        f = os.path.join(path_to_parent_experiment, f'z-archive_{generation}.json') if path_to_parent_experiment is not None else f
    
    #load file
    if(path.isfile(f)):
        archive = json.load(open(f,'r'))
    else:
        raise Exception("Cannot open archive file of gen: ", generation, '\n'
        "in folder: " , path_to_parent_experiment if path_to_parent_experiment is not None else path_to_save)   
    return archive

def save_parameters():
    path_to_save = os.path.join(params['DUMPS_DIR'], params['EXPERIMENT_NAME'], f"run_{params['RUN']}")
    params_lower = dict((k.lower(), v) for k, v in params.items())
    c = json.dumps(params_lower)
    open(os.path.join(path_to_save, '_parameters.json'), 'a').write(c)


def prepare_dumps():
    path_to_save = os.path.join(params['DUMPS_DIR'], params['EXPERIMENT_NAME'], f"run_{params['RUN']}")
    try:
        os.makedirs(path_to_save)
    except FileExistsError as e:
        pass
    save_parameters()

#extract number from file
def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)

#finds the latest generation recorded in previous running on the simualtion based on the last population recorded
def find_last_gen_recorded_in_folder(folder):

    filenames = glob.glob(os.path.join(folder, f'builtinstate_?*'))
    if(len(filenames) == 0): return None

    last_gen_name = max(filenames, key=extract_number)
    last_gen = extract_number(last_gen_name)
    return int(last_gen[0])


def find_last_generation_to_load():
    #Will find the most recent generation to load in case there are already some generations in the current experiment
    #else it will look for a parent experiment from which to load the data and change the experiment name so that now the folder of the parent expeirment is used to load the data 
    #if there is no data to load in any case it will return none
    path_to_parent_experiment = os.path.join(params['DUMPS_DIR'], params['PARENT_EXPERIMENT'], f"run_{params['RUN']}") if 'PARENT_EXPERIMENT' in params and params['PARENT_EXPERIMENT'] != False else None
    path_to_save = os.path.join(params['DUMPS_DIR'], params['EXPERIMENT_NAME'], f"run_{params['RUN']}")
    print('Looking for last generation to load in folder: ', path_to_save)
    last_gen = find_last_gen_recorded_in_folder(path_to_save)
    if last_gen == None and path_to_parent_experiment is not None: 
        last_gen = find_last_gen_recorded_in_folder(path_to_parent_experiment)
    return last_gen