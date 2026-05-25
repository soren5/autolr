from math import isclose
from operator import inv
import random
import sys
from xml.etree.ElementTree import tostring
from sge.grammar import grammar
import copy
from datetime import datetime
from sge.parameters import (
    params,
    set_parameters,
    manual_load_parameters
)
from sge.logger import find_last_generation_to_load
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate_level, mutate
from sge.operators.selection import tournament, universal_stochastic_sampling
import time
import statistics
from scipy import stats
import numpy as np
from utils.genotypes import *
from utils.smart_phenotype import smart_phenotype, single_task_key, readable_phenotype


def generate_random_individual():
    genotype = [[] for key in grammar.get_non_terminals()]
    tree_depth = grammar.recursive_individual_creation(genotype, grammar.get_start_rule()[0], 0)
    return {'genotype': genotype, 'fitness': None, 'tree_depth' : tree_depth, 'operation': "initialization"}

def make_initial_population():
    for i in range(params['POPSIZE']):
        yield generate_random_individual()

def initialize_population(solutions=[]):
    population = list(make_initial_population())
    archive = {}
    ii = 0
    for ii in range(len(solutions)):
        population[ii] = {"genotype": solutions[ii], 'id': ii, 'tree_depth': 0, 'fitness': None, 'operation': "prepopulation"}
    for i in range(len(population)):
        population[i]['id'] = i + ii
    return population

def start_population_from_scratch(solutions=[]):
    population = initialize_population(solutions)
    archive = {}
    for indiv in population:
        indiv["evaluations"] = [] 
        mapping_values = [0 for i in indiv['genotype']]
        phen, tree_depth = grammar.mapping(indiv['genotype'], mapping_values)
        indiv['phenotype'] = phen
        indiv['mapping_values'] = mapping_values
    id = len(population)
    counter = id - 1
    it = 0
    return population, archive, counter, it 

def evaluate(ind, eval_func):
    start = time.time()
    if 'phenotype' not in ind:
        mapping_values = [0 for i in ind['genotype']]
        phen, tree_depth = grammar.mapping(ind['genotype'], mapping_values)
    else:
        mapping_values = ind['mapping_values']
        phen = ind['phenotype']
        tree_depth = ind['tree_depth']
    other_info = {}
    #print(f"Registering {smart_phenotype(phen)}")
    if 'grad' in smart_phenotype(phen):
        quality, other_info = eval_func.evaluate(phen, params)
    else:
        quality = params['FITNESS_FLOOR']
        print(f"Fitness: {quality} (invalid detection)")
        other_info = {'source': 'invalid detection'}


    end = time.time()
    if "FAKE_FITNESS" in params and params['FAKE_FITNESS']:
        duration = random.random()
    else:
        duration = end - start
    ind['phenotype'] = phen 
    ind['fitness'] = quality
    ind['other_info'] = other_info
    ind['other_info']['duration'] = duration
    ind['mapping_values'] = mapping_values
    ind['tree_depth'] = tree_depth
    ind['key'] = single_task_key(ind['phenotype'], params['CURRENT_GEN'])



def setup(evaluation_function=None, parameters=None, logger=None):
    if parameters is None:
        set_parameters(sys.argv[1:])
    else:
        manual_load_parameters(parameters)
    print(params)

    #print(params)
    if 'SEED' not in params:
        params['SEED'] = int(datetime.now().microsecond)
    if logger is None:
        import sge.logger as logger
        print("Using Native Logger")
    if 'RESUME' in params and type(params['RESUME']) == str and params["RESUME"].isdecimal():
        params["RESUME"] = int(params["RESUME"])
    logger.params = params 
    logger.prepare_dumps()
    
    random.seed(params['SEED'])
    np.random.seed(params['SEED'])
    
    grammar.set_path(params['GRAMMAR'])
    grammar.read_grammar()
    grammar.set_max_tree_depth(params['MAX_TREE_DEPTH'])
    grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])
    
    if evaluation_function != None and not ("FAKE_FITNESS" in params and params['FAKE_FITNESS']):
        evaluation_function.init_net(params)
        evaluation_function.init_data(params)
        evaluation_function.init_evaluation(params)




def evolutionary_algorithm(evaluation_function=None, parameters=None, logger_module=None):
    import os
    
    logger = read_params(parameters, logger_module)
    setup(evaluation_function, parameters, logger_module)

    #check_google_colab(params, logger)


    population, archive, counter, it = initialize_pop(logger)

    print(params)
    
    return run_evolution(evaluation_function, logger, population, archive, counter, it)

def run_evolution(evaluation_function, logger, population, archive, counter, it):
    
    start_time = time.time()
    
    while simulation_is_running(it, start_time):
        
        print(f"{it}")
        params["CURRENT_GEN"] = it
        evaluation_function, population, archive, it, pre_race_snapshot = update_archive_and_fitness(evaluation_function, population, archive, it, logger)
        
        save_data(logger, population, it)


        logger, population, archive, counter, it = reproduction_and_elitism(logger, population, archive, counter, it, pre_race_snapshot)

    return population

def update_archive_and_fitness(evaluation_function, population, archive, it, logger=None):
    for i in range(len(population)):
        indiv = population[i]
        print(f"Individual {i}/{len(population)}")
        evaluation_function, archive, indiv = update_archive(evaluation_function, archive, indiv, it)
    pre_race_snapshot = build_pre_race_snapshot(population, archive)
    population, archive = update_best_fitness(population, archive, evaluation_function, logger, it, pre_race_snapshot)
       
    for indiv in population:
        archive, indiv = update_key_and_fitness_based_on_archive(archive, indiv)

    population, it  = sort_pop_and_print_best_fit(population, it)
    return evaluation_function, population, archive, it, pre_race_snapshot

def simulation_is_over(it):
    return it == params['GENERATIONS']

def simulation_is_running(it, start_time):
    return it < params['GENERATIONS'] and (True if 'TIME_STOP' not in params else (True if time.time() - start_time < params['TIME_STOP'] else False))

def reproduction_and_elitism(logger, population, archive, counter, it, pre_race_snapshot=None):
    audit_summary = make_selection_audit_summary()
    new_population, population, audit_summary = reproduce_via_elitism(population, logger, it, pre_race_snapshot, audit_summary)
    logger, population, archive, counter, it, new_population, audit_summary = reproduction(logger, population, archive, counter, it, new_population, pre_race_snapshot, audit_summary)
    write_selection_audit_summary(logger, it - 1, audit_summary)
    return logger, population, archive, counter, it

def save_data(logger, population, it):
    logger.evolution_progress(it, population)
    logger.elicit_progress(it, population)

def sort_pop_and_print_best_fit(population, it):
    population = sort_pop_based_on_fitness(population)
    print("\ngeneration: " + str(it) + "; best fit so far: " + str(population[0]['fitness']) + "\n")
    return population, it

def read_params(parameters, logger_module):
    if logger_module != None:
        logger = logger_module
    else:
        import sge.logger as logger
    return logger

def check_google_colab(params, logger):
    if "COLAB" in params and params["COLAB"]:
        from google.colab import drive
        drive.mount('/content/drive')
    if 'RESUME' in params:
        population = logger.load_population(params['RESUME'])
        if population != None:
            if params['LOAD_ARCHIVE']:
                archive = logger.load_archive(params['RESUME'])
            else:
                archive = {}
            logger.load_random_state(params['RESUME'])
            it = params['RESUME']
            counter = int(np.max([archive[x]['id'] for x in archive]))

def reproduction(logger, population, archive, counter, it, new_population, pre_race_snapshot=None, audit_summary=None):
    selection_index = 0
    while len(new_population) < params['POPSIZE']:
        new_indiv = selection(population, logger, it, pre_race_snapshot, selection_index, audit_summary)
        selection_index += 1
        new_indiv_2 = selection(population, logger, it, pre_race_snapshot, selection_index, audit_summary) 
        selection_index += 1
        #print(new_indiv)
        new_indiv = crossover(new_indiv, new_indiv_2, params['PROB_CROSSOVER'])
        new_indiv = mutation(new_indiv)
        new_indiv = map_phenotype(new_indiv)
        archive, counter, new_population, new_indiv = update_archive_with_new_indiv(archive, counter, new_population, new_indiv)
    it, population = go_to_next_generation(it, new_population)
    save_data_new_pop(logger, population, archive, it)
    return logger, population, archive, counter, it, new_population, audit_summary

def selection(population, logger=None, generation=None, pre_race_snapshot=None, selection_index=None, audit_summary=None):
    if params['SELECTION_TYPE'] == 'tournament':
        new_indiv = tournament_selection(population, logger, generation, pre_race_snapshot, selection_index, audit_summary)
    elif params['SELECTION_TYPE'] == 'stochastic':
        new_indiv = universal_stochastic_sampling(population)
    return new_indiv

def go_to_next_generation(it, new_population):
    population = new_population
    it += 1
    return it, population


def save_data_new_pop(logger, population, archive, it):
    logger.save_archive(it, archive)
    logger.save_population(it, population)
    logger.save_random_state(it)

def update_archive_with_new_indiv(archive, counter, new_population, new_indiv):
    key = single_task_key(new_indiv['phenotype'], params['CURRENT_GEN'])
    if key in archive:
        new_indiv['id'] = archive[key]['id']
    else:
        counter += 1
        new_indiv['id'] = counter
    new_population.append(new_indiv)
    return archive, counter, new_population, new_indiv

def map_phenotype(new_indiv):
    mapping_values = [0 for i in new_indiv['genotype']]
    phen, tree_depth = grammar.mapping(new_indiv['genotype'], mapping_values)
    new_indiv['phenotype'] = phen
    new_indiv['smart_phenotype'] = smart_phenotype(phen)
    return new_indiv

def mutation(new_indiv):
    if type(params['PROB_MUTATION']) == float:
        new_indiv = mutate(new_indiv, params['PROB_MUTATION'])
    elif type(params['PROB_MUTATION']) == dict:
        assert len(params['PROB_MUTATION']) == len(new_indiv['genotype'])
        new_indiv = mutate_level(new_indiv, params['PROB_MUTATION'])
    else:
        raise Exception("Invalid mutation type")
    return new_indiv

def tournament_selection(population, logger=None, generation=None, pre_race_snapshot=None, selection_index=None, audit_summary=None):
    #if random.random() < params['PROB_CROSSOVER']:
    #    p1 = tournament(population, params['TSIZE'])
    #    p2 = tournament(population, params['TSIZE'])
    #    new_indiv = crossover(p1, p2)
    #else:
    pool = random.sample(population, params['TSIZE'])
    if any(ind['fitness'] == None for ind in pool):
        raise "Some individuals have no fitness at the moment of selection"
    pool.sort(key=lambda i: i['fitness'])
    new_indiv = copy.deepcopy(pool[0])
    new_indiv["operation"] = "copy"
    new_indiv["parent"] = [new_indiv['id']]
    audit_tournament_selection(logger, generation, selection_index, pool, new_indiv, pre_race_snapshot, audit_summary)
    return new_indiv

def reproduce_via_elitism(population, logger=None, generation=None, pre_race_snapshot=None, audit_summary=None):
    behaviors_added = []
    new_population = []
    print("[ELITE] Adding following individuals to new population:")
    for indiv in population:
        if indiv['smart_phenotype'] not in behaviors_added and len(behaviors_added) < params['ELITISM']:
            behaviors_added.append(indiv['smart_phenotype'])
            new_population.append(indiv)
            print(readable_phenotype(indiv['phenotype']))
    while len(new_population) < params['ELITISM']:
        print("[ELITE] Could not fill with unique individuals, adding best individual again.")
        new_population.append(population[0])

    for indiv in new_population:
        indiv['operation'] = 'elitism'
        if 'other_info' not in indiv:
            indiv['other_info'] = {}
        indiv['other_info']['source'] = 'elitism'
    audit_elitism(logger, generation, population, new_population, pre_race_snapshot, audit_summary)
    return new_population, population, audit_summary

def sort_pop_based_on_fitness(population):
    population.sort(key=lambda x: x['fitness'])
    return population

def audit_tournament_selection(logger, generation, selection_index, pool, actual_parent, pre_race_snapshot, audit_summary):
    if not selection_audit_enabled() or audit_summary is None:
        return
    event = {
        'generation': generation,
        'event': 'tournament_audit',
        'selection_index': selection_index,
        'pool_ids': [indiv['id'] for indiv in pool],
        'pool_keys': [single_task_key(indiv['phenotype'], params['CURRENT_GEN']) for indiv in pool],
        'actual_parent_id': actual_parent['id'],
        'actual_parent_key': single_task_key(actual_parent['phenotype'], params['CURRENT_GEN']),
        'actual_fitness': actual_parent['fitness'],
        'audit_available': True,
    }

    if pre_race_snapshot is None:
        event.update(empty_counterfactual_parent_fields())
        audit_summary['audit_unavailable_count'] += 1
        write_selection_audit_event(logger, event)
        return

    pool_keys = event['pool_keys']
    if any(key not in pre_race_snapshot for key in pool_keys):
        event.update(empty_counterfactual_parent_fields())
        audit_summary['audit_unavailable_count'] += 1
        write_selection_audit_event(logger, event)
        return

    counterfactual = min(pool, key=lambda indiv: pre_race_snapshot[single_task_key(indiv['phenotype'], params['CURRENT_GEN'])]['pre_race_fitness'])
    counterfactual_key = single_task_key(counterfactual['phenotype'], params['CURRENT_GEN'])
    actual_key = event['actual_parent_key']
    outcome_changed = counterfactual['id'] != actual_parent['id']
    event.update({
        'counterfactual_parent_id': counterfactual['id'],
        'counterfactual_parent_key': counterfactual_key,
        'counterfactual_fitness': counterfactual['fitness'],
        'actual_pre_race_fitness': pre_race_snapshot[actual_key]['pre_race_fitness'],
        'counterfactual_pre_race_fitness': pre_race_snapshot[counterfactual_key]['pre_race_fitness'],
        'outcome_changed': outcome_changed,
    })
    audit_summary['tournament_events'] += 1
    if outcome_changed:
        audit_summary['tournament_changed'] += 1
    write_selection_audit_event(logger, event)

def empty_counterfactual_parent_fields():
    return {
        'counterfactual_parent_id': None,
        'counterfactual_parent_key': None,
        'counterfactual_fitness': None,
        'actual_pre_race_fitness': None,
        'counterfactual_pre_race_fitness': None,
        'outcome_changed': False,
        'audit_available': False,
    }

def audit_elitism(logger, generation, population, actual_elites, pre_race_snapshot, audit_summary):
    if not selection_audit_enabled() or audit_summary is None:
        return
    actual_keys = [single_task_key(indiv['phenotype'], params['CURRENT_GEN']) for indiv in actual_elites]
    event = {
        'generation': generation,
        'event': 'elitism_audit',
        'elitism_count': params['ELITISM'],
        'actual_elite_ids': [indiv['id'] for indiv in actual_elites],
        'actual_elite_keys': actual_keys,
        'audit_available': True,
    }

    if pre_race_snapshot is None:
        event.update(empty_counterfactual_elite_fields())
        audit_summary['audit_unavailable_count'] += 1
        write_selection_audit_event(logger, event)
        return

    population_keys = [single_task_key(indiv['phenotype'], params['CURRENT_GEN']) for indiv in population]
    if any(key not in pre_race_snapshot for key in population_keys):
        event.update(empty_counterfactual_elite_fields())
        audit_summary['audit_unavailable_count'] += 1
        write_selection_audit_event(logger, event)
        return

    counterfactual_elites = choose_counterfactual_elites(population, pre_race_snapshot)
    counterfactual_keys = [single_task_key(indiv['phenotype'], params['CURRENT_GEN']) for indiv in counterfactual_elites]
    counterfactual_ids = [indiv['id'] for indiv in counterfactual_elites]
    elite_set_changed = set(actual_keys) != set(counterfactual_keys)
    elite_order_changed = actual_keys != counterfactual_keys
    changed_positions = [
        index
        for index, (actual_key, counterfactual_key) in enumerate(zip(actual_keys, counterfactual_keys))
        if actual_key != counterfactual_key
    ]
    event.update({
        'counterfactual_elite_ids': counterfactual_ids,
        'counterfactual_elite_keys': counterfactual_keys,
        'elite_set_changed': elite_set_changed,
        'elite_order_changed': elite_order_changed,
        'changed_positions': changed_positions,
    })
    audit_summary['elitism_changed'] = elite_set_changed
    audit_summary['elitism_order_changed'] = elite_order_changed
    write_selection_audit_event(logger, event)

def empty_counterfactual_elite_fields():
    return {
        'counterfactual_elite_ids': [],
        'counterfactual_elite_keys': [],
        'elite_set_changed': False,
        'elite_order_changed': False,
        'changed_positions': [],
        'audit_available': False,
    }

def choose_counterfactual_elites(population, pre_race_snapshot):
    sorted_population = sorted(
        population,
        key=lambda indiv: pre_race_snapshot[single_task_key(indiv['phenotype'], params['CURRENT_GEN'])]['pre_race_fitness'],
    )
    behaviors_added = []
    counterfactual_elites = []
    for indiv in sorted_population:
        if indiv['smart_phenotype'] not in behaviors_added and len(behaviors_added) < params['ELITISM']:
            behaviors_added.append(indiv['smart_phenotype'])
            counterfactual_elites.append(indiv)
    while len(counterfactual_elites) < params['ELITISM']:
        counterfactual_elites.append(sorted_population[0])
    return counterfactual_elites

def update_key_and_fitness_based_on_archive(archive, indiv):
    key = update_key(indiv)
    update_fitness_based_on_archive(archive, indiv, key)
    return archive, indiv

def update_fitness_based_on_archive(archive, indiv, key):
    indiv['fitness'] = archive[key]['fitness']
    if params.get('RACING', False):
        indiv['trials'] = list(archive[key]['evaluations'])
        sync_multi_task_trials_to_individual(archive[key], indiv)
    if 'other_info' not in indiv:
        indiv['other_info'] = {}
    if 'source' not in indiv['other_info']:
        indiv['other_info']['source'] = 'archive'

def sync_multi_task_trials_to_individual(archive_entry, indiv):
    if 'multi_task_trials' not in archive_entry:
        return
    indiv['task_trials'] = copy.deepcopy(archive_entry.get('task_evaluations', {}))
    indiv['task_pass_trials'] = copy.deepcopy(archive_entry.get('task_passes', {}))
    indiv['reached_depth_trials'] = list(archive_entry.get('reached_depths', []))
    indiv['failed_task_trials'] = list(archive_entry.get('failed_tasks', []))
    indiv['multi_task_trials'] = copy.deepcopy(archive_entry.get('multi_task_trials', []))


def update_key(indiv):
    key = single_task_key(indiv['phenotype'], params['CURRENT_GEN'])
    return key

def build_pre_race_snapshot(population, archive):
    snapshot = {}
    for indiv in population:
        key = single_task_key(indiv['phenotype'], params['CURRENT_GEN'])
        if key not in archive:
            raise Exception(f"Missing archive entry for pre-race snapshot: {key}")
        if key not in snapshot:
            evaluations = archive[key]['evaluations']
            snapshot[key] = {
                'key': key,
                'archive_id': archive[key]['id'],
                'population_ids': [],
                'pre_race_fitness': archive[key]['fitness'],
                'first_fitness': evaluations[0] if len(evaluations) > 0 else None,
                'n_evals_before': len(evaluations),
                'valid': 'grad' in key,
                'initial_rank': None,
            }
        snapshot[key]['population_ids'].append(indiv['id'])

    valid_keys = [key for key, record in snapshot.items() if record['valid']]
    valid_keys.sort(key=lambda key: snapshot[key]['pre_race_fitness'])
    for rank, key in enumerate(valid_keys, start=1):
        snapshot[key]['initial_rank'] = rank
    return snapshot

def racing_logging_enabled():
    return params.get('RACING', False) and params.get('RACING_LOGGING', True)

def selection_audit_enabled():
    return params.get('RACING', False) and params.get('RACING_SELECTION_AUDIT', True)

def write_f_race_event(logger, event):
    if racing_logging_enabled() and logger is not None and hasattr(logger, 'f_race_event'):
        logger.f_race_event(event)

def write_f_race_summary(logger, row):
    if racing_logging_enabled() and logger is not None and hasattr(logger, 'f_race_summary'):
        logger.f_race_summary(row)

def write_selection_audit_event(logger, event):
    if selection_audit_enabled() and logger is not None and hasattr(logger, 'selection_audit_event'):
        logger.selection_audit_event(event)

def make_selection_audit_summary():
    return {
        'elitism_changed': False,
        'elitism_order_changed': False,
        'tournament_events': 0,
        'tournament_changed': 0,
        'audit_unavailable_count': 0,
    }

def write_selection_audit_summary(logger, generation, summary):
    if not selection_audit_enabled() or logger is None or not hasattr(logger, 'selection_audit_summary'):
        return
    events = summary['tournament_events']
    changed_rate = summary['tournament_changed'] / events if events else 0
    logger.selection_audit_summary({
        'generation': generation,
        'elitism_changed': summary['elitism_changed'],
        'elitism_order_changed': summary['elitism_order_changed'],
        'tournament_events': events,
        'tournament_changed': summary['tournament_changed'],
        'tournament_changed_rate': changed_rate,
        'audit_unavailable_count': summary['audit_unavailable_count'],
    })

def update_best_fitness(population, archive, evaluation_function=None, logger=None, generation=None, pre_race_snapshot=None):
    if not params.get('RACING', False):
        return population, archive
    if evaluation_function is None:
        raise Exception("RACING requires an evaluation function")
    if params.get('RACING_STAT_TEST') != 'mannwhitney':
        raise Exception("Only Mann-Whitney F-race is supported")

    race_individuals = collect_race_individuals(population, archive)
    if len(race_individuals) <= 1:
        write_empty_f_race_summary(logger, generation, race_individuals, pre_race_snapshot, 'not_enough_candidates')
        return population, archive

    remaining_keys = set(race_individuals.keys())
    initial_best_key = min(remaining_keys, key=lambda key: archive[key]['fitness'])
    extra_evaluations = 0
    eliminated_count = 0
    stop_reason = 'unknown'
    round_number = 0
    log_race_start(logger, generation, race_individuals, archive, pre_race_snapshot, initial_best_key)
    while len(remaining_keys) > 1:
        round_number += 1
        best_key = min(remaining_keys, key=lambda key: archive[key]['fitness'])
        remaining_keys, round_eliminated = eliminate_clearly_worse_candidates(archive, remaining_keys, best_key, logger, generation, round_number)
        eliminated_count += round_eliminated

        if len(remaining_keys) <= 1:
            stop_reason = 'single_remaining'
            break

        evaluated_any = False
        for key in list(remaining_keys):
            if candidate_can_be_reevaluated(archive, key):
                archive = reevaluate_race_candidate(evaluation_function, archive, race_individuals[key], logger, generation, round_number)
                evaluated_any = True
                extra_evaluations += 1

        if not evaluated_any:
            if all(len(archive[key]['evaluations']) >= params['RACING_MAX_EVALS'] for key in remaining_keys):
                stop_reason = 'max_evals'
            else:
                stop_reason = 'no_evaluable_candidates'
            break

    if stop_reason == 'unknown':
        stop_reason = 'single_remaining' if len(remaining_keys) <= 1 else 'max_evals'
    log_race_stop(logger, generation, archive, remaining_keys, race_individuals, pre_race_snapshot, initial_best_key, extra_evaluations, eliminated_count, stop_reason)
    return population, archive

def collect_race_individuals(population, archive):
    race_individuals = {}
    for indiv in population:
        key = single_task_key(indiv['phenotype'], params['CURRENT_GEN'])
        if key not in archive:
            raise Exception(f"Missing archive entry for race candidate: {key}")
        if 'grad' not in key:
            continue
        if key not in race_individuals:
            race_individuals[key] = indiv
    return race_individuals

def candidate_can_be_reevaluated(archive, key):
    return (
        len(archive[key]['evaluations']) < params['RACING_MAX_EVALS']
        and not candidate_has_no_reevaluation_score(archive, key)
    )

def candidate_has_no_reevaluation_score(archive, key):
    return any(-0.2 <= evaluation <= 0.0 for evaluation in archive[key]['evaluations'])

def log_race_start(logger, generation, race_individuals, archive, pre_race_snapshot, initial_best_key):
    if not racing_logging_enabled():
        return
    invalid_count = count_invalid_snapshot_records(pre_race_snapshot)
    write_f_race_event(logger, {
        'generation': generation,
        'event': 'race_start',
        'candidate_count': len(pre_race_snapshot) if pre_race_snapshot is not None else len(race_individuals),
        'eligible_count': len(race_individuals),
        'invalid_count': invalid_count,
        'max_evals': params['RACING_MAX_EVALS'],
        'alpha': params['RACING_ALPHA'],
        'stat_test': params['RACING_STAT_TEST'],
        'initial_best_key': initial_best_key,
        'initial_best_mean': archive[initial_best_key]['fitness'],
    })
    if pre_race_snapshot is None:
        snapshot_keys = sorted(race_individuals.keys(), key=lambda key: archive[key]['fitness'])
    else:
        snapshot_keys = sorted(
            pre_race_snapshot.keys(),
            key=lambda key: (
                pre_race_snapshot[key]['initial_rank'] is None,
                pre_race_snapshot[key]['initial_rank'] if pre_race_snapshot[key]['initial_rank'] is not None else 0,
                key,
            ),
        )
    for key in snapshot_keys:
        snapshot_record = pre_race_snapshot[key] if pre_race_snapshot is not None and key in pre_race_snapshot else {}
        population_ids = snapshot_record.get('population_ids')
        if population_ids is None:
            population_ids = [race_individuals[key]['id']]
        event = {
            'generation': generation,
            'event': 'candidate_snapshot',
            'key': key,
            'archive_id': archive[key]['id'],
            'population_ids': population_ids,
            'n_evals_before': len(archive[key]['evaluations']),
            'first_fitness': archive[key]['evaluations'][0] if archive[key]['evaluations'] else None,
            'mean_before': archive[key]['fitness'],
            'initial_rank': snapshot_record.get('initial_rank'),
            'valid': snapshot_record.get('valid', key in race_individuals),
        }
        add_latest_multi_task_record_to_event(event, archive.get(key, {}))
        write_f_race_event(logger, event)

def count_invalid_snapshot_records(pre_race_snapshot):
    if pre_race_snapshot is None:
        return 0
    return len([record for record in pre_race_snapshot.values() if not record['valid']])

def write_empty_f_race_summary(logger, generation, race_individuals, pre_race_snapshot, stop_reason):
    if not racing_logging_enabled():
        return
    initial_best_key = None
    if race_individuals:
        initial_best_key = next(iter(race_individuals.keys()))
    write_f_race_event(logger, {
        'generation': generation,
        'event': 'race_start',
        'candidate_count': len(pre_race_snapshot) if pre_race_snapshot is not None else len(race_individuals),
        'eligible_count': len(race_individuals),
        'invalid_count': count_invalid_snapshot_records(pre_race_snapshot),
        'max_evals': params['RACING_MAX_EVALS'],
        'alpha': params['RACING_ALPHA'],
        'stat_test': params['RACING_STAT_TEST'],
        'initial_best_key': initial_best_key,
        'initial_best_mean': None,
    })
    if pre_race_snapshot is not None:
        for key in sorted(pre_race_snapshot.keys()):
            record = pre_race_snapshot[key]
            event = {
                'generation': generation,
                'event': 'candidate_snapshot',
                'key': key,
                'archive_id': record['archive_id'],
                'population_ids': record['population_ids'],
                'n_evals_before': record['n_evals_before'],
                'first_fitness': record['first_fitness'],
                'mean_before': record['pre_race_fitness'],
                'initial_rank': record['initial_rank'],
                'valid': record['valid'],
            }
            write_f_race_event(logger, event)
    write_f_race_event(logger, {
        'generation': generation,
        'event': 'race_stop',
        'reason': stop_reason,
        'winner_key': initial_best_key,
        'initial_best_key': initial_best_key,
        'final_best_key': initial_best_key,
        'initial_best_changed': False,
        'extra_evaluations': 0,
        'remaining_count': len(race_individuals),
        'eliminated_count': 0,
        'max_evals_hit_count': 0,
        'winner_evals': 0,
    })
    write_f_race_summary(logger, {
        'generation': generation,
        'eligible_count': len(race_individuals),
        'invalid_count': count_invalid_snapshot_records(pre_race_snapshot),
        'extra_evaluations': 0,
        'initial_best_key': initial_best_key,
        'final_best_key': initial_best_key,
        'initial_best_changed': False,
        'eliminated_count': 0,
        'max_evals_hit_count': 0,
        'winner_evals': 0,
        'stop_reason': stop_reason,
    })

def log_race_stop(logger, generation, archive, remaining_keys, race_individuals, pre_race_snapshot, initial_best_key, extra_evaluations, eliminated_count, stop_reason):
    final_best_key = min(race_individuals.keys(), key=lambda key: archive[key]['fitness'])
    max_evals_hit_count = len([key for key in race_individuals if len(archive[key]['evaluations']) >= params['RACING_MAX_EVALS']])
    winner_evals = len(archive[final_best_key]['evaluations'])
    event = {
        'generation': generation,
        'event': 'race_stop',
        'reason': stop_reason,
        'winner_key': final_best_key,
        'initial_best_key': initial_best_key,
        'final_best_key': final_best_key,
        'initial_best_changed': initial_best_key != final_best_key,
        'extra_evaluations': extra_evaluations,
        'remaining_count': len(remaining_keys),
        'eliminated_count': eliminated_count,
        'max_evals_hit_count': max_evals_hit_count,
        'winner_evals': winner_evals,
    }
    write_f_race_event(logger, event)
    write_f_race_summary(logger, {
        'generation': generation,
        'eligible_count': len(race_individuals),
        'invalid_count': count_invalid_snapshot_records(pre_race_snapshot),
        'extra_evaluations': extra_evaluations,
        'initial_best_key': initial_best_key,
        'final_best_key': final_best_key,
        'initial_best_changed': initial_best_key != final_best_key,
        'eliminated_count': eliminated_count,
        'max_evals_hit_count': max_evals_hit_count,
        'winner_evals': winner_evals,
        'stop_reason': stop_reason,
    })

def eliminate_clearly_worse_candidates(archive, remaining_keys, best_key, logger=None, generation=None, round_number=None):
    best_evaluations = archive[best_key]['evaluations']
    kept_keys = set()
    eliminated_count = 0
    for key in remaining_keys:
        if key == best_key:
            kept_keys.add(key)
            continue

        is_worse, p_value, comparison_metadata = candidate_is_clearly_worse_by_rule(archive, best_key, key, best_evaluations)
        if is_worse:
            print(f"[F-RACE] Eliminating candidate {key}")
            eliminated_count += 1
            event = {
                'generation': generation,
                'event': 'elimination',
                'round': round_number,
                'key': key,
                'best_key': best_key,
                'candidate_mean': archive[key]['fitness'],
                'best_mean': archive[best_key]['fitness'],
                'candidate_n': len(archive[key]['evaluations']),
                'best_n': len(best_evaluations),
                'p_value': p_value,
                'reason': comparison_metadata.get('reason', 'clearly_worse'),
            }
            event.update(comparison_metadata)
            write_f_race_event(logger, event)
        else:
            kept_keys.add(key)
    return kept_keys, eliminated_count

def candidate_is_clearly_worse_by_rule(archive, best_key, candidate_key, best_evaluations):
    if params.get('RACING_DECISION_RULE', 'scalar_mannwhitney') == 'gated_cascade':
        is_worse, p_value, metadata = candidate_is_clearly_worse_gated(archive, best_key, candidate_key)
        if is_worse:
            return is_worse, p_value, metadata
        if metadata.get('comparison_type') not in ('insufficient_gated_evidence', 'scalar_fallback'):
            return is_worse, p_value, metadata
    is_worse, p_value = candidate_is_clearly_worse(archive, best_key, candidate_key, best_evaluations)
    return is_worse, p_value, {
        'reason': 'clearly_worse',
        'comparison_type': 'scalar_fallback' if params.get('RACING_DECISION_RULE') == 'gated_cascade' else 'scalar_mannwhitney',
        'comparison_task': None,
    }

def candidate_is_clearly_worse(archive, best_key, candidate_key, best_evaluations):
    candidate_evaluations = archive[candidate_key]['evaluations']
    if len(best_evaluations) < params['RACING_MIN_EVALS']:
        return False, None
    if len(candidate_evaluations) < params['RACING_MIN_EVALS']:
        return False, None
    if archive[candidate_key]['fitness'] <= archive[best_key]['fitness']:
        return False, None

    try:
        _, p_value = stats.mannwhitneyu(best_evaluations, candidate_evaluations)
    except ValueError:
        p_value = 1
    return bool(p_value < params['RACING_ALPHA']), p_value

def candidate_is_clearly_worse_gated(archive, best_key, candidate_key):
    best_entry = archive[best_key]
    candidate_entry = archive[candidate_key]
    if not has_multi_task_trials(best_entry) or not has_multi_task_trials(candidate_entry):
        return False, None, {
            'reason': 'missing_multi_task_trials',
            'comparison_type': 'scalar_fallback',
            'comparison_task': None,
        }

    compared_any_gated_evidence = False
    is_worse, p_value, metadata = compare_reached_depths(best_entry, candidate_entry)
    compared_any_gated_evidence = compared_any_gated_evidence or metadata.get('comparison_available', False)
    if is_worse:
        return True, p_value, metadata

    for task in multi_task_task_order(best_entry, candidate_entry):
        is_worse, p_value, metadata = compare_task_scores(best_entry, candidate_entry, task)
        compared_any_gated_evidence = compared_any_gated_evidence or metadata.get('comparison_available', False)
        if is_worse:
            return True, p_value, metadata

    if compared_any_gated_evidence:
        return False, None, {
            'reason': 'gated_cascade_not_clearly_worse',
            'comparison_type': 'gated_cascade_not_clearly_worse',
            'comparison_task': None,
        }

    return False, None, {
        'reason': 'insufficient_gated_evidence',
        'comparison_type': 'insufficient_gated_evidence',
        'comparison_task': None,
    }

def has_multi_task_trials(archive_entry):
    return bool(archive_entry.get('multi_task_trials')) and bool(archive_entry.get('reached_depths'))

def compare_reached_depths(best_entry, candidate_entry):
    best_depths = [depth for depth in best_entry.get('reached_depths', []) if depth is not None]
    candidate_depths = [depth for depth in candidate_entry.get('reached_depths', []) if depth is not None]
    metadata = {
        'reason': 'gated_cascade_clearly_worse',
        'comparison_type': 'reached_depth',
        'comparison_task': None,
        'best_reached_depth_mean': statistics.mean(best_depths) if best_depths else None,
        'candidate_reached_depth_mean': statistics.mean(candidate_depths) if candidate_depths else None,
        'comparison_available': False,
    }
    if len(best_depths) < params['RACING_MIN_EVALS'] or len(candidate_depths) < params['RACING_MIN_EVALS']:
        return False, None, metadata
    metadata['comparison_available'] = True
    if metadata['candidate_reached_depth_mean'] >= metadata['best_reached_depth_mean']:
        return False, None, metadata
    try:
        _, p_value = stats.mannwhitneyu(best_depths, candidate_depths)
    except ValueError:
        p_value = 1
    return bool(p_value < params['RACING_ALPHA']), p_value, metadata

def compare_task_scores(best_entry, candidate_entry, task):
    best_scores = best_entry.get('task_evaluations', {}).get(task, [])
    candidate_scores = candidate_entry.get('task_evaluations', {}).get(task, [])
    metadata = {
        'reason': 'gated_cascade_clearly_worse',
        'comparison_type': 'task_score',
        'comparison_task': task,
        'best_task_score_mean': statistics.mean(best_scores) if best_scores else None,
        'candidate_task_score_mean': statistics.mean(candidate_scores) if candidate_scores else None,
        'comparison_available': False,
    }
    if len(best_scores) < params['RACING_MIN_EVALS'] or len(candidate_scores) < params['RACING_MIN_EVALS']:
        return False, None, metadata
    metadata['comparison_available'] = True
    if metadata['candidate_task_score_mean'] >= metadata['best_task_score_mean']:
        return False, None, metadata
    try:
        _, p_value = stats.mannwhitneyu(best_scores, candidate_scores)
    except ValueError:
        p_value = 1
    return bool(p_value < params['RACING_ALPHA']), p_value, metadata

def multi_task_task_order(best_entry, candidate_entry):
    if best_entry.get('task_order'):
        return best_entry['task_order']
    if candidate_entry.get('task_order'):
        return candidate_entry['task_order']
    best_trials = best_entry.get('multi_task_trials', [])
    if best_trials and best_trials[0].get('task_order'):
        return best_trials[0]['task_order']
    candidate_trials = candidate_entry.get('multi_task_trials', [])
    if candidate_trials and candidate_trials[0].get('task_order'):
        return candidate_trials[0]['task_order']
    return []

def reevaluate_race_candidate(evaluation_function, archive, indiv, logger=None, generation=None, round_number=None):
    key = single_task_key(indiv['phenotype'], params['CURRENT_GEN'])
    print(f"[F-RACE] Re-evaluating candidate {key}; evaluation #{len(archive[key]['evaluations']) + 1}")
    mean_before = archive[key]['fitness']
    evaluate(indiv, evaluation_function)
    archive[key]['evaluations'].append(indiv['fitness'])
    archive[key]['fitness'] = statistics.mean(archive[key]['evaluations'])
    append_multi_task_trial_to_archive(archive[key], extract_multi_task_record(indiv))
    event = {
        'generation': generation,
        'event': 'reevaluation',
        'round': round_number,
        'key': key,
        'archive_id': archive[key]['id'],
        'eval_number': len(archive[key]['evaluations']),
        'new_fitness': indiv['fitness'],
        'n_evals_before': len(archive[key]['evaluations']) - 1,
        'n_evals_after': len(archive[key]['evaluations']),
        'mean_before': mean_before,
        'mean_after': archive[key]['fitness'],
    }
    add_latest_multi_task_record_to_event(event, archive[key])
    write_f_race_event(logger, event)
    return archive

def update_archive(evaluation_function, archive, indiv, it):
    indiv['smart_phenotype'] = smart_phenotype(indiv['phenotype'])
    key = single_task_key(indiv['phenotype'], params['CURRENT_GEN'])
    if key in archive and 'fitness' not in archive[key]:
        raise Exception('Incomplete archive entry')
    if key not in archive:
        archive[key] = {'evaluations': []}
        archive[key]['id'] = indiv['id']
        evaluate(indiv, evaluation_function)
        archive[key]['evaluations'].append(indiv['fitness'])
        archive[key]['fitness'] = statistics.mean(archive[key]['evaluations'])
        append_multi_task_trial_to_archive(archive[key], extract_multi_task_record(indiv))

    return evaluation_function, archive, indiv

def extract_multi_task_record(indiv):
    other_info = indiv.get('other_info', {})
    multi_task_record = other_info.get('multi_task')
    if isinstance(multi_task_record, dict):
        return copy.deepcopy(multi_task_record)
    return None

def append_multi_task_trial_to_archive(archive_entry, multi_task_record):
    if not isinstance(multi_task_record, dict):
        return
    task_order = multi_task_record.get('task_order', [])
    scores = multi_task_record.get('scores', {})
    passes = multi_task_record.get('passed', {})
    thresholds = multi_task_record.get('thresholds', {})

    archive_entry.setdefault('multi_task_trials', [])
    archive_entry.setdefault('task_evaluations', {})
    archive_entry.setdefault('task_passes', {})
    archive_entry.setdefault('reached_depths', [])
    archive_entry.setdefault('failed_tasks', [])
    archive_entry.setdefault('task_order', list(task_order))
    archive_entry.setdefault('task_thresholds', copy.deepcopy(thresholds))

    archive_entry['multi_task_trials'].append(copy.deepcopy(multi_task_record))
    archive_entry['reached_depths'].append(multi_task_record.get('reached_depth'))
    archive_entry['failed_tasks'].append(multi_task_record.get('failed_task'))

    for task in task_order:
        archive_entry['task_evaluations'].setdefault(task, [])
        archive_entry['task_passes'].setdefault(task, [])
        score = scores.get(task)
        passed = passes.get(task)
        if score is not None:
            archive_entry['task_evaluations'][task].append(score)
        if passed is not None:
            archive_entry['task_passes'][task].append(passed)

def add_latest_multi_task_record_to_event(event, archive_entry):
    trials = archive_entry.get('multi_task_trials', [])
    if trials:
        event['multi_task'] = copy.deepcopy(trials[-1])

def initialize_pop(logger):
    successful_resume = False
    if 'RESUME' in params:
        if params["RESUME"] == "Last":
            last_gen = find_last_generation_to_load()
            print(f"Resuming from last generation: {last_gen}")
        elif type(params["RESUME"]) == int:
            if params["RESUME"] != 0: 
                last_gen = params['RESUME']
            else: 
                last_gen = None
        else:
            raise Exception("Invalid RESUME")

        if last_gen != None:           
            population = logger.load_population(last_gen)
            logger.load_random_state(last_gen)
            if 'LOAD_ARCHIVE' in params and params['LOAD_ARCHIVE'] == True:     
                archive = logger.load_archive(last_gen)
            else:
                archive = {}
            counter = int(np.max([x['id'] for x in population]))
            it = last_gen
            successful_resume = True
    
    if not successful_resume:
        if 'PREPOPULATE' in params and params['PREPOPULATE']:
            genes_dict={
                #'all': [get_adam_genotype(params['GRAMMAR']), get_momentum_genotype(), get_rmsprop_genotype()],
                'adam': [get_adam_genotype(params['GRAMMAR'])],
                #'rmsprop': [get_rmsprop_genotype()],
                #'momentum': [get_momentum_genotype()],
            }
            population, archive, counter, it = start_population_from_scratch(genes_dict[params["PREPOPULATE"]])
        else:
            population, archive, counter, it = start_population_from_scratch()  
        for indiv in population:
            mapping_values = [0 for i in indiv['genotype']]
            phen, tree_depth = grammar.mapping(indiv['genotype'], mapping_values)
            indiv['phenotype'] = phen
            indiv['mapping_values'] = mapping_values
    if 'SINGLE_GEN' in params and params['SINGLE_GEN']:
        params['GENERATIONS'] = it + 1
    return population, archive, counter, it
