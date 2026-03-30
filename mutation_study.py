import pandas as pd
import os
from evaluators.evaluate_rastringin import Rastringin_Evaluator
from evaluators.evaluate_fmnist import FMNIST_Evaluator
import numpy as np

from sge.operators.mutation import mutate_one
from sge.parameters import reset_parameters, manual_load_parameters, params
import yaml
from sge.grammar import grammar
from utils.smart_phenotype import readable_phenotype, smart_phenotype, advanced_readable_phenotype
import seaborn as sns
import matplotlib.pyplot as plt
import ast

def init_mutation_study():

    return grammar, mutation_registry_df, archive, df
reset_parameters()
grammar._reset_grammar()
with open("parameters/base.yml", 'r') as ymlfile:
    parameters = yaml.load(ymlfile, Loader=yaml.FullLoader)
    parameters['EXPERIMENT_NAME'] = "mutation_study_fm"
    parameters['GRAMMAR'] = "grammars/original_optimizer.txt"
    #parameters['DATA_DIR'] = "/Users/soren/desktop_back_up/_Organized_Results/"
    #parameters['FAKE_FITNESS'] = True
manual_load_parameters(parameters=parameters)


# Check if dumps_dir + experiment_name exists, if it doesn't, create it.
if not os.path.exists(os.path.join(params['DUMPS_DIR'], params['EXPERIMENT_NAME'])):
    os.makedirs(os.path.join(params['DUMPS_DIR'], params['EXPERIMENT_NAME']))



df = pd.read_csv(os.path.join(params['DATA_DIR'], 'mutation_study_cache.csv'))

# Check if /mutation_registry.csv exists, if it does, load it into a dataframe, if it doesn't, create an empty dataframe with the appropriate columns
if os.path.exists(os.path.join(params['DATA_DIR'], 'mutation_registry_df.csv')):
    mutation_registry_df = pd.read_csv(os.path.join(params['DATA_DIR'], 'mutation_registry_df.csv'))
else:
    mutation_registry_df = pd.DataFrame(
        columns=[
            'original_phen_id', 
            'original_fitness', 
            'original_phenotype', 
            'original_genotype', 
            'mutation_target', 
            'mutation_log', 
            'original_other_info', 
            'phenotype', 
            'genotype', 
            'fitness', 
            'other_info'
            ])
valid_mutation_targets = {}
# get a dictionary with keys and non-terminals and values as the number of options in the production rule for that non-terminal
size_of_genes = grammar.count_number_of_options_in_production()
df = df.groupby('phen_id').first().reset_index()
# Load the archive from a pickle file if it exists
import os
import pickle
if os.path.exists('mutation_archive.pkl'):
    with open('mutation_archive.pkl', 'rb') as f:
        archive = pickle.load(f)
else:
    archive = {}

def assert_remap(remapped_phenotype, chosen_phenotype):
    remapped_phenotype = remapped_phenotype.replace(', shape=shape', '').replace(', dtype=float32', '').replace(', dtype=tf.float32', '')
    chosen_phenotype = chosen_phenotype.replace(', shape=shape', '').replace(', dtype=float32', '').replace(', dtype=tf.float32', '')
    try:
        assert remapped_phenotype == chosen_phenotype, "Remapped phenotype does not match original phenotype"
    except AssertionError as e:
        print(e)
        print("Chosen solution phenotype after cleaning: ", chosen_phenotype)
        print("Remapped        phenotype after cleaning: ", remapped_phenotype)

def mass_mutate_from_dataframe(
        mutation_registry_df, 
        grammar,
        archive,
        df,
        counter_limit=1000):
    
    # Initialize an empty list to store rows
    mutation_registry_rows = []
    counter = 0
    archive_accesses = 0

    size_of_genes = grammar.count_number_of_options_in_production()

    valid_solutions_for_mutation_target = {}
    # Go through all solutions and get their valid mutation targets.
    for ix, row in df[~df['genotype'].isna()].iterrows():
        genotype = ast.literal_eval(row['genotype'])
        mapping_values = [0 for i in genotype]
        phen, tree_depth = grammar.mapping(genotype, mapping_values)
        
        # get the mutable non-terminals, which are the ones that have more than one option in the production rule and have a non-empty genotype
        mutable_genes = [nt for index, nt in enumerate(grammar.get_non_terminals()) if size_of_genes[nt] != 1 and len(genotype[index]) > 0 and mapping_values[index] > 0]

        valid_mutation_targets[row['phen_id']] = mutable_genes

        for nt in mutable_genes:
            if nt not in valid_solutions_for_mutation_target:
                valid_solutions_for_mutation_target[nt] = []

            valid_solutions_for_mutation_target[nt].append({'phen_id': row['phen_id'], 'phenotype': row['phenotype'], 'genotype': row['genotype'], 'fitness': row['fitness']})


    not_represented_mutation_targets = []
    # Check if all the mutation targets in valid_mutation_targets are represented in the mutation_registry_df.
    for phen_id, mutation_targets in valid_mutation_targets.items():
        for mutation_target in mutation_targets:
            if mutation_target not in mutation_registry_df['mutation_target'].values:
                not_represented_mutation_targets.append(mutation_target)

    archive_accesses = 0
    fitness_for_mutation_target = {}
    mutation_counts = {}
    for mutation_target in grammar.get_non_terminals():
        if mutation_target != '<start>':            
            matching_solutions = valid_solutions_for_mutation_target[mutation_target]
            fitness_for_mutation_target[mutation_target] = df[df['phen_id'].isin([s['phen_id'] for s in matching_solutions])]['fitness']
            mutation_counts[mutation_target] = 0

    # Create a dictionary where the keys are non-terminals and the values are the solutions
    while counter < counter_limit:
        evaluator = FMNIST_Evaluator(params)

 
        # Look for the smallest value in the mutation_counts dictionary and get the corresponding mutation target as the least represented mutation target.
        least_represented_mutation_target = sorted(mutation_counts.items(), key=lambda x: x[1])[0][0]
        if mutation_counts[least_represented_mutation_target] >= 100:
            print(f"Sufficient samples")
            break
            
            #print(f"Least represented mutation target: {least_represented_mutation_target} with count {mutation_counts[least_represented_mutation_target]}")

        #least_represented_mutation_target = mutation_registry_df.groupby('mutation_target').size().sort_values().index[0]
        
        # Get a list of all solutions that have the least represented mutation target as a valid mutation target
        solutions_with_least_represented_mutation_target = valid_solutions_for_mutation_target[least_represented_mutation_target]
        #for phen_id, mutation_targets in valid_mutation_targets.items():
        #    if least_represented_mutation_target in mutation_targets:
        #        solutions_with_least_represented_mutation_target.append(phen_id)
        # Choose a solution based on a random selection, weighed by the fitness of the solutions
        fitnesses = fitness_for_mutation_target[least_represented_mutation_target]
        probabilities = fitnesses / fitnesses.sum()


        #chosen_solution = np.random.choice(solutions_with_least_represented_mutation_target, p=probabilities)
        chosen_solution = np.random.choice(solutions_with_least_represented_mutation_target)
        
        # Get the genotype and phenotype of the chosen solution
        chosen_genotype = ast.literal_eval(chosen_solution['genotype'])
        chosen_phenotype = chosen_solution['phenotype']
        # Remap the phenotype and double check that it matches the original phenotype
        mapping_values = [0 for i in chosen_genotype]
        remapped_phenotype, tree_depth = grammar.mapping(chosen_genotype, mapping_values)

        assert_remap(remapped_phenotype, chosen_phenotype)

        chosen_solution['smart_phenotype'] = smart_phenotype(chosen_solution['phenotype'])
        # Check if the chosen solution has been evaluated before, and if not, evaluate it and add it to the archive.
        if chosen_solution['smart_phenotype'] not in archive:
            o_fitness, original_other_info = evaluator.evaluate(chosen_phenotype)
            archive[chosen_solution['smart_phenotype']] = {
                'fitness': o_fitness, 
                'other_info': original_other_info}
        else:
            o_fitness = archive[chosen_solution['smart_phenotype']]['fitness']
            original_other_info = archive[chosen_solution['smart_phenotype']]['other_info']
            archive_accesses += 1

        # Mutate the chosen solution on the least represented mutation target
        chosen_indiv = {
            'genotype': chosen_genotype,
            'phenotype': chosen_phenotype,
            'mapping_values': mapping_values,
            'tree_depth': tree_depth,
            'operation': 'selection',
        }
        
        mutated_indiv, info = mutate_one(chosen_indiv, chosen_nt=least_represented_mutation_target)
        #mutated_indiv, info = mutate_one(chosen_indiv)


        # Map the mutated individual to get its phenotype
        mapping_values = [0 for i in mutated_indiv['genotype']]
        
        mutated_phen, tree_depth = grammar.mapping(mutated_indiv['genotype'], mutated_indiv['mapping_values'])
        mutated_indiv['mapping_values'] = mapping_values
        mutated_indiv['tree_depth'] = tree_depth
        mutated_indiv['operation'] = 'mutation'
        mutated_phen = mutated_phen.replace(', dtype=float32', '').replace(', dtype=tf.float32', '').replace(', shape=shape', '')
        mutated_indiv['phenotype'] = mutated_phen
        mutated_indiv['smart_phenotype'] = smart_phenotype(mutated_indiv['phenotype'])

        # Check if the mutated solution has been evaluated before, and if not, evaluate it and add it to the archive.
        if mutated_indiv['smart_phenotype'] not in archive:
            mutated_fitness, mutated_other_info = evaluator.evaluate(mutated_indiv['phenotype'])
            archive[mutated_indiv['smart_phenotype']] = {
                'fitness': mutated_fitness, 
                'other_info': mutated_other_info}
        else:
            mutated_fitness = archive[mutated_indiv['smart_phenotype']]['fitness']
            mutated_other_info = archive[mutated_indiv['smart_phenotype']]['other_info']

        # Add a row to the mutation_registry_df with the original solution, the mutated solution, the mutation target, and the fitnesses of both solutions.
        mutation_registry_rows.append({
            'original_phen_id': chosen_solution['phen_id'], 
            'original_phenotype': chosen_solution['phenotype'], 
            'original_genotype': chosen_solution['genotype'], 
            'mutation_target': least_represented_mutation_target, 
            'mutation_log': f"Mutated non-terminal {least_represented_mutation_target} in solution {chosen_solution['phen_id']} from {info['old_value']} to {info['new_value']} at position {info['position_mutated']}", 
            'original_fitness': o_fitness, 
            'original_other_info': original_other_info, 
            'phenotype': mutated_indiv['phenotype'], 
            'genotype': mutated_indiv['genotype'], 
            'fitness': mutated_fitness, 
            'other_info': mutated_other_info
        })

        mutation_counts[least_represented_mutation_target] += 1
        counter += 1
    mutation_registry_df = pd.concat([mutation_registry_df, pd.DataFrame(mutation_registry_rows)], ignore_index=True)
    mutation_registry_df.to_csv(os.path.join(params['DUMPS_DIR'], params['EXPERIMENT_NAME'], 'mutation_registry_df.csv'), index=False)
    import pickle
    with open(os.path.join(params['DUMPS_DIR'], params['EXPERIMENT_NAME'], 'mutation_archive.pkl'), 'wb') as f:
        pickle.dump(archive, f)
    print(f"Saved mutation registry and archive at iteration {counter}")
    print(f"Archive/evaluations: {archive_accesses}/{counter-archive_accesses}")
    mutation_counts = mutation_registry_df['mutation_target'].value_counts()
    print(f"Current sample size for each mutation target: { {k: v for k, v in mutation_counts.items()} }")
    return mutation_registry_df, archive

def plot_fitness_by_mutation_target(mutation_registry_df, grammar, extra_stuff_for_title="", file_name_suffix=""):
    # I want to make a box plot.
    # In the x axis, I want the mutation target, which is the non-terminal that was mutated.
    # In the y axis, I want the fitness in 2d space.
    # Additionally, i want all data points to be plotted as well, so I can see the distribution of the data and not just the summary statistics of the box plot.
    # The fitness in 2d space of the original solution should be included as a reference line in the box plot.
    plt.figure(figsize=(12, 6))
    mutation_registry_df['fitness_difference'] = mutation_registry_df['original_fitness'] - mutation_registry_df['fitness'] 

    # Get mutation targets sorted as they are in the grammar
    mutation_targets_sorted = [_ for  _ in grammar.get_non_terminals()]

    # Filter out mutation_targets that never that did not change the smart phenotype, as those are not informative for this analysis
    df2_filtered = mutation_registry_df 
    #sns.boxplot(x='mutation_target', y='fitness_difference', data=df2_filtered, order=mutation_targets_sorted,width=1.0, gap=0.5)
    sns.stripplot(x='mutation_target', y='fitness_difference', data=df2_filtered, order=mutation_targets_sorted, alpha=0.5, jitter=True)
    plt.axhline(y=0.0, color='r', linestyle='--', label='Original Loss in 2D')
    plt.legend()
    plt.title('Loss in 2D Space by Mutation Target' + extra_stuff_for_title)
    plt.xlabel('Mutation Target (Non-terminal Mutated)')
    plt.ylabel('Loss in 2D Space')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(params['DUMPS_DIR'], params['EXPERIMENT_NAME'], f'_loss_by_mutation_target{file_name_suffix}.png'))
    #plt.show()


def plot_effectiveness_by_mutation_target(mutation_registry_df, grammar, extra_stuff_for_title="", file_name_suffix=""):
    # I want to make a plot that shows the percentage of mutations were effective, divergent and redundant for each mutation target.
    # I want these to appear as stacked bars showing the percentage of effective, divergent and redundant solutions for each mutation target.
    # I want to use the following colors:
    # Effective: Green
    # Divergent: Red
    # Redundant: Gray
    plt.figure(figsize=(18, 6))
    # Get mutation targets sorted as they are in the grammar
    mutation_targets_sorted = [_ for  _ in grammar.get_non_terminals()]

    # Create a new column in df2 that indicates whether the fitness in 2d space is equal to float(infinity)
    df2 = mutation_registry_df.copy()
    df2['effective'] = (df2['fitness'] != df2['original_fitness']) & (df2['fitness'] > 0.2)
    df2['redundant'] = df2['fitness'] == df2['original_fitness']
    df2['divergent'] = (df2['fitness'] < 0.2) & (df2['redundant'] == False)

    # Group by mutation target and calculate the percentage of effective, divergent and redundant solutions
    effective_percentages = df2.groupby('mutation_target')['effective'].mean()
    divergent_percentages = df2.groupby('mutation_target')['divergent'].mean()
    redundant_percentages = df2.groupby('mutation_target')['redundant'].mean()
    # Create a stacked bar plot of the effective, divergent and redundant percentages
    bar_width = 0.5
    mutation_targets = effective_percentages.index

    #Sort effective_percentages, divergent_percentages and redundant_percentages by the order of mutation_targets_sorted
    effective_percentages = effective_percentages.reindex(mutation_targets_sorted)
    divergent_percentages = divergent_percentages.reindex(mutation_targets_sorted)
    redundant_percentages = redundant_percentages.reindex(mutation_targets_sorted)
    plt.bar(mutation_targets_sorted, effective_percentages, color='green', label='Effective', width=bar_width)
    plt.bar(mutation_targets_sorted, divergent_percentages, bottom=effective_percentages, color='red', label='Divergent', width=bar_width)
    plt.bar(mutation_targets_sorted, redundant_percentages, bottom=effective_percentages+divergent_percentages, color='gray', label='Redundant', width=bar_width)
    plt.title('Percentage of Effective, Divergent and Redundant Solutions by Mutation Target' + extra_stuff_for_title)
    plt.xlabel('Mutation Target (Non-terminal Mutated)')
    plt.ylabel('Percentage of Solutions')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Save the plot as a png file
    plt.savefig(os.path.join(params['DUMPS_DIR'], params['EXPERIMENT_NAME'], f'_effectiveness_by_mutation_target{file_name_suffix}.png'))
    #plt.show()

    mutation_probabilities = {
        '<start>': 0.0, #
        '<alpha_expr>': 0.01, #
        '<alpha_func>': 0.01, #
        '<alpha_terminal>': 0.01, #
        '<alpha_var_const>': 0.05, #
        '<alpha_const>': 0.15, #
        '<beta_expr>': 0.01, #
        '<beta_func>': 0.01, #
        '<beta_terminal>': 0.01, #
        '<beta_var_const>': 0.05, #
        '<beta_const>': 0.15, #
        '<sigma_expr>': 0.01, #
        '<sigma_func>': 0.01, #
        '<sigma_terminal>': 0.01, #
        '<sigma_var_const>': 0.05, #
        '<sigma_const>': 0.15, #
        '<grad_expr>': 0.01, #
        '<grad_func>': 0.01, #
        '<grad_terminal>': 0.05, #
        '<grad_const>': 0.15, #
        }
    # Present this information as a table as well
    summary_table = pd.DataFrame({
        'Mutation Target': mutation_targets_sorted,
        'Effective %': effective_percentages.values,
        'Divergent %': divergent_percentages.values,
        #'MutProb': [mutation_probabilities[nt] for nt in mutation_targets_sorted],
        'Redundant %': redundant_percentages.values,
    })
    print(summary_table)

# Make a new dataframe containing only the unique rows of mutation_registry_df based on the mutation_log column, keeping the first occurrence.
#unique_mutation_registry_df = mutation_registry_df.drop_duplicates(subset=['mutation_log'], keep='first')
#plot_effectiveness_by_mutation_target(unique_mutation_registry_df, grammar)


df = df[df['setup'] == 'adaptiveTest']
grammar._reset_grammar()
    
grammar.set_path(params['GRAMMAR'])
grammar.read_grammar()
grammar.set_max_tree_depth(1000000)
grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])

i = 0

#df['fitness'] = -df['fitness']
bad_df = df[df['fitness'] < 0.5]
good_df = df[df['fitness'] >= 0.5]
# only top 100 from each
bad_df = bad_df.sort_values(by='fitness', ascending=False).head(100)
good_df = good_df.sort_values(by='fitness', ascending=False).head(100)
mutation_registry_df, archive = mass_mutate_from_dataframe(mutation_registry_df, grammar, archive, bad_df, counter_limit=100000)
mutation_registry_df, archive = mass_mutate_from_dataframe(mutation_registry_df, grammar, archive, good_df, counter_limit=100000)

thresholds = [0.0, 0.5, 1.0]
for i in range(len(thresholds)-1):
    bottom_threshold = thresholds[i]
    top_threshold = thresholds[i+1]
    working_df = mutation_registry_df[(mutation_registry_df['original_fitness'] >= bottom_threshold) & (mutation_registry_df['original_fitness'] < top_threshold)]
    extra_stuff_for_title = f" (Only original solutions with fitness >= {bottom_threshold} and < {top_threshold})"
    
    print(f"Plotting for original solutions with fitness >= {bottom_threshold} and < {top_threshold}, samplesize: {len(working_df)}")
    plot_fitness_by_mutation_target(working_df, grammar, extra_stuff_for_title, file_name_suffix=f"_fitness_by_mutation_target_{i}")
    plot_effectiveness_by_mutation_target(working_df, grammar, extra_stuff_for_title, file_name_suffix=f"_effectiveness_by_mutation_target_{i}")

