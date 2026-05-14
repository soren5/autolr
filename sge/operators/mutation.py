import copy
import random
from sge.grammar import grammar


def mutate(p, pmutation):
    p = copy.deepcopy(p)
    p['fitness'] = None
    size_of_genes = grammar.count_number_of_options_in_production()
    mutable_genes = [index for index, nt in enumerate(grammar.get_non_terminals()) if size_of_genes[nt] != 1 and len(p['genotype'][index]) > 0]
    for at_gene in mutable_genes:
        nt = list(grammar.get_non_terminals())[at_gene]
        temp = p['mapping_values']
        mapped = temp[at_gene]
        for position_to_mutate in range(0, mapped):
            if random.random() < pmutation:
                current_value, current_depth = p['genotype'][at_gene][position_to_mutate]
                choices = []
                if current_depth >= grammar.get_max_depth():
                    choices = grammar.get_non_recursive_options()[nt]
                else:
                    choices = list(range(0, size_of_genes[nt]))
                    choices.remove(current_value)
                if len(choices) == 0:
                    choices = range(0, size_of_genes[nt])
                p['genotype'][at_gene][position_to_mutate] = [random.choice(choices), current_depth]
                p['operation'] = "crossover+mutation" if p['operation'] == "crossover" else "mutation"
    return p

def mutate_one(p, chosen_nt=None):
    p = copy.deepcopy(p)
    p['fitness'] = None
    # get a dictionary with keys and non-terminals and values as the number of options in the production rule for that non-terminal
    size_of_genes = grammar.count_number_of_options_in_production()

    # get the mutable non-terminals, which are the ones that have more than one option in the production rule and have a non-empty genotype
    mutable_genes = [index for index, nt in enumerate(grammar.get_non_terminals()) if size_of_genes[nt] != 1 and len(p['genotype'][index]) > 0 and p['mapping_values'][index] > 0]

    if chosen_nt is None:
        # Pick one random non-terminal to mutate
        at_gene = random.choice(mutable_genes)
        nt = list(grammar.get_non_terminals())[at_gene]
    else:
        # If a specific non-terminal is provided, find its index in the list of non-terminals
        at_gene = list(grammar.get_non_terminals()).index(chosen_nt)
        if at_gene not in mutable_genes:
            raise ValueError(f"Chosen non-terminal {chosen_nt} is not mutable.\n Available mutable non-terminals are: {[list(grammar.get_non_terminals())[index] for index in mutable_genes]}")
        nt = list(grammar.get_non_terminals())[at_gene]

    temp = p['mapping_values']
    mapped = temp[at_gene]

    # Pick one random position to mutate
    try:
        position_to_mutate = random.randint(0, mapped - 1)
    except ValueError:
        print(f"Error: mapped value is {mapped} for gene {nt} at index {at_gene}. Genotype length is {len(p['genotype'][at_gene])}.")
        position_to_mutate = 0

    current_value, current_depth = p['genotype'][at_gene][position_to_mutate]
    choices = []
    if current_depth >= grammar.get_max_depth():
        choices = grammar.get_non_recursive_options()[nt]
    else:
        choices = list(range(0, size_of_genes[nt]))
        choices.remove(current_value)
    if len(choices) == 0:
        choices = range(0, size_of_genes[nt])
    #print(f"Mutating gene {nt} at position {position_to_mutate} from value {current_value} to one of {choices}")
    p['genotype'][at_gene][position_to_mutate] = [random.choice(choices), current_depth]
    p['operation'] = "crossover+mutation" if p['operation'] == "crossover" else "mutation"
    
    return p, {'non-terminal': nt, 'position_mutated': position_to_mutate, 'old_value': current_value, 'new_value': p['genotype'][at_gene][position_to_mutate]} 

def mutate_level(p, pmutation):
    p = copy.deepcopy(p)
    p['fitness'] = None
    size_of_genes = grammar.count_number_of_options_in_production()
    mutable_genes = [index for index, nt in enumerate(grammar.get_non_terminals()) if size_of_genes[nt] != 1 and len(p['genotype'][index]) > 0]
    for at_gene in mutable_genes:
        nt = list(grammar.get_non_terminals())[at_gene]
        temp = p['mapping_values']
        mapped = temp[at_gene]
        for position_to_mutate in range(0, mapped):
            if random.random() < pmutation[at_gene]:
                # print("I am mutating, master!")
                current_value, current_depth = p['genotype'][at_gene][position_to_mutate]
                choices = []
                if current_depth >= grammar.get_max_depth():
                    choices = grammar.get_non_recursive_options()[nt]
                else:
                    choices = list(range(0, size_of_genes[nt]))
                    choices.remove(current_value)
                if len(choices) == 0:
                    choices = range(0, size_of_genes[nt])
                p['genotype'][at_gene][position_to_mutate] = [random.choice(choices), current_depth]
                p['operation'] = "crossover+mutation" if p['operation'] == "crossover" else "mutation"
    return p
