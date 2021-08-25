from deap import creator, base, tools
from benchmarks.evaluate_cifar_model import evaluate_cifar_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import random
import os
import pandas as pd

creator.create("FitnessModelAccuracy", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessModelAccuracy)
cwd_path = os.getcwd()

IND_SIZE=4

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    optimizer = Adam(learning_rate=individual[0], beta_1=individual[1], beta_2=individual[2], epsilon=individual[3])

    result = evaluate_cifar_model(optimizer=optimizer, verbose=2, epochs= 10)



    data_frame = pd.read_csv(os.path.join(cwd_path, 'results/' , "development_results.csv"))
    if len(data_frame) > 1:
        total_epochs = data_frame.loc[len(data_frame) - 2, "epochs"]
    else:
        total_epochs = 0

    col_values = [total_epochs + 10, *individual]
    col_names = ["epochs", "learning_rate", "beta_1", "beta_2", "epsilon"] 
    data_frame.loc[len(data_frame) - 1, col_names] = col_values
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "development_results.csv"), index=False)

    return (result[0],)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def main():

    col_names = ["epochs", "max_val_accuracy", "min_val_loss", "test_accuracy", "learning_rate", "beta_1", "beta_2", "epsilon"] 
    data_frame = pd.DataFrame(columns=col_names)
    data_frame.to_csv(os.path.join(cwd_path, 'results/' , "development_results.csv"), index=False)

    pop = toolbox.population(n=50)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    return pop

main()