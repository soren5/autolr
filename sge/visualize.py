import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
def prot_div(left, right):
    if right == 0:
        return 0
    else:
        return left / right

def if_func(condition, state1, state2):
    if condition:
        return state1
    else:
        return state2

def read_experiment_results(experiment_name, iterations, epochs=100):
    dir_name = '/Users/soren/Work/Research/dsge_learning_rate/results/' + experiment_name
    results = []
    for it in range(iterations):
        with open(dir_name + 'iteration_' + str(it) + '.json') as json_file:
            if it % 10 == 0:
                print(it)
            data = json.load(json_file)
            for i in data:
                #print(i['phenotype'])
                past_lr = 0.01
                i['values'] = []
                for epoch in range(epochs):
                    learning_rate = past_lr
                    lr = eval(i['phenotype'])
                    past_lr = lr
                    i['values'].append(lr)
            results.append(data)
    print("Finished reading ", experiment_name)
    return results
results = []
month = '1'
day = '17'
trial_number = '2'
#folder_name = 'dumps_' + month + '_' + day
folder_name = 'dumps/adaptiveTest'
#folder_name = 'trial' + trial_number
#run_number = ['2','3','4','5','6','7','8','9','11','12']
run_number = ['1','2','3','4','5','6','7','8','9','10']
iterations = 1500
for i in run_number: 
    results.append(read_experiment_results(folder_name + '/run_' + i + '/', iterations))
def plot_iter(iter_results, stop_epoch, indivs=None, color='blue', id_num=0):
    epochs = np.arange(len(iter_results[0]['values']))
    if indivs == None:
        for i in iter_results:
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.axvline(stop_epoch)
            plt.plot(epochs, i['values'], label='val_acc: ' + str(-i['fitness']), color=color)
            plt.savefig(str(id_num) + '.pdf')
            plt.show()
            foo = 0
            for x in i['other_info']['val_accuracy']:
                print(x, foo)
                foo += 1
    plt.show()
        
        

def plot_fit(results):
    epochs = np.arange(len(results[0]))
    #print(epochs)
    averages_all = []
    bests_all = []
    stds_all = []
    best_of_all = []
    stds_best_all = [] 
    boa_fit = 0
    boa_indiv = None
    for iteration in epochs:
        averages_all.append([])
        bests_all.append([])
        stds_all.append([])
        best_of_all.append(0)
        stds_best_all.append(0)        
        for result in results:
            all_fits = []
            best = 0
            gen_best = 0
            for indiv in result[iteration]:
                if indiv['fitness'] < best:
                    best = indiv['fitness']
                    #plot_iter([indiv])
                    #print(indiv['other_info'].keys())
                    if indiv['fitness'] < boa_fit:
                        boa_fit = best
                        boa_indiv = indiv
                        print('NEW BEST\n', indiv['phenotype'], iteration)
                        plot_iter([boa_indiv], len(indiv['other_info']['loss']), color='red', id_num=iteration)
                best = best if indiv['fitness'] > best else indiv['fitness']
                all_fits.append(indiv['fitness'])
            averages_all[iteration].append(np.average(all_fits))
            stds_all[iteration].append(np.std(all_fits))
            bests_all[iteration].append(best)
        stds_best_all[iteration] = np.std(bests_all[iteration]) * -1
        stds_all[iteration] = np.std(averages_all[iteration])
        averages_all[iteration] = np.average(averages_all[iteration]) * -1
        best_of_all[iteration] = np.min(bests_all[iteration])  * -1
        bests_all[iteration] = np.average(bests_all[iteration]) * -1
    #plt.plot(epochs, averages_all, label='population average')
    #plt.fill_between(epochs, [i + j for i, j in zip(averages_all, stds_all)], [i - j for i, j in zip(averages_all, stds_all)], alpha=0.2)
    plt.plot(epochs, bests_all, label='best average')   
    plt.fill_between(epochs, [i + j for i, j in zip(bests_all, stds_best_all)], [i - j for i, j in zip(bests_all, stds_best_all)], alpha=0.2)
    #plt.plot(epochs, [0.7862666646639506 for i in bests_all], label='fixed lr val_acc')
    #plt.plot(epochs, best_of_all, label='best of all')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('best_average_evolution.pdf')
    plt.show()

plot_fit(results)