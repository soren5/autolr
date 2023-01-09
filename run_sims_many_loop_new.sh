#!/bin/bash
# SBATCH --partition=short
# SBATCH --time=00:10:00
# SBATCH --job-name=bt_sending_loop
# SBATCH --output=bt_sending_loop%j.log


###usage:
### set the data_path variable to the folder where the data should be saved
### set the folders variable as the array of folders from where the parameters files should be read
### set the tasks variables WITH THE CORRECT NOMENCLATURE as the array of simulations you want to run
### TASK NOMENCLATURE:
### (separate each word with an dash '_')
### First word: 'cif, mnist, fmni' -> name of the task on whihc to train the network 
### (optional -> for transference experiments)
### Second word : 'from' -> bash script will understand it is a transference experiment
### Third word : 'rapid' -> bash script will understand that it is a rapid transference experiment and adjust starting and finishing generation
### Fourth word : 'cif, mnist, fmni' -> name of task from which the population is seeded

export PATH=$HOME/.local/bin:$PATH
module load Python
module load CUDA
module load cuDNN
pip install --upgrade pip
pip install -r requirements.txt
python -m utils.create_models
declare -a folders=("many_runs_no_crossover")
declare -a tasks=("mnist")
declare data_path=../data/p288427   

for folder in "${folders[@]}"
do(

  path=./parameters/$folder
  echo "path: $path"

  for task in "${tasks[@]}"
  do(

    experiment_name=$data_path/$folder/$task

    #based on task find correct dataset and task parameter
    dataset=""
    model=""
    validation_set=""
    fitness_set=""

    if [[ "$task" == "fmni"* ]]; then
      dataset="fmnist"
      model="models/mnist_model.h5"
      validation_set="3500"
      fitness_set="50000"
      echo "dataset and model is fmni"
    fi

    if [[ "$task" == "mnist"* ]]; then
      dataset="mnist"
      model="models/mnist_model.h5"
      validation_set="3500"
      fitness_set="50000"
      echo "dataset and model is mnist"

    fi

    if [[ "$task" == "cif"* ]]; then
      dataset="cif"
      model="models/cifar_model.h5"
      validation_set="3000"
      fitness_set="3000"
      echo "dataset and model is cif"
    fi
    
    #based ont ask find if resume paramerter is needed
    # and adjust based if it is rapid or normal resume time and generations
    resume="0"
    generations="200"
    parent_experiment="None"

    if [[ "$task" == *"from"* ]]; then
      resume="100"
      if [[ "$task" == *"rapid"* ]]; then
        resume="30"
        generations="60"
      fi

      #find based on name the name of parent experiment
      if [[ "${task: -6}" == *"cif" ]]; then
        parent_experiment=$data_path/$folder/"cif"
      elif [[ "${task: -6}" == *"fmni" ]]; then
        parent_experiment=$data_path/$folder/"fmni"
      elif [[ "${task: -6}" == *"mnist" ]]; then
        parent_experiment=$data_path/$folder/"mnist"
      else
        parent_experiment="None"
      fi
    fi

    for filename in "$path"/*.yml ; 
    do(
        for seed in $(seq 1 1)
        do(
          echo "sending: $filename"
          echo "seed: $seed"
          echo "experiment name: $experiment_name"
          echo "parent experiment: $parent_experiment" 
          echo "model: $model"
          echo "dataset $dataset" 
          echo "resume $resume"
          echo "generations $generations"
          echo "validation size $validation_set"
          echo "fitness size $fitness_set"
          [ -e "$filename" ] || continue 
          declare -a todos=("--parameters" "$filename" "--run" "$seed" "--seed" "$seed" "--parent_experiment" "$parent_experiment" "--experiment_name" "$experiment_name" "--model" "$model" "--dataset" "$dataset" "--resume" "$resume" "--generations" "$generations" "--validation_size" "$validation_set" "--test_size" "$fitness_set")
          echo "todos: ${todos[@]}"  
          bash run_sims_given_param_from_loop_new.sh "--parameters" "$filename" "--run" "$seed" "--seed" "$seed" "--parent_experiment" "$parent_experiment" "--experiment_name" "$experiment_name" "--model" "$model" "--dataset" "$dataset" "--resume" "$resume" "--generations" "$generations" "--validation_size" "$validation_set" "--test_size" "$fitness_set"
          )
        done
      ) 
      done
    )
  done
  )
done