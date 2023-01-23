#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --job-name=bt_sending_loop
#SBATCH --output=bt_sending_loop%j.log


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
declare -a folders=("many_runs_no_crossover")
declare -a tasks=("cif_from_rapid_fmni_big")
declare data_path=./many_results  

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
    
    #find if resume paramerter is needed
    resume="0"
    generations="200"
    parent_experiment="None"

    #based on task adjust based if it is rapid or normal resume time and generations
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
        for seed in $(seq 1 30)
        do(
         
          #based on files present in folder decide if resume needs to be 'Last' (need to resume run that did not complete)
          files=$(shopt -s nullglob dotglob; echo $experiment_name/run_$seed/*) # check if folder has files
          if (( ${#files} ))
          then
            #check if expected last file exists
            expected_last_file=$experiment_name/run_$seed/"population_"$generations.json
            if ( [ -e "$expected_last_file" ])
            then
              echo "contains files for last expected iteration= $generations, no need to re-run"
              continue
            else
              echo "does not contain file for last expected iteration"
              echo "resume becomes: Last"
              resume="Last"
            fi
          else 
            echo "does not contain files, leave resume as defined previously, resume=$resume"
          fi
          
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
          sbatch run_sims_given_param_from_loop_new.sh "${todos[@]}" 
          )
        done
      ) 
      done
    )
  done
  )
done