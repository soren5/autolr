import json
import glob
import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import pandas as pd

# extract number from file
def extract_number(file_path):
    # Define the regular expression pattern to match "iteration_X" where X is a number
    pattern = r'iteration_(\d+)'

    # Use re.search() to find the pattern in the file path
    match = re.search(pattern, file_path)

    if match:
        # Extract the numeric part (X) from the matched pattern
        iteration_number = int(match.group(1))
        return iteration_number
    else:
        # If the pattern is not found, return None or raise an exception
        return None


def get_all_json_files(root_dir):
    # Step 2: Combine Data in Chunks
    json_files = []
    experiment_names = []
    run_numbers = []

    for experiment_name in os.listdir(root_dir):
        experiment_dir = os.path.join(root_dir, experiment_name)
        if not os.path.isdir(experiment_dir):
            continue
        for run_number in os.listdir(experiment_dir):
            run_dir = os.path.join(experiment_dir, run_number)
            if not os.path.isdir(run_dir):
                continue
            for file_name in os.listdir(run_dir):
                if file_name.endswith('.json'):
                    file_path = os.path.join(run_dir, file_name)
                    json_files.append(file_path)
                    experiment_names.append(experiment_name)
                    run_numbers.append(run_number)  # Convert run_number to int if it's an integer

    return json_files, experiment_names, run_numbers


def load_results(root_dir):
    # Step 1: Data Preprocessing
    def preprocess_population_data_from_json(json_file_path, experiment_name, run_number):
        processed_data = []
        try:
            with open(json_file_path, 'r') as file:
                content = file.read()
            
            modified_content = content.replace('NaN', '0.0')
            with open(json_file_path, 'w') as file:
                file.write(modified_content)
            with open(json_file_path, 'r') as file:
                if "iteration" in json_file_path:
                    print(json_file_path, extract_number(json_file_path))
                    data = json.load(file)
                    for x in data:
                        if type(x) == dict:
                            indiv_data = {
                                'Experiment name': experiment_name,
                                'Run number': run_number,
                                'Individual number': x['id'],
                                'Generation': extract_number(json_file_path),
                                'Phenotype': x['phenotype'],
                                'Smart Phenotype': x['smart_phenotype'],
                                'Fitness': x['fitness']
                            }
                            processed_data.append(indiv_data)
        except Exception as e:
            # code to handle the exception
            print(f"An exception of type {type(e).__name__} occurred: {e}")
            print("Error in reading results ", json_file_path)
        return processed_data

    json_files, experiment_names, run_numbers = get_all_json_files(root_dir)
    chunk_size = 200

    chunks = [json_files[i:i + chunk_size] for i in range(0, len(json_files), chunk_size)]

    combined_data = []

    for i, chunk in enumerate(chunks):
        for j, file_path in enumerate(chunk):
            experiment_name = experiment_names[i * chunk_size + j]
            run_number = run_numbers[i * chunk_size + j]
            processed_data = preprocess_population_data_from_json(file_path, experiment_name, run_number)
            combined_data.extend(processed_data)
    df = pd.DataFrame(combined_data)
    return df
    # Step 2: Combine Data in Chunks
"""     json_files, experiment_names, run_numbers = get_all_json_files(root_dir)
    chunk_size = 200

    chunks = [json_files[i:i + chunk_size] for i in range(0, len(json_files), chunk_size)]

    combined_data = []

    for chunk in chunks:
        for file_path in chunk:
            processed_data = preprocess_data_from_json(file_path)
            combined_data.extend(processed_data) """

    # Step 3: Create DataFrame


