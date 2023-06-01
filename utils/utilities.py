import os

def delete_directory(experiment_name, runs):
    if type(runs) != list:
        runs = [runs]
    for run in runs:
        path = os.path.join(experiment_name, run)
        for file_name in os.listdir(path):
            # construct full file path
            file = os.path.join(path, file_name)
            if os.path.isfile(file):
                print('Deleting file:', file)
                os.remove(file)
        os.rmdir(path)
    os.rmdir(experiment_name)