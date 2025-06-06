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

def prune_population(pop):
    for indiv in pop:
        if 'other_info' in indiv:
            indiv.pop('other_info')
    return pop

def _unidiff_output(expected, actual):
    """
    Helper function. Returns a string containing the unified diff of two multiline strings.
    """

    import difflib
    expected=expected.splitlines(1)
    actual=actual.splitlines(1)

    diff=difflib.unified_diff(expected, actual)

    return ''.join(diff)
