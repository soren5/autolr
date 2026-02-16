from dataset_loaders.fmnist import FMNIST_Dataset
from evaluators.evaluator_utils import Evaluator

class FMNIST_Evaluator(Evaluator):
    def __init__(self, params, configuration_file='FMNIST_CONFIG', task_name='fmnist'):
        super().__init__(params, configuration_file=configuration_file, task_name=task_name)

    def _init_dataset_(self, validation_size, fitness_size, run, normalize, subtract_mean):
        self.dataset = FMNIST_Dataset(validation_size=validation_size, fitness_size=fitness_size, seed=run, normalize=normalize, subtract_mean=subtract_mean)
        self.dataset.load_data_for_evolution()