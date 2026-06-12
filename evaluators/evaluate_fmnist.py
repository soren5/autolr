from dataset_loaders.fmnist import FMNIST_Dataset
from evaluators.evaluator_utils import Evaluator

class FMNIST_Evaluator(Evaluator):
    def __init__(self, params, configuration_file='FMNIST_CONFIG', task_name='fmnist', benchmark_data=False):
        super().__init__(params, configuration_file=configuration_file, task_name=task_name, benchmark_data=benchmark_data)

    def _init_dataset_(self, validation_size, fitness_size, run, normalize, subtract_mean, benchmark_data=False, test_size=None):
        self.dataset = FMNIST_Dataset(validation_size=validation_size, fitness_size=fitness_size, seed=run, normalize=normalize, subtract_mean=subtract_mean)
        if benchmark_data:
            self.dataset.test_size = test_size
            self.dataset.load_data_for_benchmark()
        else:
            self.dataset.load_data_for_evolution()
