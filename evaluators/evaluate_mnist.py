from dataset_loaders.mnist import MNIST_Dataset
from evaluators.evaluator_utils import Evaluator

class MNIST_Evaluator(Evaluator):
    def __init__(self, params, configuration_file=None, task_name=None):
        super().__init__(params, configuration_file=configuration_file, task_name=task_name)

    def _init_dataset_(self, validation_size, fitness_size, run, normalize, subtract_mean):
        self.dataset = MNIST_Dataset(validation_size=validation_size, fitness_size=fitness_size, seed=run, normalize=normalize, subtract_mean=subtract_mean)
        self.dataset.load_data_for_evolution()
    