from dataset_loaders.tiny_imagenet import TINY_IMAGENET_Dataset
from models.keras_model_adapters.resnet_adapter import ResNet_Interface
from evaluators.evaluator_utils import Evaluator

class TINY_IMAGENET_Evaluator(Evaluator):
    def __init__(self, params, configuration_file='TINY_IMAGENET_CONFIG', task_name='tiny_imagenet', benchmark_data=False):
        super().__init__(params, configuration_file=configuration_file, task_name=task_name, benchmark_data=benchmark_data)

    def _init_dataset_(self, validation_size, fitness_size, run, normalize, subtract_mean, benchmark_data=False, test_size=None):
        self.dataset = TINY_IMAGENET_Dataset(validation_size=validation_size, fitness_size=fitness_size, seed=run, normalize=normalize, subtract_mean=subtract_mean)
        if benchmark_data:
            self.dataset.test_size = test_size
            self.dataset.load_data_for_benchmark()
        else:
            self.dataset.load_data_for_evolution()

    def _init_model_(self, model_path):
        # For ImageNet we use an adapted Keras model instead of a local one
        # self.model = load_model(model_path, compile=False)
        self.model = ResNet_Interface(incoming_data_shape=(self.dataset.img_rows, self.dataset.img_cols, self.dataset.channels)).get_model()
        self.model_initial_weights = self.model.get_weights()
