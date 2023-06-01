from sge.parameters import (
    params,
    set_parameters
)
class Optimizer_Evaluator_Tensorflow:
    def __init__(self, train_model=None):  #should give a function 
        if train_model == None: 
            from evaluators.adaptive_optimizer_evaluator_f_race import train_model_tensorflow_fmnist as train_model
        self.train_model = train_model
    
    def evaluate(self, phen, params):
        foo = self.train_model([phen, params])
        return -foo[0], foo[1]

class Optimizer_Evaluator_Torch:
    def __init__(self, train_model=None):   
        if train_model == None: 
            from evaluators.adaptive_optimizer_evaluator_f_race_torch import train_model_torch
        self.train_model = train_model_torch

    def evaluate(self, phen, params):
        value, other_info = self.train_model([phen, params])
        return -value, other_info


if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    import sge
    import sys

    set_parameters(sys.argv[1:])   

    if 'MODEL' in params and params['MODEL'] == 'models/cifar_model.h5': 
        from evaluators.adaptive_optimizer_evaluator_f_race import train_model_tensorflow_cifar10
        evaluation_function = Optimizer_Evaluator_Tensorflow(train_model=train_model_tensorflow_cifar10)
    elif 'MODEL' in params and params['MODEL'] == 'models/mnist_model.h5' and params['DATASET'] == 'fmnist':    
        from evaluators.adaptive_optimizer_evaluator_f_race import train_model_tensorflow_fmnist 
        evaluation_function = Optimizer_Evaluator_Tensorflow(train_model_tensorflow_fmnist)
    elif 'MODEL' in params and params['MODEL'] == 'models/mnist_model.h5' and params['DATASET'] == 'mnist':    
        from evaluators.adaptive_optimizer_evaluator_f_race import train_model_tensorflow_mnist 
        evaluation_function = Optimizer_Evaluator_Tensorflow(train_model_tensorflow_mnist)

    sge.evolutionary_algorithm(evaluation_function=evaluation_function)
        

