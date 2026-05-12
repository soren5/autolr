import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from sge.parameters import (
    params,
    set_parameters
)
from fitness_functions.fitness_functions import *

if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   
    import sge
    import sys

    set_parameters(sys.argv[1:])   

    models_dir = params.get('MODELS_DIR', 'models')
    cifar_model_path = os.path.join(models_dir, 'cifar_model.h5')
    mnist_model_path = os.path.join(models_dir, 'mnist_model.h5')

    if False:
        evaluation_function = Optimizer_Evaluator_Torch()
        if 'MODEL' in params and params['MODEL'] == cifar_model_path: 
            from evaluators.adaptive_optimizer_evaluator_f_race import train_model_tensorflow_cifar10
            evaluation_function = Optimizer_Evaluator_Tensorflow(train_model=train_model_tensorflow_cifar10)
        elif 'MODEL' in params and params['MODEL'] == mnist_model_path and params['DATASET'] == 'fmnist':    
            from evaluators.adaptive_optimizer_evaluator_f_race import train_model_tensorflow_fmnist 
            evaluation_function = Optimizer_Evaluator_Tensorflow(train_model_tensorflow_fmnist)
        elif 'MODEL' in params and params['MODEL'] == mnist_model_path and params['DATASET'] == 'mnist':    
            from evaluators.adaptive_optimizer_evaluator_f_race import train_model_tensorflow_mnist 
            evaluation_function = Optimizer_Evaluator_Tensorflow(train_model_tensorflow_mnist)
    from sge.parameters import params
    if params['MULTI_TASK']:
        evaluator = Optimizer_Evaluator_FMNIST_CIFAR10_CIFAR100_TIN(params)
    else:
        evaluator = Optimizer_Evaluator_Tensorflow(params)
        # Infer evaluator from model path
        if 'cifar' in params['MODEL']:
            from evaluators.evaluate_cifar10 import CIFAR10_Evaluator
            evaluator = Optimizer_Evaluator_Tensorflow(params, evaluator=CIFAR10_Evaluator)
        elif 'resnet' in params['MODEL']:
            from evaluators.evaluate_tiny_imagenet import TINY_IMAGENET_Evaluator
            evaluator = Optimizer_Evaluator_Tensorflow(params, evaluator=TINY_IMAGENET_Evaluator)
        elif 'mnist' in params['MODEL']:
            from evaluators.evaluate_fmnist import FMNIST_Evaluator
            evaluator = Optimizer_Evaluator_Tensorflow(params, evaluator=FMNIST_Evaluator)
        else:
            # If nothing else, default to mnist
            from evaluators.evaluate_mnist import MNIST_Evaluator
            evaluator = Optimizer_Evaluator_Tensorflow(params, evaluator=MNIST_Evaluator)

        

    #sge.evolutionary_algorithm(evaluation_function=Optimizer_Evaluator_Dual_Task())
    sge.evolutionary_algorithm(evaluation_function=evaluator)  

