from sge.parameters import (
    params,
    set_parameters
)


class Optimizer_Evaluator_Tensorflow:
    def __init__(self, train_model=None):  #should give a function 
        # Note: only works with fmnist
        from evaluators.adaptive_optimizer_evaluator_f_race import train_model_tensorflow_fmnist as train_model
        self.train_model = train_model
    
    def evaluate(self, phen, params):
        foo = self.train_model([phen, params])
        return -foo[0], foo[1]


if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    import sge
    import sge.logger as logger
    import sys

    if len(sys.argv) != 1:
        raise Exception("profile.py doesn't support command line arguments")
    sys.argv = ["", "--parameters", "parameters/adaptive_autolr_mutate_level_1.yml"]
    set_parameters(sys.argv[1:])

    from evaluators.adaptive_optimizer_evaluator_f_race import train_model_tensorflow_fmnist_cached
    evaluation_function = Optimizer_Evaluator_Tensorflow(train_model_tensorflow_fmnist_cached)
    sge.evolutionary_algorithm(evaluation_function=evaluation_function)
        

