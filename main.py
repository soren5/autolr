class Optimizer_Evaluator_Tensorflow:
    def __init__(self, train_model=None):   
        if train_model == None: 
            from evaluators.adaptive_optimizer_evaluator_f_race import train_model_tensorflow_fmnist
        self.train_model = train_model_tensorflow_fmnist
    
    def evaluate(self, phen, params):
        import contextlib
        import multiprocessing
        from multiprocessing import Pool    
        multiprocessing.set_start_method('spawn', True)
        num_pool_workers=1 
        value, other_info = None, None
        with contextlib.closing(Pool(num_pool_workers)) as po: 
            phen_params = (phen, params)
            foo = po.map(self.train_model, [phen_params])
            value = foo[0][0]
            other_info = foo[0][1]
        return -value, other_info

class Optimizer_Evaluator_Torch:
    def __init__(self, train_model=None):   
        if train_model == None: 
            from evaluators.adaptive_optimizer_evaluator_f_race_torch import train_model_torch
        self.train_model = train_model_torch

    def evaluate(self, phen, params):
        value, other_info = self.train_model([phen, params])
        return -value, other_info

if __name__ == "__main__":
    import sge
    evaluation_function = Optimizer_Evaluator_Tensorflow()
    
    sge.evolutionary_algorithm(evaluation_function=evaluation_function)
        

