from evaluators.adaptive_optimizer_evaluator import train_model
import contextlib
import multiprocessing
from multiprocessing import Pool

class LROptimizer:
    def __init__(self):
        pass
    def evaluate(self, phen, params):
        multiprocessing.set_start_method('spawn', True)
        num_pool_workers=1 
        value, other_info = None, None
        with contextlib.closing(Pool(num_pool_workers)) as po: 
            foo = po.map(train_model, [phen])
            value = foo[0][0]
            other_info = foo[0][1]
        
        #value, other_info = train_model(phen)
        return -value, other_info

if __name__ == "__main__":
    import sge.grammar as grammar
    import sge
    evaluation_function = LROptimizer()
    sge.evolutionary_algorithm(evaluation_function=evaluation_function)
        

