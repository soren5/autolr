import datetime
from evaluators.dynamic_optimizer_evaluator import train_model
import contextlib
import multiprocessing
from multiprocessing import Pool
class LRScheduler:
    def __init__(self):
        pass
    def evaluate(self, phen):
        multiprocessing.set_start_method('spawn', True)
        num_pool_workers=1 
        value, other_info = None, None
        with contextlib.closing(Pool(num_pool_workers)) as po: 
            foo = po.map(train_model, [phen])
            value = foo[0][0]
            other_info = foo[0][1]
        
        #value, other_info = train_model(phen)
        return -value, other_info

def test_lr_scheduler():
    import core.grammar as grammar
    import core.sge
    class Test_LRScheduler:
        def __init__(self):
            pass
        def evaluate(self, phen):
            return (1, '')
    experience_name = "test_LR" + str(datetime.datetime.now()) + "/"
    grammar = grammar.Grammar("grammars/grammar_proposal.txt", 6, 17)
    evaluation_function = Test_LRScheduler()
    core.sge.evolutionary_algorithm(grammar = grammar, eval_func=evaluation_function, exp_name=experience_name)
    return True
   
if __name__ == "__main__":
    import sge.grammar as grammar
    import sge
    evaluation_function = LRScheduler()
    sge.evolutionary_algorithm(evaluation_function=evaluation_function)
        

