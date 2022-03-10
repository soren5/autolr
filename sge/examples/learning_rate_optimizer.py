import datetime
from .model_evaluator_adaptive import train_model
import contextlib
import multiprocessing
from multiprocessing import Pool
class LROptimizer:
    def __init__(self):
        pass
    def evaluate(self, phen):
        #multiprocessing.set_start_method('spawn', True)
        #num_pool_workers=1 
        value, other_info = None, None
        # with contextlib.closing(Pool(num_pool_workers)) as po: 
        #     from .model_evaluator_adaptive import train_model
        #     foo = po.map(train_model, [[phen, weights]])
        #     value = foo[0][0]
        #     other_info = foo[0][1]
        value, other_info = train_model(phen)
        
        #value, other_info = train_model(phen)
        return -value, other_info

class Test_LROptimizer:
    def __init__(self):
        pass
    def evaluate(self, phen):
        return (1, '')


def test_lr_optimizer():
    import sge.grammar as grammar
    import sge
    experience_name = "test_LR" + str(datetime.datetime.now()) + "/"
    evaluation_function = Test_LROptimizer()
    sge.evolutionary_algorithm(evaluation_function=evaluation_function)
    return True


   
if __name__ == "__main__":
    # print("Starting sanity check...")
    # test_lr_optimizer()
    # print("Finished sanity check.")
    from numpy.random import seed
    import sge.grammar as grammar
    import sge
    #remote = input("Remote? (reply with y)")
    remote = 'y'
    if remote == 'y':
        from sge.utilities.email_script import send_email
    evaluation_function = LROptimizer()
    sge.evolutionary_algorithm(evaluation_function=evaluation_function)
        

