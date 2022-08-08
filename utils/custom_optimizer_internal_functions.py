import functools
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import rmsprop
from sge.grammar import *

tf.compat.v1.disable_eager_execution()

def alpha_func_filler(shape, alpha, grad):
    return grad

def beta_func_filler(shape, alpha, beta, grad):
    return grad

def sigma_func_filler(shape, alpha, beta, sigma, grad):
    return grad

def grad_func_filler(shape, alpha, beta, sigma, grad):
    return tf.math.negative(shape, alpha)

def alpha_func_momentum(shape, alpha, grad):
    foo = tf.math.negative(
                    tf.math.subtract(
                                tf.math.multiply(
                                    tf.math.subtract(
                                            tf.constant(0.99, shape=shape, dtype=tf.float32), 
                                            tf.constant(1.0, shape=shape, dtype=tf.float32),
                                    alpha,
                                ),
                                tf.math.multiply(
                                        tf.constant(0.01, shape=shape, dtype=tf.float32),
                                        grad
                                )
                        )
                    )
                )
    return foo

def grad_func_momentum(shape, alpha, beta, sigma, grad):
    return tf.math.negative(shape, alpha)

def alpha_func_nesterov(shape, alpha, grad):
    foo = tf.math.negative(
                    tf.math.subtract(
                                tf.math.multiply(
                                    tf.math.subtract(
                                            tf.constant(0.99), 
                                            tf.constant(1.0)),
                                    alpha
                                ),
                                tf.math.multiply(
                                        tf.constant(0.01),
                                        grad
                                )
                        )
                    )
    return foo

def grad_func_nesterov(shape, alpha, beta, sigma, grad):
    return tf.math.negative(shape, alpha)

def alpha_func_adagrad(shape, alpha, grad):
    foo = tf.negative(tf.math.square(grad))
    return foo

def grad_func_adagrad(shape, alpha, beta, sigma, grad):
    return tf.math.multiply(
            grad,
            tf.math.divide_no_nan(
                tf.constant(0.001),
                tf.math.add(
                    tf.math.sqrt(shape, alpha),
                    tf.constant(1e-7)
                )
            )
        )

def alpha_func_rmsprop(shape, alpha, grad):
    foo = tf.math.add(
            tf.math.multiply(
                tf.math.subtract(
                    tf.constant(0.0),
                    tf.constant(0.1)),
                alpha
            ),
            tf.math.multiply(
                tf.constant(0.1),
                tf.math.square(grad)
            )
        )
    return tf.negative(foo)

def beta_func_rmsprop(shape, alpha, beta, grad):
    foo = tf.math.add(
        tf.math.multiply(
            tf.math.subtract(
                tf.constant(0.0),
                tf.constant(0.01)),
            beta
        ),
        tf.math.divide_no_nan(
            tf.math.multiply(
                tf.constant(0.001),
                grad
            ),
            tf.math.sqrt(
                tf.math.add(
                    alpha,
                    tf.constant(1e-7)
                )
            )
        )
    )
    return tf.negative(foo)

def grad_func_rmsprop(shape, alpha, beta, sigma, grad):
    return beta

def alpha_func_adam(shape, alpha, grad):
    foo = tf.math.add(
            tf.math.multiply(
                tf.math.negative(
                tf.constant(0.1)),
                alpha
            ),
            tf.math.multiply(
                tf.constant(0.1),
                grad
            )
        )
    return tf.negative(foo)

def beta_func_adam(shape, alpha, beta, grad):
    foo = tf.math.add(
            tf.math.multiply(
                tf.math.negative(
                    tf.constant(0.001)),
                beta 
            ),
            tf.math.multiply(
                tf.constant(0.001),
                tf.math.square(grad)
            )
        )
    return tf.negative(foo)

def sigma_func_adam(shape, alpha, beta, sigma, grad):
    foo = tf.constant(1.0)
    return tf.negative(foo)

def grad_func_adam(shape, alpha, beta, sigma, grad):
    foo = tf.math.divide(
        tf.math.multiply(
            tf.math.multiply(
                tf.constant(0.001),
                tf.math.divide(
                    tf.math.sqrt(
                        tf.math.subtract(
                            tf.constant(1.0),
                            tf.math.pow(
                                tf.constant(0.999),
                                sigma
                            )
                        )
                    ),
                    tf.math.subtract(
                        tf.constant(1.0),
                        tf.math.pow(
                            tf.constant(0.9),
                            sigma
                        )
                    )
                )
            ),
            alpha
        ),
        tf.math.add(
            tf.math.sqrt(beta),
            tf.constant(1E-7)
        )
    ) 
    return foo

def alpha_func_adamax(shape, alpha, grad):
    foo = tf.math.add(
            tf.math.multiply(
                tf.constant(-0.1),
                alpha
            ),
            tf.math.multiply(
                tf.constant(0.1),
                grad
            )
        )
    return tf.negative(foo)

def beta_func_adamax(shape, alpha, beta, grad):
    foo = tf.math.maximum(
            tf.math.multiply(
                tf.constant(-0.001),
                beta 
            ),
            tf.math.multiply(
                tf.constant(0.001),
                tf.math.square(grad)
            )
        )
    return tf.negative(foo)

def sigma_func_adamax(shape, alpha, beta, sigma, grad):
    foo = tf.constant(1.0)
    return tf.negative(foo)

def grad_func_adamax(shape, alpha, beta, sigma, grad):
    foo = tf.math.divide(
        tf.math.multiply(
            tf.math.multiply(
                tf.constant(0.001),
                tf.math.divide(
                    tf.math.sqrt(
                        tf.math.subtract(
                            tf.constant(1.0),
                            tf.math.pow(
                                tf.constant(0.999),
                                sigma
                            )
                        )
                    ),
                    tf.math.subtract(
                        tf.constant(1.0),
                        tf.math.pow(
                            tf.constant(0.9),
                            sigma
                        )
                    )
                )
            ),
            alpha
        ),
        tf.math.add(
            tf.math.sqrt(beta),
            tf.constant(1E-7)
        )
    ) 
    return foo

def print_op(tensor):
    inputs = ''
    for input in tensor.op.inputs:
        inputs += print_op(input) + ","
    return f"{tensor.name}({inputs[:-1]})"

def read_genotype(genome):
    random.seed(42)
    g = Grammar()
    g.set_path("grammars/adaptive_autolr_grammar.txt")
    g.set_min_init_tree_depth(1)
    g.set_max_tree_depth(17)
    g.read_grammar()

    
    mapping_numbers = [0] * len(genome)
    bar = g.mapping(genome, mapping_numbers)[0]
    foo = {"tf": tf}
    exec(bar, foo)
    alpha_func = foo["alpha_func"]
    beta_func = foo["beta_func"]
    sigma_func = foo["sigma_func"]
    grad_func = foo["grad_func"]

    alpha_tensor = alpha_func(tf.Variable(0.0).shape, tf.Variable(0.0), tf.Variable(0.0),)
    beta_tensor = beta_func(tf.Variable(0.0).shape, tf.Variable(0.0), tf.Variable(0.0), tf.Variable(0.0),)
    sigma_tensor = sigma_func(tf.Variable(0.0).shape, tf.Variable(0.0), tf.Variable(0.0),tf.Variable(0.0), tf.Variable(0.0),)
    grad_tensor = grad_func(tf.Variable(0.0).shape, tf.Variable(0.0), tf.Variable(0.0), tf.Variable(0.0), tf.Variable(0.0),)

    return [alpha_tensor, beta_tensor, sigma_tensor, grad_tensor], [alpha_func, beta_func, sigma_func, grad_func], bar

if __name__ == "__main__":
    #from benchmarks.evaluate_fashion_mnist_model import evaluate_fashion_mnist_model
    #from utils.custom_optimizer import CustomOptimizer
    from utils.genotypes import *
    

    #print(print_op(alpha_func_adam(tf.Variable(0.0).shape,tf.Variable(0.0),tf.Variable(0.0))))
    #print(print_op(beta_func_adam(tf.Variable(0.0).shape,tf.Variable(0.0),tf.Variable(0.0),tf.Variable(0.0))))
    #print(print_op(sigma_func_adam(tf.Variable(0.0).shape,tf.Variable(0.0),tf.Variable(0.0),tf.Variable(0.0),tf.Variable(0.0))))
    #print(print_op(grad_func_adam(tf.Variable(0.0).shape,tf.Variable(0.0),tf.Variable(0.0),tf.Variable(0.0),tf.Variable(0.0))))


    tensors, funcs, phen = read_genotype(get_rmsprop_genotype()) 

    print(phen)
    print("#")
    phen2 = "alpha_func, beta_func, sigma_func, grad_func = lambda shape,  alpha, grad: tf.math.negative(tf.math.add(tf.math.multiply(tf.math.subtract(tf.constant(0.0, shape=shape, dtype=tf.float32), tf.constant(1.07052146e-01, shape=shape, dtype=tf.float32)), alpha), tf.math.multiply(tf.constant(1.07052146e-01, shape=shape, dtype=tf.float32), tf.math.square(grad)))), lambda shape,  alpha, beta, grad: tf.math.negative(tf.math.add(tf.math.negative(beta), tf.math.divide_no_nan(tf.math.multiply(tf.constant(1.14904229e-03, shape=shape, dtype=tf.float32), grad), tf.math.sqrt(tf.math.add(alpha, tf.constant(5.55606489e-05, shape=shape, dtype=tf.float32)))))), lambda shape,  alpha, beta, sigma, grad: tf.math.negative(tf.math.add(tf.constant(5.55606489e-05, shape=shape, dtype=tf.float32), grad)), lambda shape,  alpha, beta, sigma, grad: beta"
    print(phen2)
    #print(print_op(tensors[0]))

    #opt = CustomOptimizer(grad_func=funcs[3], alpha_func=funcs[0], beta_func=funcs[1], sigma_func=funcs[2])
    #evaluate_fashion_mnist_model(optimizer=opt, batch_size=1000, epochs=10,experiment_name="momentum_functions", verbose=2)
    #foo = grad_func_momentum(tf.Variable(0.0),tf.Variable(0.0),tf.Variable(0.0),tf.Variable(0.0),)
    #print(print_op(foo))




