import tensorflow as tf

def alpha_func_filler(alpha, grad):
    return grad

def beta_func_filler(alpha, beta, grad):
    return grad

def sigma_func_filler(alpha, beta, sigma, grad):
    return grad

def grad_func_filler(alpha, beta, sigma, grad):
    return tf.math.negative(alpha)

def alpha_func_momentum(alpha, grad):
    foo = tf.math.negative(
                    tf.math.subtract(
                                tf.math.multiply(
                                    tf.math.subtract(
                                            tf.constant(0.99), 
                                            tf.constant(1.0)),
                                    alpha,
                                ),
                                tf.math.multiply(
                                        tf.constant(0.01),
                                        grad
                                )
                        )
                    )
    return foo

def grad_func_momentum(alpha, beta, sigma, grad):
    return tf.math.negative(alpha)

def alpha_func_nesterov(alpha, grad):
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

def grad_func_nesterov(alpha, beta, sigma, grad):
    return tf.math.negative(alpha)

def alpha_func_adagrad(alpha, grad):
    foo = tf.negative(tf.math.square(grad))
    return foo

def grad_func_adagrad(alpha, beta, sigma, grad):
    return tf.math.multiply(
            grad,
            tf.math.divide_no_nan(
                tf.constant(0.001),
                tf.math.add(
                    tf.math.sqrt(alpha),
                    tf.constant(1e-7)
                )
            )
        )

def alpha_func_rmsprop(alpha, grad):
    foo = tf.math.add(
            tf.math.multiply(
                tf.constant(-0.1),
                alpha
            ),
            tf.math.multiply(
                tf.constant(0.1),
                tf.math.square(grad)
            )
        )
    return tf.negative(foo)

def beta_func_rmsprop(alpha, beta, grad):
    foo = tf.math.add(
        tf.math.multiply(
            tf.constant(-0.01),
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

def grad_func_rmsprop(alpha, beta, sigma, grad):
    return beta

def alpha_func_adam(alpha, grad):
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

def beta_func_adam(alpha, beta, grad):
    foo = tf.math.add(
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

def sigma_func_adam(alpha, beta, sigma, grad):
    foo = tf.constant(1.0)
    return tf.negative(foo)

def grad_func_adam(alpha, beta, sigma, grad):
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

def alpha_func_adam(alpha, grad):
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

def beta_func_adam(alpha, beta, grad):
    foo = tf.math.add(
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

def sigma_func_adam(alpha, beta, sigma, grad):
    foo = tf.constant(1.0)
    return tf.negative(foo)

def grad_func_adam(alpha, beta, sigma, grad):
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

def alpha_func_adamax(alpha, grad):
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

def beta_func_adamax(alpha, beta, grad):
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

def sigma_func_adamax(alpha, beta, sigma, grad):
    foo = tf.constant(1.0)
    return tf.negative(foo)

def grad_func_adamax(alpha, beta, sigma, grad):
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