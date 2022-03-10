import tensorflow as tf
from tensorflow import keras
from tensorflow.python.training import training_ops
from optimizers.custom_optimizer import CustomOptimizer

class ADES(CustomOptimizer):
    def __init__(self,
                            name="ADES",
                            alpha=None,
                            beta_1=0.08922,
                            beta_2=0.0891,
                            **kwargs):

        super(ADES, self).__init__(name, **kwargs)

        self._alpha_dict = alpha
        self._alpha_func = lambda shape, alpha, grad: tf.math.add(
                tf.math.multiply(
                    tf.constant(beta_1, shape=shape, dtype=tf.float32),
                    tf.math.multiply(
                        alpha,
                        alpha
                    )
                ),
                tf.math.add(
                    tf.math.multiply(
                        tf.math.multiply(
                            alpha,
                            grad
                        ),
                        tf.constant(beta_2, shape=shape, dtype=tf.float32)
                    ),
                    tf.math.multiply(
                        tf.constant(beta_2, shape=shape, dtype=tf.float32),
                        grad
                    )
                )
            )
        self._grad_func = lambda shape, alpha, grad: tf.math.negative(alpha)

