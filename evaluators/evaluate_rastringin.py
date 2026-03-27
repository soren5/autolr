from dataset_loaders.fmnist import FMNIST_Dataset
from evaluators.evaluator_utils import Evaluator
from optimizers.custom_optimizer import CustomOptimizer
import tensorflow as tf
import numpy as np

class Rastringin_Evaluator(Evaluator):
    def __init__(self, params, configuration_file='RASTRINGIN_CONFIG', task_name='rastringin', limit=5.12, start_point=[3.1, 4.2], epochs=1000, scale=1.0 / 40.0):
        self.limit = limit
        self.start_point = start_point
        self.epochs = epochs
        self.scale = scale
    
    def evaluate(self, phen):
        # Use a copy of self.start_point for each optimizer to avoid in-place modifications that affect other optimizers
        x_var = tf.Variable(self.start_point[0], dtype=tf.float32, name='x')
        y_var = tf.Variable(self.start_point[1], dtype=tf.float32, name='y')
        evaluation_start_point = [x_var, y_var]
        path = []


        opt = CustomOptimizer(phen=phen, model=evaluation_start_point)
        diverged = False
        best_loss = float('inf')
        path.append(np.array([x_var.numpy().item(), y_var.numpy().item()]))
        for epoch in range(self.epochs):
            if not diverged:
                # Compute the loss and gradients
                with tf.GradientTape() as tape:
                    loss = self.cost_func(x_var, y_var)
                best_loss = min(best_loss, loss.numpy().item())
                grads = tape.gradient(loss, [x_var, y_var])

                # Use the optimizer to apply the gradients to the variables
                opt.apply_gradients(zip(grads, [x_var, y_var]))
                #print(f"Optimizer {methods[i]} at epoch {epoch} has x={x_var.numpy().item()} and y={y_var.numpy().item()} with loss={loss.numpy().item()}")
                    
                try:
                    if x_var.numpy().item() > self.limit or x_var.numpy().item() < -self.limit or y_var.numpy().item() > self.limit or y_var.numpy().item() < -self.limit:
                        x_var.assign(path[-1][0])
                        y_var.assign(path[-1][1])
                        diverged = True
                except AttributeError:
                    if x_var > self.limit or x_var < -self.limit or y_var > self.limit or y_var < -self.limit:
                        x_var.assign(path[-1][0])
                        y_var.assign(path[-1][1])
                        diverged = True

            # Log this step's variables to paths_
            path.append(np.array([x_var.numpy().item(), 
                                                y_var.numpy().item()]))
        if diverged:
            best_loss = float('inf')
        return best_loss, np.array(path)

    def evaluate_optimizer(self, opt):
        # Use a copy of self.start_point for each optimizer to avoid in-place modifications that affect other optimizers
        x_var = tf.Variable(self.start_point[0], dtype=tf.float32)
        y_var = tf.Variable(self.start_point[1], dtype=tf.float32)
        evaluation_start_point = [x_var, y_var]
        path = []
        diverged = False
        best_loss = float('inf')
        path.append(np.array([x_var.numpy().item(), y_var.numpy().item()]))
        for epoch in range(self.epochs):
            if not diverged:
                # Compute the loss and gradients
                with tf.GradientTape() as tape:
                    loss = self.cost_func(x_var, y_var)
                best_loss = min(best_loss, loss.numpy().item())
                grads = tape.gradient(loss, [x_var, y_var])

                # Use the optimizer to apply the gradients to the variables
                opt.apply_gradients(zip(grads, [x_var, y_var]))
                #print(f"Optimizer {methods[i]} at epoch {epoch} has x={x_var.numpy().item()} and y={y_var.numpy().item()} with loss={loss.numpy().item()}")
                    
                try:
                    if x_var.numpy().item() > self.limit or x_var.numpy().item() < -self.limit or y_var.numpy().item() > self.limit or y_var.numpy().item() < -self.limit:
                        x_var.assign(path[-1][0])
                        y_var.assign(path[-1][1])
                        diverged = True
                except AttributeError:
                    if x_var > self.limit or x_var < -self.limit or y_var > self.limit or y_var < -self.limit:
                        x_var.assign(path[-1][0])
                        y_var.assign(path[-1][1])
                        diverged = True

            # Log this step's variables to paths_
            path.append(np.array([x_var.numpy().item(), 
                                                y_var.numpy().item()]))
        if diverged:
            best_loss = float('inf')
        return best_loss, np.array(path)
    
    def cost_func(self, x, y):
        z = 10 * 2 + (tf.pow(x, 2) - 10 * tf.cos(2 * np.pi * x)) + (tf.pow(y, 2) - 10 * tf.cos(2 * np.pi * (y)))
        return z * self.scale # We try to divide by 40 to get the values in a similar range as the default cost function