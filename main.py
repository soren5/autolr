import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from sge.parameters import (
    params,
    set_parameters
)
from utils.xor_sanity_check import xor_check
class Optimizer_Evaluator_Tensorflow:
    def __init__(self, train_model=None):  #should give a function 
        if train_model == None: 
            from evaluators.adaptive_optimizer_evaluator_f_race import train_model_tensorflow_fmnist as train_model
        self.train_model = train_model
    
    def evaluate(self, phen, params):
        #if xor_check(phen):
        if True:
            foo = self.train_model([phen, params])
            fit = -foo[0]
            other_info = foo[1]
            other_info['source'] = 'evaluation'
        else:
            fit = -0.1
            other_info = {'source': 'sanity check'}
        return fit, other_info

    def init_net(self, params):
        pass
    def init_data(self, params):
        pass
    def init_evaluation(self, params):
        pass

class Optimizer_Evaluator_Dual_Task:
    def __init__(self):  #should give a function 
        pass
    
    def evaluate(self, phen, params):
        #print(params["CURRENT_GEN"])
        """
        if params["CURRENT_GEN"] % 2 == 0:
            foo = self.train_model_fmnist(phen)
            fit = -foo[0]
            other_info = foo[1]
            other_info['task'] = 'fmnist'
        elif params["CURRENT_GEN"] % 2 == 1:
            print("Running CIFAR")
            foo = self.train_model_cifar(phen)
            fit = -foo[0]
            other_info = foo[1]
            other_info['task'] = 'cifar'
        else:
            raise Exception("CURRENT GEN IN EVALUATE IS INVALID")
        """
        if xor_check(phen):
            foo = self.train_model_fmnist(phen)
            fit = -foo[0]
            other_info = foo[1]
            other_info['source'] = 'evaluation'
        else:
            fit = -0.1
            other_info = {'source': 'sanity check'}
        return fit, other_info

    def init_net(self, params):
        from models.keras_model_adapter import VGG16_Interface, MobileNet_Interface
        self.fmnist_model_interface = VGG16_Interface()
        self.cifar_model_interface = MobileNet_Interface()

        
    def init_data(self, params):
        from utils.data_functions import load_fashion_mnist_training, load_cifar10_training, load_mnist_training, select_fashion_mnist_training
        import tensorflow as tf
        import numpy as np
        training_size = params['TRAINING_SIZE']
        validation_size = params['VALIDATION_SIZE']


        self.fmnist_data = {}

        data = load_fashion_mnist_training(training_size=training_size, validation_size=validation_size, normalize=False, subtract_mean=False)
        with tf.device('/cpu:0'):
            for key in data:
                if 'x' in key:
                    self.fmnist_data[key] = tf.convert_to_tensor(self.fmnist_model_interface.prepare_input(data[key]), np.float32)
                else:
                    self.fmnist_data[key] = data[key]

            self.cifar_data = {}
            data = load_cifar10_training(training_size=training_size, validation_size=validation_size)
            for key in data:
                if 'x' in key:
                    self.cifar_data[key] = tf.convert_to_tensor(self.cifar_model_interface.prepare_input(data[key]), np.float32)
                else:
                    self.cifar_data[key] = data[key]

    def init_evaluation(self, params):
        from evaluators.adaptive_optimizer_evaluator_f_race import create_train_model
        fmnist_model = self.fmnist_model_interface.get_model()
        cifar_model = self.cifar_model_interface.get_model()

        self.train_model_fmnist = create_train_model(fmnist_model, self.fmnist_data, fmnist_model.get_weights())
        self.train_model_cifar = create_train_model(cifar_model, self.cifar_data, cifar_model.get_weights())

class Optimizer_Evaluator_Torch:
    def __init__(self, train_model=None): 
        import torch  
        if train_model == None: 
            from evaluators.adaptive_optimizer_evaluator_f_race_torch import train_model_torch as train_model_torch
        self.train_model = train_model_torch
        self.net = None
        self.trainloader = None
        self.testloader = None


        
    def init_net(self, params):
        import torch.nn as nn
        import torch.nn.functional as F
        import torch
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                else:
                    device = torch.device('cpu')
                self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=device, dtype=None)
                self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=device, dtype=None)
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(0.25)
                self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,  kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=device, dtype=None)
                self.conv4 = nn.Conv2d(in_channels=64, out_channels=64,  kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=device, dtype=None)
                self.fc1 = nn.Linear(1600, 512, device=device)
                self.dropout2 = nn.Dropout(0.5)
                self.fc2 = nn.Linear(512, 10, device=device)
                self.softmax = nn.Softmax()

            def forward(self, x):
                x = self.conv1(x)
                #print(f"conv1: {x.shape}")
                x = F.relu(x)
                #print(f"Relu: {x.shape}")
                x = self.conv2(x)
                #print(f"conv2: {x.shape}")
                x = F.relu(x)
                #print(f"Relu: {x.shape}")
                x = self.pool(x)
                #print(f"Pool: {x.shape}")
                x = self.dropout(x)
                #print(f"Dropout: {x.shape}")
                x = self.conv3(x)
                #print(f"conv3: {x.shape}")
                x = F.relu(x)
                #print(f"ReLU: {x.shape}")
                x = self.conv4(x)
                #print(f"conv4: {x.shape}")
                x = F.relu(x)
                #print(f"relu: {x.shape}")
                x = self.pool(x)
                #print(f"Pool: {x.shape}")
                x = self.dropout(x)
                #print(f"Dropout: {x.shape}")
                x = torch.flatten(x, 1) # flatten all dimensions except batch
                #print(f"Flatten: {x.shape}")
                x = self.fc1(x)
                #print(f"Dense1: {x.shape}")
                x = self.dropout2(x)
                #print(f"Dropout: {x.shape}")
                x = self.softmax(self.fc2(x))
                return x

        self.net = Net()
        for param in self.net.parameters():
            param.grad = None
        torch.save(self.net.state_dict(), './cifar_net.pth')
        #print(sum(p.numel() for p in self.net.parameters() if p.requires_grad))


    def init_data(self, params):
        import torch
        import torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        batch_size = params['BATCH_SIZE']

        cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        train_data, validation_data, fitness_data = torch.utils.data.random_split(cifar10, [len(cifar10) - params['VALIDATION_SIZE'] - params['FITNESS_SIZE'], params['VALIDATION_SIZE'], params['FITNESS_SIZE']])
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                shuffle=True, num_workers=1, pin_memory=True)
        self.validation_loader =  torch.utils.data.DataLoader(validation_data,
                                                shuffle=True, num_workers=1, pin_memory=True)
        self.fitness_loader = torch.utils.data.DataLoader(fitness_data,
                                                shuffle=False, num_workers=1, pin_memory=True)
    def evaluate(self, phen, params):
        import torch
        self.net.load_state_dict(torch.load('./cifar_net.pth'))
        value, other_info = self.train_model(phen, params, self.net, self.train_loader, self.validation_loader, self.fitness_loader)
        return -value, other_info
    def init_evaluation(self, params):
        pass

if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    import sge
    import sys

    set_parameters(sys.argv[1:])   

    if False:
        evaluation_function = Optimizer_Evaluator_Torch()
        if 'MODEL' in params and params['MODEL'] == 'models/cifar_model.h5': 
            from evaluators.adaptive_optimizer_evaluator_f_race import train_model_tensorflow_cifar10
            evaluation_function = Optimizer_Evaluator_Tensorflow(train_model=train_model_tensorflow_cifar10)
        elif 'MODEL' in params and params['MODEL'] == 'models/mnist_model.h5' and params['DATASET'] == 'fmnist':    
            from evaluators.adaptive_optimizer_evaluator_f_race import train_model_tensorflow_fmnist 
            evaluation_function = Optimizer_Evaluator_Tensorflow(train_model_tensorflow_fmnist)
        elif 'MODEL' in params and params['MODEL'] == 'models/mnist_model.h5' and params['DATASET'] == 'mnist':    
            from evaluators.adaptive_optimizer_evaluator_f_race import train_model_tensorflow_mnist 
            evaluation_function = Optimizer_Evaluator_Tensorflow(train_model_tensorflow_mnist)

    #sge.evolutionary_algorithm(evaluation_function=Optimizer_Evaluator_Dual_Task())
    sge.evolutionary_algorithm(evaluation_function=Optimizer_Evaluator_Tensorflow())  

