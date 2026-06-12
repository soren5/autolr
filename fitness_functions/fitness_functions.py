import numpy as np
from utils.xor_sanity_check import xor_check

class Optimizer_Evaluator_Tensorflow:
    def __init__(self, params, evaluator=None):  #should give a function 
        if evaluator is None:
            # If we have no evaluator, let's infer it from the model in params
            if 'cifar' in params['MODEL']:
                from evaluators.evaluate_cifar10 import CIFAR10_Evaluator
                evaluator = CIFAR10_Evaluator(params)
            elif 'resnet' in params['MODEL']:
                from evaluators.evaluate_tiny_imagenet import TINY_IMAGENET_Evaluator
                evaluator = TINY_IMAGENET_Evaluator(params)
            elif 'mnist' in params['MODEL']:
                from evaluators.evaluate_fmnist import FMNIST_Evaluator
                evaluator = FMNIST_Evaluator(params)
            else:
                # If nothing else, default to mnist
                from evaluators.evaluate_mnist import MNIST_Evaluator
                evaluator = MNIST_Evaluator(params)
            self.evaluator = evaluator
        else:
            self.evaluator = evaluator(params)
    
    def evaluate(self, phen, params):
        #print(f"\n\n\nTesting phenotype {smart_phenotype(phen)}:\n{readable_phenotype(phen)}")
        #if xor_check(phen):
        if True:
            foo = self.evaluator.evaluate(phen)
            fit = -foo[0]
            if np.isnan(fit):
                print("NAN fitness, returning FITNESS_FLOOR")
                fit = params['FITNESS_FLOOR']
            print(f"Fitness: {fit}")
            other_info = foo[1]
            other_info['source'] = 'evaluation'
        else:
            fit = params['FITNESS_FLOOR']
            other_info = {'source': 'degenerate detection'}
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
            fit = params['FITNESS_FLOOR']
            other_info = {'source': 'degenerate detection'}
        return fit, other_info

    def init_net(self, params):
        from models.keras_model_adapters.keras_model_adapter import VGG16_Interface, MobileNet_Interface
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
        from evaluators.evaluate_fmnist import create_train_model
        fmnist_model = self.fmnist_model_interface.get_model()
        cifar_model = self.cifar_model_interface.get_model()

        self.train_model_fmnist = create_train_model(fmnist_model, self.fmnist_data, fmnist_model.get_weights())
        self.train_model_cifar = create_train_model(cifar_model, self.cifar_data, cifar_model.get_weights())
class Optimizer_Evaluator_Torch:
    def __init__(self, train_model=None): 
        import torch  
        if train_model == None: 
            from evaluators.evaluate_pytorch import train_model_torch as train_model_torch
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
class Optimizer_Evaluator_Multi_Task:
    def __init__(self, params):  #should give a function 
        # The tasks included are determined by which configurations are loaded in the parameters
        task_configs = ['FMNIST_CONFIG', 'CIFAR10_CONFIG', 'CIFAR100_CONFIG', 'TINY_IMAGENET_CONFIG']
        
        if 'FMNIST_CONFIG' in params:    
            from evaluators.evaluate_fmnist import FMNIST_Evaluator
            self.fmnist_evaluator = FMNIST_Evaluator(params, configuration_file='FMNIST_CONFIG', task_name='fmnist')
        else: 
            print("WARNING: FMNIST_CONFIG not found in params, skipping FMNIST evaluation.")
            self.fmnist_evaluator = None

        if 'CIFAR10_CONFIG' in params:
            from evaluators.evaluate_cifar10 import CIFAR10_Evaluator
            self.cifar10_evaluator = CIFAR10_Evaluator(params, configuration_file='CIFAR10_CONFIG', task_name='cifar10')
        else:
            print("WARNING: CIFAR10_CONFIG not found in params, skipping CIFAR10 evaluation.")
            self.cifar10_evaluator = None

        if 'CIFAR100_CONFIG' in params:
            from evaluators.evaluate_cifar100 import CIFAR100_Evaluator
            self.cifar100_evaluator = CIFAR100_Evaluator(params, configuration_file='CIFAR100_CONFIG', task_name='cifar100')
        else:
            print("WARNING: CIFAR100_CONFIG not found in params, skipping CIFAR100 evaluation.")
            self.cifar100_evaluator = None
        
        if 'TINY_IMAGENET_CONFIG' in params:
            from evaluators.evaluate_tiny_imagenet import TINY_IMAGENET_Evaluator
            self.tiny_imagenet_evaluator = TINY_IMAGENET_Evaluator(params, configuration_file='TINY_IMAGENET_CONFIG', task_name='tiny_imagenet')
        else:
            print("WARNING: TINY_IMAGENET_CONFIG not found in params, skipping Tiny Imagenet evaluation.")
            self.tiny_imagenet_evaluator = None


    def evaluate(self, phen, params, opt=None):
        #print(f"\n\n\nTesting phenotype {smart_phenotype(phen)}:\n{readable_phenotype(phen)}")
        #if xor_check(phen):
        fmnist_results = (0.0, {})
        cifar10_results = (0.0, {})
        cifar100_results = (0.0, {})
        tiny_imagenet_results = (0.0, {})

        if opt is not None:
            print(f"WARNING: Evaluating an optimizer, if this is an evolution experiment it is compromised.")

        other_info = {}
        multi_task_record = self._init_multi_task_record(params)
        other_info['multi_task'] = multi_task_record
        evaluated_tasks = 0
        if self.fmnist_evaluator is not None:
            # Evaluate FMNIST
            fmnist_results = self.fmnist_evaluator.evaluate(phen) if opt is None else self.fmnist_evaluator.evaluate_optimizer(opt)
            fitness = fmnist_results[0] + evaluated_tasks
            other_info['fmnist'] = fmnist_results[1]
            other_info['source'] = 'fmnist_evaluation'
            self._record_multi_task_result(multi_task_record, 'fmnist', fmnist_results[0], params['FMNIST_THRESHOLD'])
            if fitness <= params['FMNIST_THRESHOLD'] + evaluated_tasks:
                multi_task_record['failed_task'] = 'fmnist'
                fitness = -fitness
                return fitness, other_info
            evaluated_tasks += 1

        if self.cifar10_evaluator is not None:
            cifar10_results = self.cifar10_evaluator.evaluate(phen) if opt is None else self.cifar10_evaluator.evaluate_optimizer(opt)
            fitness = cifar10_results[0] + evaluated_tasks
            other_info['cifar10'] = cifar10_results[1]
            other_info['source'] = 'cifar10_evaluation'
            self._record_multi_task_result(multi_task_record, 'cifar10', cifar10_results[0], params['CIFAR10_THRESHOLD'])
            if fitness <= params['CIFAR10_THRESHOLD'] + evaluated_tasks:
                multi_task_record['failed_task'] = 'cifar10'
                fitness = -fitness
                return fitness, other_info
            evaluated_tasks += 1

        if self.cifar100_evaluator is not None:
            cifar100_results = self.cifar100_evaluator.evaluate(phen) if opt is None else self.cifar100_evaluator.evaluate_optimizer(opt)
            fitness = cifar100_results[0] + evaluated_tasks
            other_info['cifar100'] = cifar100_results[1]
            other_info['source'] = 'cifar100_evaluation'
            self._record_multi_task_result(multi_task_record, 'cifar100', cifar100_results[0], params['CIFAR100_THRESHOLD'])
            if fitness <= params['CIFAR100_THRESHOLD'] + evaluated_tasks:
                multi_task_record['failed_task'] = 'cifar100'
                fitness = -fitness
                return fitness, other_info
            evaluated_tasks += 1

        if self.tiny_imagenet_evaluator is not None:
            tiny_imagenet_results = self.tiny_imagenet_evaluator.evaluate(phen) if opt is None else self.tiny_imagenet_evaluator.evaluate_optimizer(opt)
            fitness = tiny_imagenet_results[0] + evaluated_tasks
            other_info['tiny_imagenet'] = tiny_imagenet_results[1]
            other_info['source'] = 'tiny_imagenet_evaluation'
            self._record_multi_task_result(multi_task_record, 'tiny_imagenet', tiny_imagenet_results[0], params['TINY_IMAGENET_THRESHOLD'])
            if fitness <= params['TINY_IMAGENET_THRESHOLD'] + evaluated_tasks:
                multi_task_record['failed_task'] = 'tiny_imagenet'
                fitness = -fitness
                return fitness, other_info
            evaluated_tasks += 1
            
        print(f"Fitness: {fitness:.4f} (fmnist: {fmnist_results[0]:.4f}, cifar10: {cifar10_results[0]:.4f}, cifar100: {cifar100_results[0]:.4f}, tiny_imagenet: {tiny_imagenet_results[0]:.4f})")
        
        # Negate fitness because evolutionary algorithm is minimizing.
        fitness = -fitness
        return fitness, other_info

    def _init_multi_task_record(self, params):
        task_order = []
        thresholds = {}
        evaluator_thresholds = [
            ('fmnist', self.fmnist_evaluator, 'FMNIST_THRESHOLD'),
            ('cifar10', self.cifar10_evaluator, 'CIFAR10_THRESHOLD'),
            ('cifar100', self.cifar100_evaluator, 'CIFAR100_THRESHOLD'),
            ('tiny_imagenet', self.tiny_imagenet_evaluator, 'TINY_IMAGENET_THRESHOLD'),
        ]
        for task_name, evaluator, threshold_key in evaluator_thresholds:
            if evaluator is not None:
                task_order.append(task_name)
                thresholds[task_name] = params[threshold_key]
        return {
            'task_order': task_order,
            'scores': {task: None for task in task_order},
            'thresholds': thresholds,
            'passed': {task: None for task in task_order},
            'reached_depth': 0,
            'failed_task': None,
        }

    def _record_multi_task_result(self, multi_task_record, task_name, score, threshold):
        multi_task_record['scores'][task_name] = score
        multi_task_record['thresholds'][task_name] = threshold
        multi_task_record['passed'][task_name] = score > threshold
        multi_task_record['reached_depth'] += 1

    def init_net(self, params):
        pass
    def init_data(self, params):
        pass
    def init_evaluation(self, params):
        pass

class Optimizer_Evaluator_FMNIST_CIFAR10_CIFAR100_TIN():
    def __init__(self, params):  #should give a function 
        from evaluators.evaluate_fmnist import FMNIST_Evaluator
        self.fmnist_evaluator = FMNIST_Evaluator(params, configuration_file='FMNIST_CONFIG', task_name='fmnist')

        from evaluators.evaluate_cifar10 import CIFAR10_Evaluator
        self.cifar10_evaluator = CIFAR10_Evaluator(params, configuration_file='CIFAR10_CONFIG', task_name='cifar10')

        from evaluators.evaluate_cifar100 import CIFAR100_Evaluator
        self.cifar100_evaluator = CIFAR100_Evaluator(params, configuration_file='CIFAR100_CONFIG', task_name='cifar100')

        from evaluators.evaluate_tiny_imagenet import TINY_IMAGENET_Evaluator
        self.tiny_imagenet_evaluator = TINY_IMAGENET_Evaluator(params, configuration_file='TINY_IMAGENET_CONFIG', task_name='tiny_imagenet')
    
    def evaluate(self, phen, params, opt=None):
        #print(f"\n\n\nTesting phenotype {smart_phenotype(phen)}:\n{readable_phenotype(phen)}")
        #if xor_check(phen):
        fmnist_results = cifar10_results = cifar100_results = tiny_imagenet_results = (0.0, {})  # Default results in case we skip evaluation

        if opt is not None:
            print(f"WARNING: Evaluating an optimizer, if this is an evolution experiment it is compromised.")
        if True:
            other_info = {}
            fmnist_results = self.fmnist_evaluator.evaluate(phen) if opt is None else self.fmnist_evaluator.evaluate_optimizer(opt)
            fitness = fmnist_results[0]
            other_info['fmnist'] = fmnist_results[1]
            other_info['source'] = 'fmnist_evaluation'
            
            if fitness > params['FMNIST_THRESHOLD']:
            #if True:
                #Evaluate CIFAR
                cifar10_results = self.cifar10_evaluator.evaluate(phen) if opt is None else self.cifar10_evaluator.evaluate_optimizer(opt)
                fitness = cifar10_results[0] + 1.0
                other_info['cifar10'] = cifar10_results[1]
                other_info['source'] = 'cifar10_evaluation'
                if fitness > 1.0 + params['CIFAR10_THRESHOLD']:
                    #Evaluate CIFAR100
                    cifar100_results = self.cifar100_evaluator.evaluate(phen) if opt is None else self.cifar100_evaluator.evaluate_optimizer(opt)
                    fitness = cifar100_results[0] + 2.0
                    other_info['cifar100'] = cifar100_results[1]
                    other_info['source'] = 'cifar100_evaluation'

                    if fitness > 2.0 + params['CIFAR100_THRESHOLD']:
                        #Evaluate Imagenet
                        tiny_imagenet_results = self.tiny_imagenet_evaluator.evaluate(phen) if opt is None else self.tiny_imagenet_evaluator.evaluate_optimizer(opt)
                        fitness = tiny_imagenet_results[0] + 3.0
                        other_info['tiny_imagenet'] = tiny_imagenet_results[1]
                        other_info['source'] = 'tiny_imagenet_evaluation'
        fitness = - fitness
        return fitness, other_info
    def init_net(self, params):
        pass
    def init_data(self, params):
        pass
    def init_evaluation(self, params):
        pass
