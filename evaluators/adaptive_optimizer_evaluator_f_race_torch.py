import csv
from utils.data_functions import load_fashion_mnist_training

import torch
import torchvision
import torchvision.transforms as transforms
from optimizers.custom_optimizer import CustomOptimizerTorch
import torch.nn as nn
import torch.nn.functional as F

import sys

import numpy as np
import datetime
experiment_time = datetime.datetime.now()

def train_model_torch(phen, params, net, train_loader, validation_loader, fitness_loader):
    epochs = params['EPOCHS']
    patience = params['PATIENCE']

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print("Using CPU")
        device = torch.device('cpu')

    criterion = nn.CrossEntropyLoss()

    optimizer = CustomOptimizerTorch(net.parameters(), phen=phen, device=device)

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    best_loss = None
    patience_counter = 0
    torch.backends.cudnn.benchmark = True
    for epoch in range(epochs):  # loop over the dataset multiple times
        training_loss = 0.0
        validation_loss = 0.0
        epoch_labels = []
        epoch_predictions = []
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data


            # forward + backward + optimize
            outputs = net(inputs.to(device))
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs.to(device), labels.to(device))
            loss.backward()
            optimizer.step()
            
            #training_loss += loss.item()
        with torch.no_grad():
            for i, data in enumerate(validation_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # forward + backward + optimize
                outputs = net(inputs.to(device))
                _, predictions = torch.max(outputs, 1)
                loss = criterion(outputs.to(device), labels.to(device))
                
                validation_loss += loss.item()

        if best_loss is None or best_loss > validation_loss:
            best_loss = validation_loss
            patience_counter = 0
        else:
            patience_counter += 1
        #print(f'[{epoch + 1}][{patience_counter}/{params["PATIENCE"]}] loss: {training_loss} val_loss:{validation_loss}')

        if loss.isnan() or patience_counter == params["PATIENCE"]:
            break
            
            

    #print('Finished Training')

    dataiter = iter(fitness_loader)
    images, labels = dataiter.next()

    _, predicted = torch.max(outputs, 1)

    #print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in fitness_loader:
            images, labels = data    
            outputs = net(images.to(device))    
            _, predictions = torch.max(outputs.to(device), 1)
            predictions = np.array(predictions.cpu())
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    total_accuracy = 0
    
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        total_accuracy += accuracy
        #print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
    total_accuracy /= len(classes)
    #print(f"Total Accuracy {total_accuracy}")
    return total_accuracy / 100, {}

if __name__ == "__main__":
    params = {
    'POPSIZE': 50,
    'GENERATIONS': 3,
    'ELITISM': 0,
    'PROB_CROSSOVER': 0.0,
    'PROB_MUTATION': 0.15,
    'TSIZE': 2,
    'GRAMMAR': 'grammars/adaptive_autolr_grammar_torch.txt',
    'EXPERIMENT_NAME': 'dumps/torch',
    'RUN': 1,
    'INCLUDE_GENOTYPE': True,
    'SAVE_STEP': 1,
    'VERBOSE': True,
    'MIN_TREE_DEPTH': 6,
    'MAX_TREE_DEPTH': 17,
    'MODEL': 'models/mnist_model.h5',
    'VALIDATION_SIZE': 3500,
    'FITNESS_SIZE': 35000,
    'BATCH_SIZE': 1000,
    'EPOCHS': 100,
    'PREPOPULATE': False,
    'PATIENCE': 5,
    'FAKE_FITNESS': False,
    'VERBOSE': False
}

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
            print(sum(p.numel() for p in self.net.parameters() if p.requires_grad))


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
    t = Optimizer_Evaluator_Torch(train_model_torch)
    t.evaluate("alpha_func, beta_func, sigma_func, grad_func = lambda size, alpha, grad, device: torch.sqrt(torch.add(torch.multiply(alpha, torch.negative(torch.multiply(torch.negative(grad), torch.full(size=size, fill_value = 9.96148968e-01, dtype=torch.float32, device=device)))), grad)), lambda size, alpha, beta, grad, device: grad, lambda size, alpha, beta, sigma, grad, device: torch.full(size=size, fill_value = 9.65554804e-01, dtype=torch.float32, device=device), lambda size, alpha, beta, sigma, grad, device: torch.negative(beta)",)

