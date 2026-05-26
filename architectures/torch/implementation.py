# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
#             INRIA
#
# University: Universidade Federal do Rio de Janeiro (UFRJ) - Brazil  
#             Ecole Polytechnique - France
#

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from .resnet import ResNet18

from .custom_models import (
        Net,
        resnet10
)

from timeit import default_timer as timer

from utils.torch.utils import allocate_cuda

# create a model
def build_model(features_shape:int=3,
                labels_shape:int=10,
                client_id:int=0,
                model_name:str="RESNET18",
                lr:float=0.1):

    model = criterion = optimizer = device = scheduler = None

    device = "cuda:0"
     
    if model_name == "RESNET18":

        model = ResNet18(num_classes=labels_shape)
        
        if features_shape != 3:

            original = model.conv1

            model.conv1 = nn.Conv2d(features_shape,
                                    original.out_channels,
                                    kernel_size=original.kernel_size,
                                    stride=original.stride,
                                    padding=original.padding,
                                    bias=(original.bias is not None))

        criterion = nn.CrossEntropyLoss()
    
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=0.9, 
                                    weight_decay=5e-4)
   
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    elif model_name == "RESNET34":

        model = torchvision.models.resnet34(weights=None)
        
        if features_shape != 3:

            original = model.conv1

            model.conv1 = nn.Conv2d(features_shape,
                                    original.out_channels,
                                    kernel_size=original.kernel_size,
                                    stride=original.stride,
                                    padding=original.padding,
                                    bias=(original.bias is not None))

        model.fc = nn.Linear(model.fc.in_features, 
                             labels_shape) 

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=0.9, 
                                    weight_decay=5e-4)
   
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        
    elif model_name == "MOBILENETV2":

        model = torchvision.models.mobilenet_v2(weights=None)
        
        if features_shape != 3:

            original = model.conv1

            model.conv1 = nn.Conv2d(features_shape,
                                    original.out_channels,
                                    kernel_size=original.kernel_size,
                                    stride=original.stride,
                                    padding=original.padding,
                                    bias=(original.bias is not None))

        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 
                                        labels_shape)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=0.9, 
                                    weight_decay=5e-4)
   
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
  

    elif model_name == "RESNET10":
        
        model = resnet10(num_classes=labels_shape)
        
        if features_shape != 3:

            original = model.conv1

            model.conv1 = nn.Conv2d(features_shape,
                                    original.out_channels,
                                    kernel_size=original.kernel_size,
                                    stride=original.stride,
                                    padding=original.padding,
                                    bias=(original.bias is not None))

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=0.9, 
                                    weight_decay=5e-4)
   
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        

    elif model_name == "CNN":

        model = Net(num_classes=labels_shape,
                    features_shape=features_shape)
        
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr,
                                    momentum=0.9, 
                                    weight_decay=5e-4)
   
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        
    else:
        
        raise ValueError('Model not implemented')
        
    return model, criterion, optimizer, device, scheduler


def train(model, 
          n_epochs, 
          optimizer, 
          criterion,
          scheduler,
          device,
          trainloader,
          logger):
    
    model.train()
    running_loss = 0.0
    
    for epoch in range(n_epochs):
        
        for index, data in enumerate(trainloader):
            
            if len(data[0]) >= 2:

                images, labels = data
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                loss = criterion(model(images), labels)
                
                loss.backward()

                optimizer.step()
    
                running_loss += loss.item()

            else:

                logger.debug(f'data batch size less than 2: {len(data[0])}')
    
    scheduler.step()

    avg_trainloss = running_loss / len(trainloader.dataset)
    
    return avg_trainloss


def evaluate(model,
             device,
             criterion,
             testloader,
             logger=None):

    model.eval()
    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for data in testloader:
            
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct/total, loss

def train_eval(model, 
               n_epochs, 
               optimizer, 
               criterion,
               scheduler,
               device,
               trainloader,
               testloader,
               RESULT_PATH,
               exec_id,
               logger):
    
    # """Train the model on the training set."""
    best_acc = 0
    running_loss = 0.0
    
    for epoch in range(n_epochs):
        
        model.train()
        

        for index, data in enumerate(trainloader):
            
            if len(data[0]) >= 2:

                images, labels = data

                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                loss = criterion(model(images), labels)
                
                loss.backward()

                optimizer.step()

                scheduler.step()
                
                running_loss += loss.item()
        
        
        test_acc, loss = evaluate(model,
                                  device,
                                  criterion,
                                  testloader,
                                  logger)

        print(f'acc : {test_acc}, loss: {loss}, epoch: {epoch}')

        logger.debug(f'accuracy {test_acc}, loss {loss}')

        with open(f"{RESULT_PATH}/{exec_id}", "a") as writer:
              
            writer.writelines(f"{test_acc:.9f}\n")

    avg_trainloss = running_loss / len(trainloader.dataset)
    
    return avg_trainloss


def get_weights(model):

    return [ val.cpu().numpy() for _, val in model.state_dict().items() ]
