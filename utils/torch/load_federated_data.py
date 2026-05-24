# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
#             INRIA
#
# University: Universidade Federal do Rio de Janeiro (UFRJ) - Brazil  
#             Ecole Polytechnique - France
#

import skimage
import torch

import numpy as np

from pickle import load
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from torchvision import (
        datasets, 
        transforms)

def load_data_client(dataset_name='CIFAR-10',   # dataset used on the system
                     clientID=1,                # client identifier
                     numClients=1,              # number of clients
                     trPer=0.8,                 # percentage of test samples
                     distribution="dirichlet",  # manual or dirichlet
                     alpha=0.5):                # dirichlet distribution parameter

    if distribution == "manual":

        X = np.array([],
                     dtype=np.float32)
        
        Y = np.array([],
                     dtype=np.float32)

        for i in range(0,10):

            X_train_current = np.array([],dtype=np.float32)
            Y_train_current = np.array([],dtype=np.float32)
            X_test_current = np.array([],dtype=np.float32)
            Y_test_current = np.array([],dtype=np.float32)

            X_train_current = np.asarray(load(open('datasets/'+
                dataset_name+'/class'+str(i)+'Train','rb')), dtype=np.float32)
            
            X_test_current = np.asarray(load(open('datasets/'+
                dataset_name+'/class'+str(i)+'Test','rb')), dtype=np.float32)

            Y_train_current = np.asarray(load(open('datasets/'+
                dataset_name+'/class'+str(i)+'TrainLabel','rb')), dtype=np.float32)


            Y_test_current = np.asarray(load(open('datasets/'+
                dataset_name+'/class'+str(i)+'TestLabel','rb')),dtype=np.float32)

            begin_slice_train = int(len(Y_train_current)/numClients*(clientID-1))
            end_slice_train = int(len(Y_train_current)/numClients*clientID)
            begin_slice_test = int(len(Y_test_current)/numClients*(clientID-1))
            end_slice_test = int(len(Y_test_current)/numClients*clientID)

            X_train_current = X_train_current[begin_slice_train:end_slice_train]
            Y_train_current = Y_train_current[begin_slice_train:end_slice_train]
            X_test_current = X_test_current[begin_slice_test:end_slice_test]
            Y_test_current = Y_test_current[begin_slice_test:end_slice_test]
    
            if len(X) == 0:

                X = np.concatenate((X_train_current,X_test_current))
                Y = np.concatenate((Y_train_current,Y_test_current))

            else:

                X = np.concatenate((X,X_train_current,X_test_current))
                Y = np.concatenate((Y,Y_train_current,Y_test_current))
    
        # reshape MNIST and FMNIST
        if dataset_name == "MNIST" or dataset_name == "FMNIST":
            
            X = skimage.transform.resize(X, (len(X), 32, 32, 1))

        X, Y = shuffle(X, Y, random_state=47527)

        x_train, x_test, y_train, y_test = train_test_split(X, 
                                                            Y, 
                                                            test_size=trPer, 
                                                            random_state=42, 
                                                            stratify=Y)
        
        return x_train, x_test, y_train, y_test
    
    elif distribution == "dirichlet":

        data_path = f"datasets/{dataset_name}/distributions/nclients_{numClients}/alpha_{alpha}/cliente_{clientID}.pkl"

        with open(data_path ,"rb") as reader:
            
            data = load(reader)

        return data


def load_data_server_list(dataset_name:str,      # dataset used on the system
                          numClients,            # number of clients
                          alpha=0.5,             # dirichlet distribution parameter
                          percentage=1.0):       # percentage of data used on the knowledge distillation     
    
    x_train = np.array([])
    y_train = np.array([])
    
    for clientID in range(numClients):
        
        data_path = f"datasets/{dataset_name}/distributions/nclients_{numClients}/alpha_{alpha}/cliente_{clientID}.pkl"

        with open(data_path ,"rb") as reader:
            
            data = load(reader)
        
        if len(x_train):
            
            x_train = np.append(x_train, 
                                data[0], 
                                axis=0)
            
            y_train = np.append(y_train, 
                                data[2], 
                                axis=0)
            
        else:
            
            x_train = data[0]
            y_train = data[2]
            
            x_test  = data[1]
            y_test  = data[3]

    if percentage < 1:

        rng = np.random.default_rng(seed=42)
        indexes = rng.choice(len(x_train),
                             int(percentage*len(x_train)),
                             replace=False)   
    
        x_train = x_train[indexes] 
        y_train = y_train[indexes]

    return x_train, x_test, y_train, y_test

def load_data_server_dataset(dataset_name:str,      # dataset used on the system
                             numClients:int,        # number of clients
                             alpha:float=0.5,       # dirichlet distribution parameter
                             percentage=1.0):       # percentage of data used on the knowledge distillation     
   
    datasets = []

    for clientID in range(numClients):
        
        data_path = f"datasets/{dataset_name}/distributions/nclients_{numClients}/alpha_{alpha}/cliente_{clientID}.pkl"

        with open(data_path ,"rb") as reader:
            
            data = load(reader)
        
        datasets.append(data[0])

    train = torch.utils.data.ConcatDataset(datasets)
    
    test = data[1]

    return train, test

def load_centralized_data(dataset_name):

    if dataset_name == "CIFAR-10":
        
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                   (0.2023, 0.1994, 0.2010)),
            ])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                  (0.2023, 0.1994, 0.2010)),
            ])

        train_data = datasets.CIFAR10(root=f'datasets/{dataset_name}', 
                                      train=True, 
                                      download=True, 
                                      transform=transform_train)
        
        test_data = datasets.CIFAR10(root=f'datasets/{dataset_name}', 
                                     train=False, 
                                     download=True, 
                                     transform=transform_test)

    else:

        raise ValueError('Dataset non implemented')

    return train_data, test_data

def load_centralized_data_from_fl_dataset(dataset_name:str,      # dataset used on the system
                                          numClients:int,        # number of clients
                                          alpha:float=0.5):      # dirichlet distribution parameter
   

    datasets = []

    for clientID in range(numClients):
        
        data_path = f"datasets/{dataset_name}/distributions/nclients_{numClients}/alpha_{alpha}/cliente_{clientID}.pkl"

        with open(data_path ,"rb") as reader:
            
            data = load(reader)
        
        datasets.append(data[0])

    train = torch.utils.data.ConcatDataset(datasets)
    
    test = data[1]

    return train, test


def load_centralized_data_from_fl_list(dataset_name,          # dataset used on the system
                                       numClients,            # number of clients
                                       alpha=5.0):            # dirichlet distribution parameter
    
    x_train = np.array([])
    y_train = np.array([])
    
    for clientID in range(numClients):  

        data_path = f"datasets/{dataset_name}/distributions/nclients_{numClients}/alpha_{alpha}/cliente_{clientID}.pkl"

        with open(data_path ,"rb") as reader:
            
            data = load(reader)
        
        if len(x_train):

            x_train = np.append(x_train, 
                                data[0], 
                                axis=0)
            
            y_train = np.append(y_train, 
                                data[2], 
                                axis=0)
            
        else:
            
            x_train = data[0]
            y_train = data[2]
            
            x_test  = data[1]
            y_test  = data[3]
    
    x_train = np.array(x_train,
                       dtype=np.float32)
        
    y_train = np.array(y_train,
                       dtype=np.float32)
        
    x_test = np.array(x_test,
                      dtype=np.float32)
        
    y_test = np.array(y_test,
                      dtype=np.float32)
    
    return x_train, x_test, y_train, y_test

