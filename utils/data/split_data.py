import os 
import sys

import numpy as np

from torch.utils.data  import Subset

from torchvision import transforms

from pickle import (
        dump, 
        load)

from utils.visualization.distribution import distribution_plot
from .sign_dataset import SignDataset

def generate_datasets(dataset_name:str="SIGN", 
                      alpha:float=5.0, 
                      n_clients:int=60) -> None:
    
    if dataset_name == "SIGN":

        transform = transforms.Compose([transforms.ToTensor()])

        # load dataset
        with open("datasets/traffic_signs/datasets/valentynsichkar/traffic-signs-preprocessed/versions/2/data1.pickle","rb") as reader:

            data = load(reader)

        # join data
        x_train = np.concatenate((data['x_train'], 
                                  data['x_validation']), 
                                  axis=0)

        y_train = np.concatenate((data['y_train'], 
                                  data['y_validation']), 
                                  axis=0)
        
        x_test = data['x_test']

        x_train = np.transpose(x_train, 
                               (0, 3, 2, 1))
        
        x_test = np.transpose(x_test, 
                              (0, 3, 2, 1))

        train_data = SignDataset(x_train,
                                 y_train,
                                 transform)
        
        test_data = SignDataset(x_test,
                                data['y_test'],
                                transform)

    else:

        raise ValueError("Dataset not found.")

    n_classes = len(train_data.classes)

    # group indexes by class
    class_indexes = [ [] 
                     for _ in 
                     range(n_classes) ]
    
    for idx, (_, label) in enumerate(train_data):
        
        class_indexes[label].append(idx)

    # apply dirichlet
    client_indexes = [ [] 
                      for _ in 
                      range(n_clients) ]

    for c in range(n_classes):

        class_indices = class_indexes[c]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(np.repeat(alpha, 
                                                    n_clients))

        # verify null distributions
        if proportions.sum() == 0 or np.isnan(proportions.sum()):

            proportions = np.ones(n_clients) / n_clients

        else:

            proportions = proportions / proportions.sum()

        split = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        split_indices = np.split(class_indices, split)

        for i, idxs in enumerate(split_indices):

            client_indexes[i].extend(idxs.tolist())

    # create path
    data_path = f"datasets/{dataset_name}/distributions/nclients_{n_clients}/alpha_{alpha}"
    os.makedirs(data_path, 
                exist_ok=True)
    
    with open(f"{data_path}/indexes", "wb") as writer:
            
        dump(client_indexes,
             writer)
        
    for cid, idxs in enumerate(client_indexes):
        
        client_train = Subset(train_data, idxs)
        
        with open(f"{data_path}/cliente_{cid}.pkl", "wb") as writer:
            
            dump([client_train,
                  test_data],
                  writer)
    
    distribution_plot(number_of_clients=n_clients,
                      number_of_classes=n_classes,
                      dataset_name=dataset_name,
                      dataset_object=train_data,
                      alpha=alpha)

if __name__ == "__main__":
    
    # parameters
    n_clients = int(sys.argv[1])
    dataset = sys.argv[2]
    alpha = float(sys.argv[3])

    generate_datasets(dataset_name=dataset,
                      alpha=alpha,
                      n_clients=n_clients)
