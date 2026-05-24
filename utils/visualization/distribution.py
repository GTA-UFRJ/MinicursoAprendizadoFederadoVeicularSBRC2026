import os

import matplotlib.pyplot as plt
import numpy as np

from pickle import load
from torch.utils.data import Dataset

def distribution_plot(number_of_clients:int=50,
                      number_of_classes:int=43,
                      dataset_name:str="SIGN",
                      dataset_object:Dataset=None,
                      alpha:float=5.0) -> None:

    # verify if we implement the dataset
    if dataset_name != "SIGN":
     
        ValueError("Dataset not found.")

    all_labels = np.array(dataset_object.targets) 
    
    # indexes
    clients_file = f"datasets/{dataset_name}/distributions/nclients_{number_of_clients}/alpha_{alpha}/indexes"

    with open(clients_file,"rb") as reader:

        data = load(reader)

    counts_matrix = np.zeros((number_of_classes, number_of_clients))

    for client_id, indices in enumerate(data):
        
        client_classes = all_labels[indices]
        
        for cls in range(number_of_classes):

            counts_matrix[cls, client_id] = np.sum(client_classes == cls)

    x_coords, y_coords = np.mgrid[0:number_of_classes, 0:number_of_clients]

    plt.figure(figsize=(18, 7))

    scatter = plt.scatter(
        x_coords.flatten(), 
        y_coords.flatten(), 
        s=counts_matrix.flatten() * 0.6, 
        c=counts_matrix.flatten(), 
        cmap='Blues', 
        alpha=0.7, 
        edgecolors="navy", 
        linewidth=0.5
    )

    plt.colorbar(scatter, label='Número de Amostras')
    plt.xticks(range(number_of_classes), [f'{i}' for i in range(number_of_classes)])
    plt.yticks(range(number_of_clients), fontsize=14, rotation=90)
    plt.gca().invert_yaxis()

    plt.xlabel('ID da Classe')
    plt.ylabel('ID do Cliente')
    plt.grid(True, linestyle=':', alpha=0.3)

    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # verify path
    figure_name = f"figures/data/distributions/{dataset_name}_nclients_{number_of_clients}_alpha_{alpha}"
    os.makedirs(os.path.dirname(figure_name),
                exist_ok=True)

    
    plt.savefig(f"{figure_name}.png", 
                dpi=300,
                bbox_inches='tight')
    
    plt.plot()

if __name__ == "__main__":

    distribution_plot()
