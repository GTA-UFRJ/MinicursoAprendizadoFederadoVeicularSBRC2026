# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
#             INRIA
#
# University: Universidade Federal do Rio de Janeiro (UFRJ) - Brazil  
#             Ecole Polytechnique - France
#
from pickle import load

def load_data_client(dataset_name='SIGN',       # dataset used on the system
                     clientID=1,                # client identifier
                     numClients=1,              # number of clients
                     distribution="dirichlet",  # manual or dirichlet
                     alpha=0.5):                # dirichlet distribution parameter

    if distribution == "dirichlet":

        data_path = f"datasets/{dataset_name}/distributions/nclients_{numClients}/alpha_{alpha}/cliente_{clientID}.pkl"

        with open(data_path ,"rb") as reader:
            
            data = load(reader)

        return data