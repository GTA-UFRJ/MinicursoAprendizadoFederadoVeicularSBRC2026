import os
import numpy as np
from utils.torch.utils import get_args_client

import torch
import flwr as fl

from architectures.torch.implementation import build_model
from utils.torch.load_federated_data import load_data_client

from utils.loader import load_config
from .client import FLClient

from utils.torch.utils import create_logger_client

cfg = load_config('configs/config.yaml')

client_id_1 = 1
i_epochs = cfg['simulation']['federated_learning']['client']['local_epochs']
bs = 32
ts = 0.2
SERVER_IP = cfg['simulation']['federated_learning']['server']['ip']
SERVER_PORT = cfg['simulation']['federated_learning']['server']['port']
DATASET = cfg['simulation']['data']['name']
LOG_PATH = 'logs/clients/flwr/'
IMAGE_DATA = 1
MODEL = cfg['simulation']['model']['name']
num_clients = cfg['simulation']['cars']
num_selected_clients = cfg['simulation']['federated_learning']['server']['n_clients_selected']
alpha = cfg['simulation']['data']['alpha']
strategy = cfg['simulation']['federated_learning']['server']['strategy']


logger = create_logger_client(LOG_PATH+MODEL+'/', 
                              client_id_1)

message_length = 800 * 1024 * 1024

logger.debug("Loading dataset")
train_dataset_1, test_dataset_1 = load_data_client(dataset_name=DATASET, 
                                               clientID=client_id_1, 
                                               numClients=num_clients, 
                                               alpha=alpha,
                                               trPer=ts,
                                               distribution="dirichlet") 

trainloader_1 = torch.utils.data.DataLoader(train_dataset_1, 
                                            batch_size=bs, 
                                            shuffle=True,
                                            num_workers=0,
                                            pin_memory=True)

testloader_1 = torch.utils.data.DataLoader(test_dataset_1, 
                                           batch_size=bs, 
                                           shuffle=True,
                                           num_workers=0,
                                           pin_memory=True)

logger.debug("Building model")

labels = cfg['simulation']['data']['n_classes']
features_shape = int(cfg['simulation']['data']['shape'][-1])

model_1, criterion_1, optimizer_1, device_1, scheduler_1 = build_model(features_shape=features_shape,
                                                                       labels_shape=labels,
                                                                       model_name=MODEL,
                                                                       lr=0.1)    


fl.client.start_client(server_address=f'{SERVER_IP}:{SERVER_PORT}', 
                       client=FLClient(cid=client_id_1,                                    
                                       model=model_1,
                                       i_epochs=i_epochs,
                                       model_name=MODEL,
                                       batch_size=bs,
                                       dataset=DATASET,
                                       strategy=strategy,
                                       logger=logger,
                                       optimizer=optimizer_1,
                                       criterion=criterion_1,
                                       scheduler=scheduler_1,
                                       trainloader=trainloader_1,
                                       testloader=testloader_1,
                                       device=device_1).to_client(),
                                       grpc_max_message_length=message_length)
