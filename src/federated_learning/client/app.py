import os
import numpy as np
from utils.torch.utils import get_args_client

# Get parameters
args = get_args_client()

# Set Parameters
                                                    # default parameters
client_id = args.client_id                          # 1
i_epochs = args.number_of_local_epochs              # 5
bs = args.batch_size                                # 32
ts = args.test_size                                 # 0.2
SERVER_IP = args.server_ip                          # [::]
SERVER_PORT = args.server_port                      # 8080
DATA_PATH = args.data_path                          # ../../datasets/VeReMi_Extension/mixalldata_clean.csv
DATASET_PATH = args.dataset_path                    # ../../datasets/VeReMi_Extension
DATASET = args.dataset                              # CIFAR-10         
MODEL_PATH = args.model_path                        # models/clients/flwr/
RESULT_PATH = args.result_path                      # results/clients/flwr/
LOG_PATH = args.log_path                            # logs/clients/flwr/
COMP_PATH = args.computation_time_path              # results/clients/flwr/computation_time
IMAGE_DATA = args.image_flag                        # 1
MODEL = args.model                                  # MOBILENET
num_clients = args.num_clients                      # 10
num_selected_clients = args.num_clients_fit         # 10
alpha = args.alpha                                  # 1
strategy = args.strategy                            # fedavg
scenario = args.scenario                            # all_in_one
sysd = args.sys_d                                   # 1

if scenario == "all_in_one":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{client_id%2}"

import torch
import flwr as fl

from architectures.torch.implementation import build_model
from utils.torch.load_federated_data import load_data_client

from utils.loader import load_config
from .client import FLClient

from utils.torch.utils import create_logger_client

cfg = load_config('configs/config.yaml')

system_models_ids = cfg[f'system_models_ids_{sysd}']

logger = create_logger_client(LOG_PATH+MODEL+'/', 
                              client_id)

message_length = 800 * 1024 * 1024

try:
    
    props = torch.cuda.get_device_properties(device=None)
    total_memory = props.total_memory
    client_memory = 1024 * 1024 * 512 # 256 MB for each client
    memory_percentage = client_memory/total_memory
    torch.cuda.set_per_process_memory_fraction(memory_percentage, 
                                               device=None)
    
    logger.debug(f"Total memory on the system: {total_memory}.")
    logger.debug(f"Total memory percentage: {memory_percentage}.")
    logger.debug(f"Execution path: {os.getcwd()}.")
    logger.debug(f"Training with model architecture {MODEL} and dataset {DATASET}.")
    logger.debug(f"GPU: {torch.cuda.current_device()}")

except:

    pass


logger.debug("Loading dataset")
train_dataset, test_dataset = load_data_client(dataset_name=DATASET, 
                                               clientID=client_id, 
                                               numClients=num_clients, 
                                               alpha=alpha,
                                               trPer=ts,
                                               distribution="dirichlet") 

trainloader = torch.utils.data.DataLoader(train_dataset, 
                                          batch_size=bs, 
                                          shuffle=True,
                                          num_workers=0,
                                          pin_memory=True)

testloader = torch.utils.data.DataLoader(test_dataset, 
                                         batch_size=bs, 
                                         shuffle=True,
                                         num_workers=0,
                                         pin_memory=True)

logger.debug("Building model")

labels = cfg['datasets'][DATASET]['classes']
features_shape = int(cfg['datasets'][DATASET]['features'][-1])

model, criterion, optimizer, device, scheduler = build_model(features_shape=features_shape,
                                                             labels_shape=labels,
                                                             model_name=MODEL,
                                                             lr=0.1)    


logger.debug("Starting training")
logger.debug(f'cid {client_id} with mid {system_models_ids[MODEL]} model {MODEL}')
fl.client.start_client(server_address=f'{SERVER_IP}:{SERVER_PORT}', 
                       client=FLClient(cid=client_id,
                                       mid=system_models_ids[MODEL],
                                       model=model,
                                       i_epochs=i_epochs,
                                       model_name=MODEL,
                                       batch_size=bs,
                                       dataset=DATASET,
                                       strategy=strategy,
                                       model_path=MODEL_PATH,
                                       result_path=RESULT_PATH,
                                       computation_time_path=COMP_PATH,
                                       logger=logger,
                                       optimizer=optimizer,
                                       criterion=criterion,
                                       scheduler=scheduler,
                                       trainloader=trainloader,
                                       testloader=testloader,
                                       device=device).to_client(),
                                       grpc_max_message_length=message_length)

logger.debug(f"GPU: {torch.cuda.current_device()}")
