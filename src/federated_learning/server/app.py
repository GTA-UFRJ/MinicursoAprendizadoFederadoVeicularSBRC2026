# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
#             INRIA
#
# University: Universidade Federal do Rio de Janeiro (UFRJ) - Brazil  
#             Ecole Polytechnique - France
#

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import flwr as fl

from .strategy.fedavg import FedAvg
from utils.torch.utils import get_args_server
from utils.torch.utils import create_logger_server
from architectures.torch.implementation import build_model
from flwr.common import ndarrays_to_parameters

from architectures.torch.implementation import get_weights
from utils.loader import load_config

cfg = load_config('configs/config.yaml')

num_rounds = cfg['simulation']['federated_learning']['server']['rounds']               
server_ip = cfg['simulation']['federated_learning']['server']['ip']
server_port = cfg['simulation']['federated_learning']['server']['port']
num_clients_fit = cfg['simulation']['cars']
num_clients = cfg['simulation']['cars']
aggregation = cfg['simulation']['federated_learning']['server']['strategy']
server_log_path = cfg['simulation']['federated_learning']['server']['log_path']
server_models_path = cfg['simulation']['federated_learning']['server']['model_path']
time_path = cfg['simulation']['federated_learning']['server']['time_path']                
DATASET = cfg['simulation']['data']['name']
alpha = cfg['simulation']['data']['alpha']
MODEL = cfg['simulation']['model']['name']

os.makedirs(server_log_path, 
            exist_ok=True)

message_length = 800 * 1024 * 1024

props = torch.cuda.get_device_properties(device=None)
total_memory = props.total_memory
client_memory = 1024 * 1024 * 1024 # 1 GB for the server
memory_percentage = client_memory/total_memory
torch.cuda.set_per_process_memory_fraction(memory_percentage, 
                                           device=None)


logger = create_logger_server(log_path=server_log_path+aggregation)

logger.debug(f"Execution path: {os.getcwd()}.")

logger.debug(f"starting training with aggregation strategy {aggregation}, {num_clients} available clients, selecting {num_clients_fit} to fit, during {num_rounds} global epochs")

message_length = 800 * 1024 * 1024

# Initialize model parameters
n_classes = cfg['datasets'][DATASET]['classes']
features_shape = int(cfg['datasets'][DATASET]['features'][-1])

model, _, _, _, _ = build_model(features_shape=features_shape, 
                                labels_shape=n_classes,
                                model_name=MODEL,
                                lr=0.1)

ndarrays = get_weights(model)

parameters = ndarrays_to_parameters(ndarrays)


strategy = FedAvg(min_available_clients=num_clients,
                    min_fit_clients=num_clients_fit,
                    min_evaluate_clients=num_clients,
                    fraction_fit=0.01,
                    fraction_evaluate=0.01,
                    logger=logger,
                    initial_parameters=parameters,
                    time_path=time_path)

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=num_rounds),
                       server_address=server_ip+":"+server_port,
                       strategy=strategy,
                       grpc_max_message_length=message_length)



