#!/bin/bash

if [ -z $1 ]; then
	
	server="fedavg"
	alpha=0.5
	model="MOBILENET"
	server_port=8080

else
	
	server=$1
	alpha=$2
	model=$3
	server_port=$4

fi

bs=64
eps=100
numClients=80
numClientsFit=20
dataset="CIFAR-10"
execution=$(date +%Y-%m-%d_%H-%M-%S)

server_log_path="logs/server/flwr/$server/$dataset/$alpha/$execution/$model/"
server_model_path="models/server/flwr/$server/$dataset/$alpha/$execution/$model/"
time_path_server="results/server/flwr/training/$server/$dataset/$alpha/$execution/$model/"

clients_result_path="results/clients/flwr/classification/$server/$dataset/$alpha/$execution/$model/"
clients_log_path="logs/clients/flwr/$server/$dataset/$alpha/$execution/$model/"
clients_model_path="models/clients/flwr/$server/$dataset/$alpha/$execution/$model/"
time_path_client="results/clients/flwr/training/$server/$dataset/$alpha/$execution/$model/"

data_path="datasets/$dataset/distributions/nclients_$numClients/alpha_$alpha/"

echo "Creating paths"
mkdir -p $clients_result_path
mkdir -p $clients_result_path/raw
mkdir -p $clients_log_path
mkdir -p $clients_model_path 
mkdir -p $server_log_path
mkdir -p $server_model_path 
mkdir -p $time_path_client
mkdir -p $time_path_server

[ ! -d "datasets/$dataset/distributions/nclients_$numClients/alpha_$alpha/"  ] && python src/data_division/split_data.py

echo "Creating network"
sudo docker network create federated_learning_network

echo "Initializing server"
sudo docker run -d -it --name="server" --network=federated_learning_network -p $server_port:$server_port server_keras bash

sleep 10 

echo "Creating paths on server container"

sudo docker exec -it server mkdir -p $time_path_server
sudo docker exec -it server mkdir -p $server_log_path
sudo docker exec -it server mkdir -p $server_model_path 

echo "Initializing clients"
for i in $(seq 0 $(($numClients - 1 )))
do
        echo "Initializing client ""$i"
        # 0_5 limitation
        #docker run -d -it --cpu-period="100000" --cpu-quota="150000" --name="client_""$i" --network=federated_learning_network client bash
        # 1_0 limitation
        #docker run -d -it --cpu-period="100000" --cpu-quota="150000" --name="client_""$i" --network=federated_learning_network client bash
        # no limitation
        sudo docker run -d -it --name="client_""$i" --network=federated_learning_network client_keras bash
done

echo "Starting training on server"
sudo docker exec -d -it "server" python app.py -ncf=$numClientsFit -nc=$numClients -nor=$eps -sn=$server -smp=$server_model_path -slp=$server_log_path -sp=$server_port -tp=$time_path_server 

sleep 10

for i in $(seq 0 $(($numClients - 1)))
do

	echo "Starting training on client ""$i"
	
	echo "Creating paths on client container"
	
	sudo docker exec -it "client_""$i" mkdir -p $clients_result_path
	sudo docker exec -it "client_""$i" mkdir -p $clients_result_path/raw
 	sudo docker exec -it "client_""$i" mkdir -p $clients_log_path
 	sudo docker exec -it "client_""$i" mkdir -p $clients_model_path 
 	sudo docker exec -it "client_""$i" mkdir -p $time_path_client
 	sudo docker exec -it "client_""$i" mkdir -p $data_path

	echo "Copying data to client"
	sudo docker cp "$data_path"cliente_$i.pkl client_$i:/app/code/$data_path

done
	
clients_result_path="results/clients/flwr/classification/$server/$dataset/$alpha/$execution/"
clients_log_path="logs/clients/flwr/$server/$dataset/$alpha/$execution/"
clients_model_path="models/clients/flwr/$server/$dataset/$alpha/$execution/"
time_path_client="results/clients/flwr/training/$server/$dataset/$alpha/$execution/"
	
for i in $(seq 0 $(($numClients - 1)))
do
	
	sudo docker exec -d -it "client_""$i" python app.py -sip server -md=$model -nc=$numClients -cid=$i -b=$bs -ncf=$numClientsFit -mp=$clients_model_path -lp=$clients_log_path -rp=$clients_result_path -ctp=$time_path_client -sp=$server_port -a=$alpha

done
