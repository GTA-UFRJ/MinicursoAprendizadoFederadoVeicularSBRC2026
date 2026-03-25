#!/bin/bash

if [ -z $1 ]; then
	
	server="fedavg"
	server_ip=0.0.0.0
	server_port=8082
	alpha=0.1
	model="RESNET18"
	numClients=2
	dataset="SIGN"
	exp=0
	rounds=10
else
	
	server=$1
	server_ip=$2
	server_port=$3
	alpha=$4
	model=$5
	numClients=$6
	dataset=$7
	exp=$8
	rounds=$9

fi

bs=64

server_log_path="logs/server/flwr/$server/$dataset/$alpha/$framework/$execution/$exp/$model/"
server_model_path="models/server/flwr/$server/$dataset/$alpha/$framework/$execution/$exp/$model/"
time_path_server="results/server/flwr/training/$server/$dataset/$alpha/$framework/$execution/$exp/$model/"

echo "Creating paths"
mkdir -p $server_log_path
mkdir -p $server_model_path 
mkdir -p "$time_path_server/sys_model"
mkdir -p "$time_path_server/eval"

[ ! -d "datasets/$dataset/distributions/nclients_$numClients/alpha_$alpha/"  ] && python -m utils.data.split_data $fixed_n_clients $dataset $alpha


if [ $server == "fedavg" ]; then
	
	if [ "$use_all_data" = 1 ]; then

		num_clients=$numClients

	else
		
		num_clients=$fixed_n_clients

	fi

	echo "Starting server fedavg"
	sleep 3
	python -m src.federated_learning.server.$framework.app -ds=$dataset -ncf=$numClientsFit -nc=$numClients -nor=$rounds -sn=$server -smp=$server_model_path -md=$model -slp=$server_log_path -sp=$server_port -tp=$time_path_server -a=$alpha -bk=$agg & 
		
	clients_result_path="results/clients/flwr/classification/$server/$dataset/$alpha/$framework/$execution/$exp/$model/"
	clients_log_path="logs/clients/flwr/$server/$dataset/$alpha/$framework/$execution/$exp/$model/"
	clients_model_path="models/clients/flwr/$server/$dataset/$alpha/$framework/$execution/$exp/$model/"
	time_path_client="results/clients/flwr/training/$server/$dataset/$alpha/$framework/$execution/$exp/$model/"

	mkdir -p $clients_result_path
	mkdir -p $clients_result_path/raw
	mkdir -p $clients_log_path
	mkdir -p $clients_model_path 
	mkdir -p $time_path_client
	
	clients_result_path="results/clients/flwr/classification/$server/$dataset/$alpha/$framework/$execution/$exp/"
	clients_log_path="logs/clients/flwr/$server/$dataset/$alpha/$framework/$execution/$exp/"
	clients_model_path="models/clients/flwr/$server/$dataset/$alpha/$framework/$execution/$exp/"
	time_path_client="results/clients/flwr/training/$server/$dataset/$alpha/$framework/$execution/$exp/"
		
	echo "Starting clients fedavg"
	sleep 10
	
	for i in $(seq 0 $(($numClients-1)))
	do
	
		echo "Waiting client "$i" initialization"
		python -m src.federated_learning.client.$framework.app -ds=$dataset -md=$model -nc=$num_clients -cid=$i -b=$bs -ncf=$numClientsFit -mp=$clients_model_path -lp=$clients_log_path -rp=$clients_result_path -ctp=$time_path_client -sp=$server_port -a=$alpha >> $clients_result_path"$model""/raw/client_""$i" &
		

	done

wait

echo "baremetal script finished"
