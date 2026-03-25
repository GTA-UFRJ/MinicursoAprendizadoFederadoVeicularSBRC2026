#!/bin/bash

strategie=$(yq '.simulation.federated_learning.server.strategie' "configs/config.yaml")
rnd=$(yq '.simulation.federated_learning.server.rounds' "configs/config.yaml")
dataset=$(yq '.simulation.data.name' "configs/config.yaml")
alpha_dirichlet=$(yq '.simulation.data.alpha' "configs/config.yaml")
ip=$(yq '.simulation.federated_learning.server.ip' "configs/config.yaml")
port=$(yq '.simulation.federated_learning.server.port' "configs/config.yaml")
cars=$(yq '.simulation.cars' "configs/config.yaml")
model=$(yq '.simulation.model.name' "configs/config.yaml")

source scripts/run/baremetal.sh "$strategie" "$ip" "$port" "$alpha_dirichlet" "$model" "$cars" "$dataset" "$exp" "$rnd"
