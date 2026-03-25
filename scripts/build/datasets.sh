#!/bin/bash

if [ ! -d datasets/traffic_signs ]; then
	
	mkdir -p datasets/traffic_signs
	export KAGGLEHUB_CACHE="datasets/traffic_signs"
	python utils/data/get_signs_dataset.py

fi
