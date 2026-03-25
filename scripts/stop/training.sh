#!/bin/bash

server_processes=$(ps aux | grep server | grep torch | awk '{print $2}')

for process in $server_processes;
do
	kill -9 $process
done

client_processes=$(ps aux | grep client | grep torch | awk '{print $2}')

for process in $client_processes;
do
	kill -9 $process
done

