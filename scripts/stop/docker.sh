#!/bin/bash

if [[ "$(sudo docker ps -q)" != "" ]]; then
        sudo docker kill $(sudo docker ps -q)
fi

if [[ "$(sudo docker ps -q -a)" != "" ]]; then
        sudo docker rm --force $(sudo docker ps -a -q)
fi

sudo docker network prune -f
