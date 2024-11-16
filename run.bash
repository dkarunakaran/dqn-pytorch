#!/bin/bash
# Run below step one by one in terminal
sudo docker rmi -f $(sudo docker images -f "dangling=true" -q)
docker build -t dqn_pytorch .
docker run --net host --gpus all -it -v /home/$(whoami)/Documents/projects/dqn-pytorch:/app dqn_pytorch


