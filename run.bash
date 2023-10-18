#!/bin/bash

# Run below step one by one in terminal

docker pull tensorflow/tensorflow:2.14.0-gpu-jupyter
docker run --gpus all -it -p 8888:8888 -v /home/beastan/Documents/blogs_code/peft:/tf tensorflow/tensorflow:2.14.0-gpu-jupyter

