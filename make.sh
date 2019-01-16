#!/bin/bash

set -e

echo nvcc
nvcc -o test inn.cpp caffe.pb.a int8_layer.cpp int8_fc_layer.cpp int8_fc_layer.cu -lcudnn -lcublas -std=c++11 `pkg-config --cflags --libs opencv` -lprotobuf -lpthread

echo testing
./test



# echo test
# nvcc -c int8_fc_layer.cu -lcudnn -lcublas -std=c++11 `pkg-config --cflags --libs opencv` -lprotobuf -lpthread