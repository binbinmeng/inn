#!/bin/bash

set -e

echo nvcc
nvcc -o lenet main.cpp caffe.pb.a int8_conv_layer.cpp int8_convba_layer.cpp int8_data_layer.cpp int8_pooling_layer.cpp int8_relu_layer.cpp int8_layer.cpp int8_net.cpp int8_fc_layer.cpp int8_fc_layer.cu -lcudnn -lcublas -std=c++11 `pkg-config --cflags --libs opencv` -lprotobuf -lpthread -lglog -lgflags

echo testing
./lenet



# echo test
# nvcc -c int8_fc_layer.cu -lcudnn -lcublas -std=c++11 `pkg-config --cflags --libs opencv` -lprotobuf -lpthread