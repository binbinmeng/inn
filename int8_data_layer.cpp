#include "int8_data_layer.h"

void Int8DataLayer::CreateCudnn() {}

void Int8DataLayer::CreateCuda() {
  checkCudaErrors(cudaMalloc(&top_data_, sizeof(int8_t) * top_count_));
  bottom_data_ = top_data_;
}

void Int8DataLayer::FreeCudnn() {}

void Int8DataLayer::FreeCuda() {
  checkCudaErrors(cudaFree(top_data_));
}

void Int8DataLayer::Forward() {}
