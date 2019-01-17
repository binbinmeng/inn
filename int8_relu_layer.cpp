#include "int8_relu_layer.h"

void Int8ReluLayer::Forward() {
  cudnnActivationForward(this->handle_,
      activ_desc_,
      &one_float_,
      this->bottom_desc_, bottom_data_,
      &zero_float_,
      this->top_desc_, top_data_);
}

void Int8ReluLayer::CreateCudnn() {
  checkCUDNN(cudnnCreate(&handle_));
  checkCUDNN(cudnnCreateTensorDescriptor(&bottom_desc_));  // bottom
  checkCUDNN(cudnnCreateTensorDescriptor(&top_desc_));  // top
  checkCUDNN(cudnnCreateActivationDescriptor(&activ_desc_));
  SetCudnn();
}

void Int8ReluLayer::FreeCudnn() {
  checkCUDNN(cudnnDestroy(handle_));
  checkCUDNN(cudnnDestroyTensorDescriptor(bottom_desc_));
  checkCUDNN(cudnnDestroyTensorDescriptor(top_desc_));
  checkCUDNN(cudnnDestroyActivationDescriptor(activ_desc_));
}

void Int8ReluLayer::CreateCuda() {
  checkCudaErrors(cudaMalloc(&top_data_, sizeof(int8_t) * top_count_));
}

void Int8ReluLayer::FreeCuda() {
  checkCudaErrors(cudaFree(top_data_));
}

void Int8ReluLayer::SetCudnn() {
  checkCUDNN(cudnnSetTensor4dDescriptor(
      bottom_desc_, CUDNN_TENSOR_NHWC, CUDNN_DATA_INT8,
      batch_size_, in_channels_, in_height_, in_width_));
  checkCUDNN(cudnnSetTensor4dDescriptor(
      top_desc_, CUDNN_TENSOR_NHWC, CUDNN_DATA_INT8,
      batch_size_, out_channels_, out_height_, out_width_));
  checkCUDNN(cudnnSetActivationDescriptor(
      activ_desc_, CUDNN_ACTIVATION_RELU,
      CUDNN_PROPAGATE_NAN, double(0)));
}