#include "int8_pooling_layer.h"

void Int8PoolingLayer::Forward() {
  cudnnPoolingForward(handle_, pooling_desc_,
      &one_float_,
      bottom_desc_, bottom_data_,
      &zero_float_,
      top_desc_, top_data_);
}

void Int8PoolingLayer::CreateCudnn() {
  checkCUDNN(cudnnCreate(&handle_));
  checkCUDNN(cudnnCreateTensorDescriptor(&bottom_desc_));  // bottom
  checkCUDNN(cudnnCreateTensorDescriptor(&top_desc_));  // top
  checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_desc_));  // pool
  SetCudnn();
}

void Int8PoolingLayer::FreeCudnn() {
  checkCUDNN(cudnnDestroyTensorDescriptor(bottom_desc_));
  checkCUDNN(cudnnDestroyTensorDescriptor(top_desc_));
  checkCUDNN(cudnnDestroyPoolingDescriptor(pooling_desc_));
  checkCUDNN(cudnnDestroy(handle_));
}

void Int8PoolingLayer::CreateCuda() {
  checkCudaErrors(cudaMalloc(&top_data_, sizeof(int8_t) * top_count_));
}

void Int8PoolingLayer::FreeCuda() {
  checkCudaErrors(cudaFree(top_data_));
}

void Int8PoolingLayer::SetCudnn() {
  if (method_ == "MAX") {
    mode_ = CUDNN_POOLING_MAX;
  } else if (method_ == "AVE") {
    mode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  } else {
    LOG(ERROR) << "unidentified pooling type";
  }
  checkCUDNN(cudnnSetTensor4dDescriptor(
      bottom_desc_, CUDNN_TENSOR_NHWC, CUDNN_DATA_INT8,
      batch_size_, in_channels_, in_height_, in_width_));
  checkCUDNN(cudnnSetTensor4dDescriptor(
      top_desc_, CUDNN_TENSOR_NHWC, CUDNN_DATA_INT8,
      batch_size_, out_channels_, out_height_, out_width_));
  checkCUDNN(cudnnSetPooling2dDescriptor(
      pooling_desc_, mode_, CUDNN_PROPAGATE_NAN, 
      2, 2, 0, 0, 2, 2));  // k, k, p, p, s, s
  // H_ = (H - pool_height) / stride + padding
  // int n, c, h, w;
  // checkCUDNN(cudnnGetPooling2dForwardOutputDim(
  //     pooling_desc_,
  //     bottom_desc_,
  //     &n, &c, &h, &w));
  // cout << n << " " << c << " " << h << " " << w << "\n";
}