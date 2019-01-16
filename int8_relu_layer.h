#ifndef INT8_RELU_LAYER_H
#define INT8_RELU_LAYER_H

#include "int8_layer.h"
#include "cuda.h"
#include "cudnn.h"
#include "cuda_runtime.h"
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <iostream>
#include <assert.h>

class Int8ReluLayer : public Int8Layer {
public:
  explicit Int8ReluLayer(string name, int n, int c, int h, int w)
      : Int8Layer(name, n, c, h, w, c, h, w) {
    CreateCudnn();
    CreateCuda();
  }

  virtual ~Int8ReluLayer() {
    FreeCudnn();
    FreeCuda();
  }

protected:

private:
  virtual void Forward();
  virtual void CreateCudnn();
  virtual void CreateCuda();
  virtual void FreeCudnn();
  virtual void FreeCuda();
  void SetCudnn();

  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t bottom_desc_, top_desc_;
  cudnnActivationDescriptor_t activ_desc_;
};

void Int8ReluLayer::Forward() {
  checkCUDNN(cudnnActivationForward(this->handle_,
      activ_desc_,
      &one_float_,
      this->bottom_desc_, bottom_data_,
      &zero_float_,
      this->top_desc_, top_data_));
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


#endif