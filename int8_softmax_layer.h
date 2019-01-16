#ifndef INT8_SOFTMAX_LAYER_H
#define INT8_SOFTMAX_LAYER_H

#include "int8_layer.h"

class Int8SoftmaxLayer : public Int8Layer {
public:
  explicit Int8SoftmaxLayer(string name, int n, int c, int h, int w)
      : Int8Layer(name, n, c, h, w, c, h, w) {
    CreateCudnn();
    CreateCuda();
  }

  virtual ~Int8SoftmaxLayer() {
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
};

void Int8SoftmaxLayer::Forward() {

  // incomplete, need to convert to fp32

  checkCUDNN(cudnnSoftmaxForward(handle_, CUDNN_SOFTMAX_ACCURATE,  // CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_LOG
        CUDNN_SOFTMAX_MODE_CHANNEL,  // CUDNN_SOFTMAX_MODE_INSTANCE
        &one_float_,
        bottom_desc_, bottom_data_,
        &zero_float_,
        top_desc_, top_data_));
}

void Int8SoftmaxLayer::CreateCudnn() {
  checkCUDNN(cudnnCreate(&handle_));
  checkCUDNN(cudnnCreateTensorDescriptor(&bottom_desc_));  // bottom
  checkCUDNN(cudnnCreateTensorDescriptor(&top_desc_));  // top
  SetCudnn();
}

void Int8SoftmaxLayer::FreeCudnn() {
  checkCUDNN(cudnnDestroy(handle_));
  checkCUDNN(cudnnDestroyTensorDescriptor(bottom_desc_));
  checkCUDNN(cudnnDestroyTensorDescriptor(top_desc_));
}

void Int8SoftmaxLayer::CreateCuda() {
  checkCudaErrors(cudaMalloc(&top_data_, sizeof(int8_t) * top_count_));
}

void Int8SoftmaxLayer::FreeCuda() {
  checkCudaErrors(cudaFree(top_data_));
}

void Int8SoftmaxLayer::SetCudnn() {
  checkCUDNN(cudnnSetTensor4dDescriptor(
      bottom_desc_, CUDNN_TENSOR_NHWC, CUDNN_DATA_INT8,
      batch_size_, in_channels_, in_height_, in_width_));
  checkCUDNN(cudnnSetTensor4dDescriptor(
      top_desc_, CUDNN_TENSOR_NHWC, CUDNN_DATA_INT8,
      batch_size_, out_channels_, out_height_, out_width_));
}

#endif