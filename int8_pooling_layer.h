#ifndef INT8_POOLING_LAYER_H
#define INT8_POOLING_LAYER_H

#include "int8_layer.h"

class Int8PoolingLayer : public Int8Layer {
public:
  explicit Int8PoolingLayer(string name, string method, int n, int c, int h, int w)
      : Int8Layer(name, n, c, h, w, c, h/2, w/2) {
    // only support kernel size 2 and stride 2 for now
    kernel_size_ = 2;
    stride_ = 2;
    method_ = method;
    CreateCudnn();
    CreateCuda();
  }

  virtual ~Int8PoolingLayer() {
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

  int kernel_size_;
  int stride_;
  string method_;

  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t bottom_desc_, top_desc_;
  cudnnPoolingDescriptor_t pooling_desc_;
  cudnnPoolingMode_t mode_;
};

#endif