#ifndef INT8_RELU_LAYER_H
#define INT8_RELU_LAYER_H

#include "int8_layer.h"

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

#endif