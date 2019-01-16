#ifndef INT8_DATA_LAYER_H
#define INT8_DATA_LAYER_H

#include "int8_layer.h"

class Int8DataLayer : public Int8Layer {
public:
	explicit Int8DataLayer(string name, int n, int c, int h, int w)
      : Int8Layer(name, n, c, h, w, c, h, w) {
    CreateCudnn();
    CreateCuda();
  }

  virtual ~Int8DataLayer() {
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
};

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

#endif
