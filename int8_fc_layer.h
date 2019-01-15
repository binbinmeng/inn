#ifndef INT8_FC_LAYER_H
#define INT8_FC_LAYER_H

#include "int8_layer.h"
#include "cuda.h"
#include "cudnn.h"
#include "cuda_runtime.h"
#include <cublas_v2.h>
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <iostream>
#include <assert.h>

class Int8FCLayer : public Int8Layer {
public:
  explicit Int8FCLayer(string name, int n, int ci, int hi, int wi, int co, int ho, int wo, bool bias)
      : Int8Layer(name, n, ci, hi, wi, co, ho, wo) {
    weight_count_ = in_channels_ * in_height_ * in_width_ * out_channels_ * out_height_ *  out_width_;  // override the base weight_count_
    bias_count_ = out_channels_ * out_height_ *  out_width_;
    has_bias_ = bias;
    CreateCudnn();
    CreateCuda();
  }

  virtual ~Int8FCLayer() {
    FreeCudnn();
    FreeCuda();
  }

  virtual void readWeightFromModel(const caffe::LayerParameter& layer_param, float weight_scale, float bias_scale);

  bool has_bias_;

protected:

private:
  virtual void Forward();
  virtual void CreateCudnn();
  virtual void CreateCuda();
  virtual void FreeCudnn();
  virtual void FreeCuda();
  void SetCudnn();

  void shuffleChannels(int8_t* data, int nn, int hh, int ww, int cc);
  void scaleTopData();

  cublasHandle_t handle_;

  int* top_data_int32_;
};

#endif