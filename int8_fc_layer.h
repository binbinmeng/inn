#ifndef INT8_FC_LAYER_H
#define INT8_FC_LAYER_H

#include "int8_layer.h"

class Int8FCLayer : public Int8Layer {
public:
  explicit Int8FCLayer(string name, int n, int ci, int hi, int wi, int co, int ho, int wo, bool bias)
      : Int8Layer(name, n, ci, hi, wi, co, ho, wo) {
    weight_count_ = in_channels_ * in_height_ * in_width_ * out_channels_ * out_height_ *  out_width_;
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

protected:

private:
  virtual void Forward();
  virtual void CreateCudnn();
  virtual void CreateCuda();
  virtual void FreeCudnn();
  virtual void FreeCuda();
  void SetCudnn();

  void shuffleChannels(int nn, int hh, int ww, int cc, int max_n);  // selectively do before gemm
  void scaleTopData();  // do after gemm

  cublasHandle_t handle_;

  int8_t* bottom_data_shuffled_;  // for shuffle channel
  int* top_data_int32_;  // for intermediate value of forward
  bool has_bias_;
};

#endif