#ifndef INT8_CONV_LAYER_H
#define INT8_CONV_LAYER_H

#include "int8_layer.h"

class Int8ConvLayer : public Int8Layer {
public:
  explicit Int8ConvLayer(string name, int n, int ci, int co, int hi, int wi, int k, int s, int p, bool bias)
      : Int8Layer(name, n, ci - 1 - (ci - 1) % 4 + 4, hi, wi, co, hi - k + s + 2 * p, wi - k + s + 2 * p) {
    kernel_size_ = k;
    stride_ = s;
    pad_ = p;
    weight_count_ = in_channels_ * out_channels_ * kernel_size_ * kernel_size_;
    bias_count_ = out_channels_;
    workspace_size_ = 0;
    has_bias_ = bias;
    in_channels_origin_ = ci;
    CreateCudnn();
    CreateCuda();
  }

  virtual ~Int8ConvLayer() {
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

  int kernel_size_;
  int stride_;
  int pad_;

  // input and output channels must be multiples of 4
  // the original in channels are saved for readWeightFromModel()
  int in_channels_origin_;

  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t bottom_desc_, top_desc_, bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnConvolutionFwdAlgo_t conv_algo_;
  void* work_space_;
  size_t workspace_size_;

  bool has_bias_;
};

#endif