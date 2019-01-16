#ifndef INT8_LAYER_H
#define INT8_LAYER_H

#include "cuda.h"
#include "cudnn.h"
#include "cuda_runtime.h"
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <iostream>
#include <assert.h>
#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include "caffe.pb.h"

#include "utils.h"


using namespace std;

// struct Int8Blob {
//   int8_t* data;
//   float scale_factor;

//   Int8Blob() {
//     data = NULL;
//     scale_factor = 1.0;
//   }
// };

class Int8Layer {
public:
  explicit Int8Layer(string name, int n, int ci, int hi, int wi,
      int co, int ho, int wo) : 
      name_(name), batch_size_(n), in_channels_(ci), in_height_(hi), in_width_(wi),
      out_channels_(co), out_height_(ho), out_width_(wo) {
    bottom_count_ = batch_size_ * in_channels_ * in_height_ * in_width_;
    top_count_ = batch_size_ * out_channels_ * out_height_ * out_width_;
    weight_count_ = 0;
    bias_count_ = 0;
    // top_data_ = new Int8Blob();
    // weight_data_ = new Int8Blob();
    // bias_data_ = new Int8Blob();
    alpha_ = beta_ = 0.0;  // for convolution forward
    one_ = 1.0;
    // in_channels_origin_ = ci;
  }
    
  virtual ~Int8Layer() {
    // delete top_data_;
    // delete weight_data_;
    // delete bias_data_;
  }

  void setWeight(const vector<int8_t>& weight) {
    assert(weight_count_ > 0);
    assert(weight.size() > 0);  // if not enough, pad 0 at the end 
    checkCudaErrors(cudaMemcpyAsync(
        weight_data_, &weight[0],
        sizeof(int8_t) * min(weight_count_, int(weight.size())), cudaMemcpyHostToDevice));
  }

  void setBias(const vector<int8_t>& bias) {
    assert(bias_count_ > 0);
    assert(bias.size() > 0);  // if not enough, pad 0 at the end 
    checkCudaErrors(cudaMemcpyAsync(
        bias_data_, &bias[0],
        sizeof(int8_t) * min(bias_count_, int(bias.size())), cudaMemcpyHostToDevice));
  }

  void feed(const vector<int8_t>& bottom) {
    assert(bottom_count_ > 0);
    assert(bottom.size() > 0);  // if not enough, pad 0 at the end 
    checkCudaErrors(cudaMemcpyAsync(
        bottom_data_, &bottom[0],
        sizeof(int8_t) * min(bottom_count_, int(bottom.size())), cudaMemcpyHostToDevice));
  }

  void get(vector<int8_t>& top) {
    assert(top_count_ > 0);
    assert(top.size() > 0);
    checkCudaErrors(cudaMemcpyAsync(
        &top[0], top_data_,
        sizeof(int8_t) * min(top_count_, int(top.size())), cudaMemcpyDeviceToHost));
  }

  void get(vector<float>& top) {
    assert(top_count_ > 0);
    assert(top.size() > 0);
    int minn = min(top_count_, int(top.size()));
    vector<int8_t> tmp(minn);
    checkCudaErrors(cudaMemcpyAsync(
        &tmp[0], top_data_,
        sizeof(int8_t) * minn, cudaMemcpyDeviceToHost));

    float scale = bias_scale_;
    // float scale = top_data_->scale_factor;
    cout << "[get] scale for output layer: " << scale << endl;

    for (int i = 0; i < minn; ++i) {
      top[i] = float(tmp[i]) / scale;
    }
  }

  void setAlphaAndBeta(float alpha, float beta) {
    alpha_ = alpha;
    beta_ = beta;
  }

  void setBiasScale(float scale) {
    bias_scale_ = scale;
  }

  void forward() {
    Forward();
  }

  void setBottomData(int8_t* top_ptr) {
    bottom_data_ = top_ptr;
  }

  int8_t* getTopData() {
    return top_data_;
  }

  int bottom_count() {
    return bottom_count_;
  }

  int top_count() {
    return top_count_;
  }

  string name() {
    return name_;
  }

  // void setTopScale(float scale) {
  //   top_data_->scale_factor = scale;
  // }

  // float getBottomScale() {
  //   return bottom_data_->scale_factor;
  // }

  virtual void readWeightFromModel(const caffe::LayerParameter& layer_param, float weight_scale, float bias_scale) {

  }

  int batch_size_;
  int in_channels_, in_height_, in_width_;
  int out_channels_, out_height_, out_width_;

  // input and output channels must be multiples of 4
  // the origin channels are saved for reading model
  int in_channels_origin_;

  int bottom_count_;
  int top_count_;
  int weight_count_;
  int bias_count_;
  string name_;

  // for forward convolution
  float alpha_;
  float beta_;
  float one_;
  float bias_scale_;  // the scale of the next layer

  // int8_t* bottom_data_;
  // int8_t* top_data_;
  // int8_t* weight_data_;
  int8_t* bottom_data_;
  int8_t* top_data_;
  int8_t* weight_data_;
  int8_t* bias_data_;



protected:
  virtual void Forward() = 0;
  virtual void CreateCudnn() = 0;
  virtual void CreateCuda() = 0;
  virtual void FreeCudnn() = 0;
  virtual void FreeCuda() = 0;

  // for alpha and beta in cudnn and cublas functions
  // initialized in int8_layer.cpp
  static float one_float_;
  static float zero_float_;
  static int one_int_;
  static int zero_int_;

private:

};

#endif
