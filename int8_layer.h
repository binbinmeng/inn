#ifndef INT8_LAYER_H
#define INT8_LAYER_H

#include "utils.h"

using namespace std;

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
    alpha_ = beta_ = 0.0;  // for convolution forward
  }

  virtual ~Int8Layer() {}

  void setWeight(const vector<int8_t>& weight) {
    CHECK_EQ(weight_count_, weight.size()) << "weight size does not match";
    checkCudaErrors(cudaMemcpyAsync(
        weight_data_, &weight[0],
        sizeof(int8_t) * weight_count_, cudaMemcpyHostToDevice));
  }

  void setBias(const vector<int8_t>& bias) {
    CHECK_EQ(bias_count_, bias.size()) << "bias size does not match";
    checkCudaErrors(cudaMemcpyAsync(
        bias_data_, &bias[0],
        sizeof(int8_t) * bias_count_, cudaMemcpyHostToDevice));
  }

  void feed(const vector<int8_t>& bottom) {
    CHECK_EQ(bottom_count_, bottom.size()) << "data size does not match";
    checkCudaErrors(cudaMemcpyAsync(
        bottom_data_, &bottom[0],
        sizeof(int8_t) * min(bottom_count_, int(bottom.size())), cudaMemcpyHostToDevice));
  }

  void get(vector<int8_t>& top) {
    CHECK_LE(top_count_, top.size()) << "top vector is smaller than top count";
    checkCudaErrors(cudaMemcpyAsync(
        &top[0], top_data_,
        sizeof(int8_t) * min(top_count_, int(top.size())), cudaMemcpyDeviceToHost));
  }

  void get(vector<float>& top) {
    CHECK_LE(top_count_, top.size()) << "top vector is smaller than top count";
    int minn = min(top_count_, int(top.size()));
    vector<int8_t> tmp(minn);
    checkCudaErrors(cudaMemcpyAsync(
        &tmp[0], top_data_,
        sizeof(int8_t) * minn, cudaMemcpyDeviceToHost));
    for (int i = 0; i < minn; ++i) {
      top[i] = float(tmp[i]) / bias_scale_;
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

  virtual void readWeightFromModel(const caffe::LayerParameter& layer_param, float weight_scale, float bias_scale) {}  // not needed by default

  int batch_size_;
  int in_channels_, in_height_, in_width_;
  int out_channels_, out_height_, out_width_;

  int bottom_count_;
  int top_count_;
  int weight_count_;
  int bias_count_;
  string name_;

  // for forward convolution
  float alpha_;  // restore scale after conv and fc
  float beta_;
  float bias_scale_;  // the activation scale of the next layer

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
