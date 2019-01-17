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
  }

  virtual ~Int8Layer() {}

  void feed(const vector<int8_t>& bottom);
  void get(vector<int8_t>& top);
  void get(vector<float>& top);
  void setScales(float alpha, float bias_scale);
  void forward();
  void setBottomData(int8_t* top_ptr);
  int8_t* getTopData();
  int bottom_count();
  int top_count();
  string name();

  // not needed by default
  virtual void readWeightFromModel(const caffe::LayerParameter& layer_param, float weight_scale, float bias_scale) {}

protected:
  void setWeight(const vector<int8_t>& weight);
  void setBias(const vector<int8_t>& bias);

  // to be implemented in specific layers
  virtual void Forward() = 0;
  virtual void CreateCudnn() = 0;
  virtual void CreateCuda() = 0;
  virtual void FreeCudnn() = 0;
  virtual void FreeCuda() = 0;

  string name_;
  int batch_size_;
  int in_channels_, in_height_, in_width_;
  int out_channels_, out_height_, out_width_;

  int bottom_count_;
  int top_count_;
  int weight_count_;
  int bias_count_;

  float alpha_;  // restore scale after conv and fc
  float bias_scale_;  // the activation scale of the next layer

  int8_t* bottom_data_;
  int8_t* top_data_;
  int8_t* weight_data_;
  int8_t* bias_data_;

  // for alpha and beta in cudnn and cublas functions
  // initialized in int8_layer.cpp
  static float one_float_;
  static float zero_float_;
  static int one_int_;
  static int zero_int_;

private:

};

#endif