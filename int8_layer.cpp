#include "int8_layer.h"

void Int8Layer::setScales(float alpha, float bias_scale) {
  alpha_ = alpha;
  bias_scale_ = bias_scale;
}

void Int8Layer::forward() {
  Forward();
}

void Int8Layer::setBottomData(int8_t* top_ptr) {
  bottom_data_ = top_ptr;
}

int8_t* Int8Layer::getTopData() {
  return top_data_;
}

int Int8Layer::bottom_count() {
  return bottom_count_;
}

int Int8Layer::top_count() {
  return top_count_;
}

string Int8Layer::name() {
  return name_;
}

void Int8Layer::feed(const vector<int8_t>& bottom) {
  CHECK_EQ(bottom_count_, bottom.size()) << "data size does not match";
  checkCudaErrors(cudaMemcpyAsync(
      bottom_data_, &bottom[0],
      sizeof(int8_t) * min(bottom_count_, int(bottom.size())), cudaMemcpyHostToDevice));
}

void Int8Layer::get(vector<int8_t>& top) {
  CHECK_LE(top_count_, top.size()) << "top vector is smaller than top count";
  checkCudaErrors(cudaMemcpyAsync(
      &top[0], top_data_,
      sizeof(int8_t) * min(top_count_, int(top.size())), cudaMemcpyDeviceToHost));
}

void Int8Layer::get(vector<float>& top) {
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

void Int8Layer::setWeight(const vector<int8_t>& weight) {
  CHECK_EQ(weight_count_, weight.size()) << "weight size does not match";
  checkCudaErrors(cudaMemcpyAsync(
      weight_data_, &weight[0],
      sizeof(int8_t) * weight_count_, cudaMemcpyHostToDevice));
}

void Int8Layer::setBias(const vector<int8_t>& bias) {
  CHECK_EQ(bias_count_, bias.size()) << "bias size does not match";
  checkCudaErrors(cudaMemcpyAsync(
      bias_data_, &bias[0],
      sizeof(int8_t) * bias_count_, cudaMemcpyHostToDevice));
}

float Int8Layer::one_float_ = 1.0;
float Int8Layer::zero_float_ = 0.0;
int Int8Layer::one_int_ = 1;
int Int8Layer::zero_int_ = 0;