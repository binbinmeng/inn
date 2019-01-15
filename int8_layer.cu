#include "int8_layer.h"

__global__ void ScaleBeforeForward(int n, int8_t* data, float scale_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int value = ::round(float(data[index]) * scale_factor);
    data[index] = value > 127 ? 127 : (value < -127 ? -127 : value);
  }
}

void Int8Layer::convert(float scale) {  // scale
  int8_t* data = bottom_data_->data;
  float scale_factor = scale / bottom_data_->scale_factor;
  std::cout << "[Int8Layer::convert] scale of " << name() << ": " << scale << " " << bottom_data_->scale_factor << std::endl;
  ScaleBeforeForward<<<CAFFE_GET_BLOCKS(bottom_count_), CAFFE_CUDA_NUM_THREADS>>>(bottom_count_, data, scale_factor);
}