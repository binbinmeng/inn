#include "int8_fc_layer.h"

__global__ void shuffleFCLayer(int8_t* data, int8_t* data_bak, int nn, int hh, int ww, int cc, int max_n) {
  CUDA_KERNEL_LOOP(index, max_n) {
    int n = index / (hh * ww * cc);
    int h = index % (hh * ww * cc) / (ww * cc);
    int w = index % (ww * cc) / cc;
    int c = index % cc;
    int nhwc = n * hh * ww * cc + h * ww * cc + w * cc + c;
    int nchw = n * cc * hh * ww + c * hh * ww + h * ww + w;
    // assert(index == nhwc);
    data[nchw] = data_bak[nhwc];
  }
}

__global__ void scaleFromFP32ToINT8(int* data_int32, int8_t* data_int8, float scale, int max_n) {
  CUDA_KERNEL_LOOP(index, max_n) {
    int data_scaled = data_int32[index] * scale;
    data_int8[index] = data_scaled > 127 ? 127 : (data_scaled < -127 ? -127 : data_scaled);
  }
}

void Int8FCLayer::shuffleChannels(int8_t* data, int nn, int hh, int ww, int cc) {
  int max_n = nn * hh * ww * cc;
  int8_t* data_bak;
  checkCudaErrors(cudaMalloc(&data_bak, sizeof(int8_t) * max_n));
  checkCudaErrors(cudaMemcpy(data_bak, data, sizeof(int8_t) * max_n, cudaMemcpyDefault));
  shuffleFCLayer<<<CAFFE_GET_BLOCKS(max_n), CAFFE_CUDA_NUM_THREADS>>>(data, data_bak, nn, hh, ww, cc, max_n);
  checkCudaErrors(cudaFree(data_bak));
}

void Int8FCLayer::scaleTopData() {
  scaleFromFP32ToINT8<<<CAFFE_GET_BLOCKS(top_count_), CAFFE_CUDA_NUM_THREADS>>>(top_data_int32_, top_data_, alpha_, top_count_);
}