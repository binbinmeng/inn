#include "int8_fc_layer.h"

#define m batch_size_
#define n top_count_
#define k bottom_count_

void Int8FCLayer::Forward() {
  // first, convert the bottom data into the caffe shape, from nhwc -> nchw
  int8_t* bottom_data_gemm = bottom_data_;
  if(name() == "ip1") {  // ugly, needed to be improved
    shuffleChannels(batch_size_, in_height_, in_width_, in_channels_, bottom_count_);
    bottom_data_gemm = bottom_data_shuffled_;
  }
  // then, do matrix multiplication
  cublasGemmEx(
      handle_, CUBLAS_OP_T, CUBLAS_OP_N,
      n, m, k,
      &one_int_, 
      weight_data_, CUDA_R_8I, k,  // A
      bottom_data_gemm, CUDA_R_8I, k,  // B
      &zero_int_,
      top_data_int32_, CUDA_R_32I, n,  // C
      CUDA_R_32I, CUBLAS_GEMM_DEFAULT  // default algorithm
  );

  // finally, scale top data from int32 to int8
  scaleTopData();

  // ignore bias since quantized bias are all zeros

  // if (name() == "ip2") {
  //   vector<int> ttt(10);
  //   checkCudaErrors(cudaMemcpy(&ttt[0], top_data_int32_, sizeof(int) * 10, cudaMemcpyDefault));
  //   for (int i = 0; i < 10; ++i) {
  //     cout << ttt[i] << " ";
  //   } cout << endl;
  //   for (int i = 0; i < 10; ++i) {
  //     cout << float(ttt[i]) / 516.3349614503815 / 26.302763503259253 << " ";
  //   } cout << endl;
  // }
}

void Int8FCLayer::CreateCudnn() {  // cudnn not supported in fully connected layer, so we actually use cublas
  checkCUBLAS(cublasCreate(&handle_));
}

void Int8FCLayer::FreeCudnn() {
  checkCUBLAS(cublasDestroy(handle_));
}

void Int8FCLayer::CreateCuda() {
  checkCudaErrors(cudaMalloc(&top_data_, sizeof(int8_t) * top_count_));
  checkCudaErrors(cudaMalloc(&top_data_int32_, sizeof(int) * top_count_));
  checkCudaErrors(cudaMalloc(&weight_data_, sizeof(int8_t) * weight_count_));
  checkCudaErrors(cudaMalloc(&bottom_data_shuffled_, sizeof(int8_t) * bottom_count_));
  if (has_bias_) checkCudaErrors(cudaMalloc(&bias_data_, sizeof(int8_t) * bias_count_));
}

void Int8FCLayer::FreeCuda() {
  checkCudaErrors(cudaFree(top_data_));
  checkCudaErrors(cudaFree(top_data_int32_));
  checkCudaErrors(cudaFree(weight_data_));
  checkCudaErrors(cudaFree(bottom_data_shuffled_));
  if (has_bias_) checkCudaErrors(cudaFree(bias_data_));
}

void Int8FCLayer::SetCudnn() {}

void Int8FCLayer::readWeightFromModel(const caffe::LayerParameter& layer_param, float weight_scale, float bias_scale) {
  const float *weight = layer_param.blobs(0).data().data();
  vector<int8_t> weight_data(weight_count_);
  // [input channels x input height x input width] x [output width]
  int mm, nn;
  mm = bottom_count_;
  nn = top_count_;
  LOG(INFO) << "[Int8FCLayer::readWeightFromModel] fc scaled weight: " << name() << " " << mm << " " << nn << " weight scale: " << weight_scale;
  // magic
  for (int m = 0; m < mm; ++m) {
    for (int n = 0; n < nn; ++n) {
      int scaled_weight = std::round(weight[m * nn + n] * weight_scale);
      weight_data[m * nn + n] = scaled_weight > 127 ? 127 : (scaled_weight < -127 ? -127 : scaled_weight);
      // if (name() == "ip2") cout << weight[m * nn + n] << " ";
    }
  }
  setWeight(weight_data);
  
  if (bias_count_ > 0 && layer_param.blobs_size() > 1) {
    LOG(INFO) << "\n[Int8FCLayer::readWeightFromModel] fc scaled bias " << name() << " bias scale: " << bias_scale;
    const float *bias = layer_param.blobs(1).data().data();
    vector<int8_t> bias_data(bias_count_);
    for (int k = 0; k < bias_count_; ++k) {
      int scaled_bias = std::round(bias[k] * bias_scale_);
      bias_data[k] = scaled_bias > 127 ? 127 : (scaled_bias < -127 ? -127 : scaled_bias);
      // cout << scaled_bias << " ";
    }
    setBias(bias_data);
  }
}