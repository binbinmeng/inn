#include "int8_fc_layer.h"

void Int8FCLayer::Forward() {
  // first, convert the bottom data into the caffe shape
  // nhwc -> nchw
  if(name() == "ip1") shuffleChannels(bottom_data_->data, batch_size_, in_height_, in_width_, in_channels_);
  // then, do matrix multiplication
  int alpha = 1;
  int beta = 0;
  int m = batch_size_;
  int n = top_count_;
  int k = bottom_count_;

  // if (name() == "ip2") {
  //   vector<int8_t> bbb(500);
  //   checkCudaErrors(cudaMemcpy(&bbb[0], bottom_data_->data, sizeof(int8_t) * 500, cudaMemcpyDefault));
  //   for (int i = 0; i < 500; ++i) cout << float(bbb[i]) / 26.302763503259253 << " ";
  //   cout << endl << endl;
  //   vector<int8_t> www(5000);
  //   checkCudaErrors(cudaMemcpy(&www[0], weight_data_->data, sizeof(int8_t) * 5000, cudaMemcpyDefault));
  //   // for (int i = 0; i < 5000; ++i) cout << float(www[i]) / 516.3349614503815 << " ";
  //   cout << endl << endl;
  // }

  checkCUBLAS(cublasGemmEx(
      handle_, CUBLAS_OP_T, CUBLAS_OP_N,
      n, m, k,
      &alpha, 
      weight_data_->data, CUDA_R_8I, k,  // A
      bottom_data_->data, CUDA_R_8I, k,  // B
      &beta,
      top_data_int32_, CUDA_R_32I, n,  // C
      // top_data_->data, CUDA_R_32I, n,  // C
      CUDA_R_32I, CUBLAS_GEMM_DEFAULT  // default algorithm
  ));

  // top_data_int32_ -> top_data_ (int8)
  scaleTopData();

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

  // ignore bias since quantized bias are all zeros
}



void Int8FCLayer::CreateCudnn() {  // cudnn not supported in fully connected layer, so we actually use cublas
  checkCUBLAS(cublasCreate(&handle_));
}

void Int8FCLayer::FreeCudnn() {
  checkCUBLAS(cublasDestroy(handle_));
}

void Int8FCLayer::CreateCuda() {
  checkCudaErrors(cudaMalloc(&top_data_->data, sizeof(int8_t) * top_count_));
  checkCudaErrors(cudaMalloc(&top_data_int32_, sizeof(int) * top_count_));
  checkCudaErrors(cudaMalloc(&weight_data_->data, sizeof(int8_t) * weight_count_));
  if (has_bias_) checkCudaErrors(cudaMalloc(&bias_data_->data, sizeof(int8_t) * bias_count_));
}

void Int8FCLayer::FreeCuda() {
  checkCudaErrors(cudaFree(top_data_->data));
  checkCudaErrors(cudaFree(top_data_int32_));
  checkCudaErrors(cudaFree(weight_data_->data));
  if (has_bias_) checkCudaErrors(cudaFree(bias_data_->data));
}

void Int8FCLayer::SetCudnn() {}

void Int8FCLayer::readWeightFromModel(const caffe::LayerParameter& layer_param, float weight_scale, float bias_scale) {
  const float *weight = layer_param.blobs(0).data().data();
  vector<int8_t> weight_data(weight_count_);
  // [input channels x input height x input width] x [output width]
  int mm, nn;
  mm = bottom_count_;
  nn = top_count_;
  cout << "[Int8FCLayer::readWeightFromModel] fc scaled weight: " << name() << " " << mm << " " << nn << " weight scale: " << weight_scale << "\n";
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
    cout << "\n[Int8FCLayer::readWeightFromModel] fc scaled bias " << name() << " bias scale: " << bias_scale <<  "\n";
    const float *bias = layer_param.blobs(1).data().data();
    vector<int8_t> bias_data(bias_count_);
    for (int k = 0; k < bias_count_; ++k) {
      int scaled_bias = std::round(bias[k] * bias_scale_);
      bias_data[k] = scaled_bias > 127 ? 127 : (scaled_bias < -127 ? -127 : scaled_bias);
      // if (name() == "ip2") cout << bias[k] << " ";
    }
    setBias(bias_data);
  }
}