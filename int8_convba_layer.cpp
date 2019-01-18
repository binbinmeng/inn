#include "int8_convba_layer.h"

void Int8ConvBALayer::Forward() {
  // checkCudaErrors(cudaSetDevice(m_gpuid));
  // cudnnConvolutionForward(
  //     handle_, 
  //     &alpha_, bottom_desc_, bottom_data_,
  //     filter_desc_, weight_data_, conv_desc_, 
  //     conv_algo_, work_space_, workspace_size_,
  //     &zero_float_, top_desc_, top_data_);
  // if (has_bias_) {
  //   cudnnAddTensor(
  //       handle_,
  //       &one_float_,
  //       bias_desc_, bias_data_,
  //       &one_float_,
  //       top_desc_, top_data_);
  // }

  // y = act ( alpha1 * conv(x) + alpha2 * z + bias )
  // x          w	        y and z     bias        alpha1/alpha2
  // X_INT8     X_INT8    X_INT8      X_FLOAT     X_FLOAT
  // X_INT8     X_INT8    X_FLOAT     X_FLOAT     X_FLOAT

  checkCUDNN(cudnnConvolutionBiasActivationForward(
      handle_,
      &alpha_, bottom_desc_, bottom_data_,
      filter_desc_, weight_data_, conv_desc_,
      conv_algo_, work_space_, workspace_size_,
      &zero_float_, z_desc_, z_data_,
      bias_desc_, bias_data_,
      activ_desc_,
      top_desc_, top_data_));

  // checkCUDNN(cudnnConvolutionBiasActivationForward(
  //     cudnnHandle_t                       handle,
  //     const void                         *alpha1,
  //     const cudnnTensorDescriptor_t       xDesc,
  //     const void                         *x,
  //     const cudnnFilterDescriptor_t       wDesc,
  //     const void                         *w,
  //     const cudnnConvolutionDescriptor_t  convDesc,
  //     cudnnConvolutionFwdAlgo_t           algo,
  //     void                               *workSpace,
  //     size_t                              workSpaceSizeInBytes,
  //     const void                         *alpha2,
  //     const cudnnTensorDescriptor_t       zDesc,
  //     const void                         *z,
  //     const cudnnTensorDescriptor_t       biasDesc,
  //     const void                         *bias,
  //     const cudnnActivationDescriptor_t   activationDesc,
  //     const cudnnTensorDescriptor_t       yDesc,
  //     void                               *y));
}

void Int8ConvBALayer::CreateCudnn() {
  checkCUDNN(cudnnCreate(&handle_));  // handle
  checkCUDNN(cudnnCreateTensorDescriptor(&bottom_desc_));  // bottom
  checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc_));  // weight
  checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));  // conv
  checkCUDNN(cudnnCreateTensorDescriptor(&top_desc_));  // top
  if (has_bias_) checkCUDNN(cudnnCreateTensorDescriptor(&bias_desc_));  // bias
  checkCUDNN(cudnnCreateTensorDescriptor(&z_desc_));  // z
  checkCUDNN(cudnnCreateActivationDescriptor(&activ_desc_));  // activation
  SetCudnn();
}

void Int8ConvBALayer::FreeCudnn() {
  checkCUDNN(cudnnDestroyTensorDescriptor(bottom_desc_));
  checkCUDNN(cudnnDestroyTensorDescriptor(top_desc_));
  checkCUDNN(cudnnDestroyFilterDescriptor(filter_desc_));
  checkCUDNN(cudnnDestroyConvolutionDescriptor(conv_desc_));
  if (has_bias_) checkCUDNN(cudnnDestroyTensorDescriptor(bias_desc_));
  checkCUDNN(cudnnDestroyTensorDescriptor(z_desc_));
  checkCUDNN(cudnnDestroyActivationDescriptor(activ_desc_));
  checkCUDNN(cudnnDestroy(handle_));
}

void Int8ConvBALayer::CreateCuda() {
  checkCudaErrors(cudaMalloc(&top_data_, sizeof(int8_t) * top_count_));
  checkCudaErrors(cudaMalloc(&weight_data_, sizeof(int8_t) * weight_count_));
  if (has_bias_) checkCudaErrors(cudaMalloc(&bias_data_, sizeof(int8_t) * bias_count_));
  checkCudaErrors(cudaMalloc(&work_space_, workspace_size_));
  checkCudaErrors(cudaMalloc(&z_data_, sizeof(int8_t) * bias_count_));
}

void Int8ConvBALayer::FreeCuda() {
  checkCudaErrors(cudaFree(top_data_));
  checkCudaErrors(cudaFree(weight_data_));
  if (has_bias_) checkCudaErrors(cudaFree(bias_data_));
  checkCudaErrors(cudaFree(work_space_));
  checkCudaErrors(cudaFree(z_data_));
}

void Int8ConvBALayer::SetCudnn() {
  checkCUDNN(cudnnSetTensor4dDescriptor(
      bottom_desc_, CUDNN_TENSOR_NHWC, CUDNN_DATA_INT8,
      batch_size_, in_channels_, in_height_, in_width_));

  checkCUDNN(cudnnSetFilter4dDescriptor(
      filter_desc_, CUDNN_DATA_INT8, CUDNN_TENSOR_NHWC,
      out_channels_, in_channels_, kernel_size_, kernel_size_));

  checkCUDNN(cudnnSetConvolution2dDescriptor(
      conv_desc_, pad_, pad_, stride_, stride_, 1, 1, // padding, stride, delation
      CUDNN_CONVOLUTION, CUDNN_DATA_INT32));

  int n, c, h, w;
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
      conv_desc_, bottom_desc_, filter_desc_,
      &n, &c, &h, &w));
  LOG(INFO) << name_ << " output nchw: " << n << " " << c << " " << h << " " << w << " " << batch_size_ << " " << out_channels_ << " " << out_height_ << " " << out_width_;
  
  checkCUDNN(cudnnSetTensor4dDescriptor(
      top_desc_, CUDNN_TENSOR_NHWC, CUDNN_DATA_INT8,
      n, c, h, w));

  if (has_bias_) {
    checkCUDNN(cudnnSetTensor4dDescriptor(
        bias_desc_, CUDNN_TENSOR_NHWC, CUDNN_DATA_INT8,
        1, out_channels_, 1, 1));
  }
  conv_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;  // the only algo for int8
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
      handle_, bottom_desc_, filter_desc_, conv_desc_, top_desc_,
      conv_algo_, &workspace_size_));

  LOG(INFO) << name_ << " workspace size: " << workspace_size_;
  CHECK_GE(workspace_size_, 0) << "the size of workspace should be larger than 0";

  // new for convba
  checkCUDNN(cudnnSetTensor4dDescriptor(
      z_desc_, CUDNN_TENSOR_NHWC, CUDNN_DATA_INT8,
      1, out_channels_, 1, 1));
  checkCUDNN(cudnnSetActivationDescriptor(
      activ_desc_, CUDNN_ACTIVATION_IDENTITY,
      CUDNN_NOT_PROPAGATE_NAN, double(0)));
}

void Int8ConvBALayer::readWeightFromModel(const caffe::LayerParameter& layer_param, float weight_scale, float bias_scale) {
  const float *weight = layer_param.blobs(0).data().data();
  vector<int8_t> weight_data(weight_count_);
  // caffe format: output channels x input channels per-group x kernel height x kernel width (kcrs)
  // cudnn int8 format: output channels x rows x columns x input channels (krsc)
  int kk, rr, ss, cc;
  kk = out_channels_;
  cc = in_channels_origin_;
  rr = 5;
  ss = 5;
  int cc_extend = in_channels_;
  LOG(INFO) << "conv scaled weight: " << name() << kk << " " << cc << " " << rr << " " << ss << " " << cc_extend;
  // magic
  for (int k = 0; k < kk; ++k) {
    for (int r = 0; r < rr; ++r) {
      for (int s = 0; s < ss; ++s) {
        for (int c = 0; c < cc; ++c) {
          int kcrs = k*cc*rr*ss + c*rr*ss + r*ss + s;
          int krsc = k*rr*ss*cc_extend + (rr-r-1)*ss*cc_extend + (ss-s-1)*cc_extend + c;
          int scaled_weight = std::round(weight[kcrs] * weight_scale);
          weight_data[krsc] = scaled_weight > 127 ? 127 : (scaled_weight < -127 ? -127 : scaled_weight);
          // if (k == 0) {
          //   cout << int(weight_data[krsc]) << " ";
          // }
        }
      }
    }
  }
  setWeight(weight_data);

  if (bias_count_ > 0 && layer_param.blobs_size() > 1) {
    LOG(INFO) << "conv scaled bias " << name();
    const float *bias = layer_param.blobs(1).data().data();
    vector<int8_t> bias_data(bias_count_);
    for (int k = 0; k < bias_count_; ++k) {
      int scaled_bias = std::round(bias[k] * bias_scale_);
      // int scaled_bias = std::round(bias[k] * bias_scale);
      bias_data[k] = scaled_bias > 127 ? 127 : (scaled_bias < -127 ? -127 : scaled_bias);
    }
    setBias(bias_data);
  }
}