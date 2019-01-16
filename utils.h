
#ifndef _UTILS_H
#define _UTILS_H

#include "cuda.h"
#include "cudnn.h"
#include "cuda_runtime.h"
#include <cublas_v2.h>

#include <iostream>
#include <assert.h>
#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include "caffe.pb.h"

#include <glog/logging.h>


#define checkCUDNN(status)                                                     \
  do {                                                                         \
    stringstream _error;                                                       \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      cout << "CUDNN failure: " << cudnnGetErrorString(status) << endl;        \
	    exit(1);															                                    \
    }                                                                          \
  } while (0)


#define checkCudaErrors(status)                                                \
  do {                                                                         \
    if (status != 0) {                                                         \
      cout << "Cuda failure: " << status << endl;                              \
	    exit(1);															                                   \
    }                                                                          \
  } while (0)


#define checkCUBLAS(status)                                                     \
  do {                                                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                      \
      cout << "CUBLAS failure." << endl;                                        \
	    exit(1);															                                      \
    }                                                                          \
  } while (0)


// using caffe parameters, for cuda programming
const int CAFFE_CUDA_NUM_THREADS = 512;
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
    i < (n); \
    i += blockDim.x * gridDim.x)


#endif
