
// #include "int8_pooling_layer.h"
// #include "int8_data_layer.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>        // 核心组件
#include <opencv2/highgui/highgui.hpp>  // GUI
#include <opencv2/imgproc/imgproc.hpp>  // 图像处理

#include <fstream>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include "caffe.pb.h"

#include "int8_net.h"
#include "int8_data_layer.h"
#include "int8_conv_layer.h"
#include "int8_pooling_layer.h"
#include "int8_relu_layer.h"
#include "int8_fc_layer.h"
#include "utils.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <map>

using namespace cv;
using namespace std;

static bool read_proto_from_binary(const char* filepath, google::protobuf::Message* message)
{
  std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
  if (!fs.is_open())
  {
      fprintf(stderr, "open failed %s\n", filepath);
      return false;
  }

  google::protobuf::io::IstreamInputStream input(&fs);
  google::protobuf::io::CodedInputStream codedstr(&input);

  codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

  bool success = message->ParseFromCodedStream(&codedstr);

  fs.close();

  return success;
}

bool get_quantized_weight(vector<int8_t>& conv, caffe::NetParameter& caffemodel, float scale, int index) {
  if (scale == 0.0) {
    cout << "error: scale is 0" << endl;
    return false;
  }
  const caffe::BlobProto& blob = caffemodel.layer(index).blobs(0);
  const float *weight = blob.data().data();
  conv.resize(blob.data_size());
  for (int i = 0; i < blob.data_size(); ++i) {
    conv[i] = weight[i] * scale;
    // cout << weight[i] << " " << int(conv[i]) << " " << endl;
  }
  return true;
}

int main() {

  // read image
  Mat image = imread("mnist_0.png", 0);
  cout << image.dims << " " << image.channels() << " " << image.size() << endl;
  vector<int8_t> imagedata(1 * 4 * 28 * 28);
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      int pos = i * 28 * 4 + j * 4;  // because size of channel must be 4x
      imagedata[pos] = image.data[i * 28 + j] / 2;
      imagedata[pos + 1] = imagedata[pos + 2] = imagedata[pos + 3] = 0;
      // cout << (image.data[i * 28 + j] == 0 ? 0 : 1) << " ";  // show the image
      // cout << int(image.data[i * 28 + j]) << " ";  // show the image
      cout << int(imagedata[pos]) << " ";  // show the image
    }
    cout << endl;
  }
  // return 0;
  // for (int i = 0; i < 28; ++i) {
  //   for (int j = 0; j < 28 * 4; ++j) {
  //     cout << (imagedata[i * 28 * 4 + j] == 0 ? 0 : 1); 
  //   } cout << endl;
  // }



  // read caffemodel
  // caffe::NetParameter caffemodel;
  // bool s1 = read_proto_from_binary("lenet_iter_10000.caffemodel", &caffemodel);
  // if (!s1) {
  //   cout << "caffemodel not found" << endl;
  //   exit(1);
  // }
  // for (int i = 0; i < caffemodel.layer_size(); ++i){
  //   string name = caffemodel.layer(i).name();
  //   int blobs = caffemodel.layer(i).blobs_size();
  //   cout << name << " : " << blobs << endl;
  // }
  // const caffe::BlobProto& weight_blob = caffemodel.layer(5).blobs(0);
  // const caffe::BlobProto& bias_blob = caffemodel.layer(5).blobs(1);
  // cout << "details of " << caffemodel.layer(5).name() << endl;
  // cout << "\tweight: " << weight_blob.data_size() << endl;
  // cout << "\tbias: " << bias_blob.data_size() << endl;
  // const float *bias = bias_blob.data().data();
  // cout << "bias: \n";
  // for (int i = 0; i <bias_blob.data_size(); ++i) {
  //   cout << bias[i] << " ";
  // } cout << endl;
  // const float *weight = weight_blob.data().data();
  // cout << "weight: \n";
  // for (int i = 0; i <5; ++i) {
  //   for (int j = 0; j < 5; ++j) {
  //     cout << weight[i * 5 + j] << " ";
  //   } cout << endl;
  // } 
  // return 0;



  // construct network
  Int8Net net;
  Int8DataLayer data_layer("data", 1, 4, 28, 28);
  data_layer.setTopScale(128);
  net.add(data_layer);
  Int8ConvLayer conv1("conv1", 1, 1, 20, 28, 28, 5, 1, 0, true);  // 20 24 24
  net.add(conv1);
  Int8PoolingLayer pool1("pool1", "MAX", 1, 20, 24, 24);   // 20 12 12
  net.add(pool1);
  Int8ConvLayer conv2("conv2", 1, 20, 40, 12, 12, 5, 1, 0, true);  // 40 8 8
  net.add(conv2);
  Int8PoolingLayer pool2("pool2", "MAX", 1, 40, 8, 8);   // 40 4 4
  net.add(pool2);
  Int8FCLayer ip1("ip1", 1, 40, 4, 4, 1, 1, 500, true);  // 1, 1, 500
  net.add(ip1);
  Int8ReluLayer relu1("relu1", 1, 4, 1, 125);  // must set channel to 4 for int8 cudnn
  net.add(relu1);
  Int8FCLayer ip2("ip2", 1, 4, 1, 125, 1, 1, 10, true);  // 1, 1, 10
  net.add(ip2);

  cout << "read calibration\n";
  net.readCalibration("mnist2.table");  // read calibration after all layers are added
  cout << "read caffemodel\n";
  net.readCaffeModel("lenet_inn.caffemodel");  // read caffemodel after reading calibration
  net.feed(imagedata);

  // net.forward();
  // return 0;

  int nn = 1;
  int cc = 1;
  int hh = 1;
  int ww = 10;
  vector<int8_t> result_int8(nn * cc * hh * ww);
  net.get(result_int8);
  for (int i = 0; i < nn * cc * hh * ww; ++i) {
    cout << int(result_int8[i]) << " ";
  } cout << endl;
  vector<float> result_fp32(nn * cc * hh * ww);
  net.get(result_fp32);
  for (int i = 0; i < nn * cc * hh * ww; ++i) {
    cout << result_fp32[i] << " ";
    if (i % 4 == 3) cout << "\n";
  } cout << endl;


  // measure time
  int iterations = 1000;
  auto t1 = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < iterations; ++iter) {
    net.forward();
  }
  checkCudaErrors(cudaDeviceSynchronize());
  auto t2 = std::chrono::high_resolution_clock::now();
  printf("Iteration time: %f ms\n", 
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / iterations);



  // vector<int8_t> result_int8(1 * 20 * 24 * 24);
  // net.get(result_int8);
  // for (int i = 0; i < 24; ++i) {
  //   for (int j = 0; j < 24; ++j) {
  //     int pos = i * 24 * 20 + j * 20;
  //     cout << int(result_int8[pos]) << " ";
  //   } cout << endl;
  // }
  // for (int i = 0; i < result_int8.size(); ++i) {
  //   cout << int(result_int8[i]) << " ";
  // } cout << endl;

  // vector<float> result_float(1 * 20 * 24 * 24);
  // net.get(result_float);
  // for (int i = 0; i < 24; ++i) {
  //   for (int j = 0; j < 24; ++j) {
  //     int pos = i * 24 * 20 + j * 20;
  //     cout << result_float[pos] << " ";
  //   } cout << endl;
  // }
  // for (int i = 0; i < result_float.size(); ++i) {
  //   cout << result_float[i] << " ";
  // } cout << endl;





  // int iterations = 1;

  // int n = 100;
  // int ci = 40;
  // int co = 40;
  // int h = 300;
  // int w = 300;
  // int k = 3;

  // int8
  // vector<int8_t> weight(ci * k * k * co);
  // for (int i = 0; i < weight.size(); ++i) weight[i] = int8_t(1);
  // vector<int8_t> data(n * ci * h * w);
  // for (int i = 0; i < data.size(); ++i) {
  //   if (i % 2 == 0) data[i] = int8_t(i);
  //   else data[i] = int8_t(-i);
  // }
  // vector<int8_t> result(n * co * h * w);

  // Int8Net net;

  // Int8DataLayer data_layer(n, ci, h, w);
  // Int8ConvLayer layer(n, ci, co, h, w, k); layer.set(weight);
  // Int8PoolingLayer layer("MAX", n, ci, h, w);
  // Int8ReluLayer layer(n, ci, h, w);

  // net.add(data_layer).add(layer);

  // net.feed(data);
  // net.get(result);

  // auto t1 = std::chrono::high_resolution_clock::now();
  // for (int iter = 0; iter < iterations; ++iter) {
  //   net.forward();
  // }
  // checkCudaErrors(cudaDeviceSynchronize());
  // auto t2 = std::chrono::high_resolution_clock::now();
  // printf("Iteration time: %f ms\n",
  //     std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / iterations);

  // for (int i = 0; i < result.size(); ++i) cout << int(result[i]) << " ";


  // conv_layer.bottom(data_layer.top());

  // data_layer.forward();
  // conv_layer.forward();
  // conv_layer.get(&result[0]);

  // for (int i = 0; i < result.size(); ++i) cout << int(result[i]) << " ";
  // for (int i = 0; i <100; ++i) conv_layer.forward();


  // Int8PoolingLayer pooling_layer(ci, co, h, w, n);

  // Int8FCLayer fclayer(1, 4, 4, 4, 10);


  cout << endl;

  return 0;
}