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
// #include "int8_softmax_layer.h"
#include "int8_convba_layer.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <map>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
  FLAGS_alsologtostderr = 1;
  // google::InitGoogleLogging("INFO");

  string image_file;
  if (argc == 1) {
    image_file = "mnist_0.png";
  }
  else {
    string number(argv[1]);
    image_file = "mnist_" + number + ".png";
  }
  cout << "reading image " << image_file << endl;


  // read image
  Mat image = imread(image_file, 0);
  cout << image.dims << " " << image.channels() << " " << image.size() << endl;
  vector<int8_t> imagedata(1 * 4 * 28 * 28);
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      int pos = i * 28 * 4 + j * 4;  // because size of channel must be 4x
      imagedata[pos] = image.data[i * 28 + j] / 2;
      imagedata[pos + 1] = imagedata[pos + 2] = imagedata[pos + 3] = 0;
      cout << (imagedata[pos] == 0 ? ' ' : '@');  // show the image
    }
    cout << endl;
  }


  // construct network
  Int8Net net;
  Int8DataLayer data_layer("data", 1, 4, 28, 28);
  net.add(data_layer);
  Int8ConvBALayer conv1("conv1", 1, 1, 20, 28, 28, 5, 1, 0, true, "none");  // 20 24 24
  net.add(conv1);
  // Int8ConvLayer conv1("conv1", 1, 1, 20, 28, 28, 5, 1, 0, true);  // 20 24 24
  // net.add(conv1);
  Int8PoolingLayer pool1("pool1", "MAX", 1, 20, 24, 24);   // 20 12 12
  net.add(pool1);
  Int8ConvBALayer conv2("conv2", 1, 20, 40, 12, 12, 5, 1, 0, true, "none");  // 40 8 8
  net.add(conv2);
  // Int8ConvLayer conv2("conv2", 1, 20, 40, 12, 12, 5, 1, 0, true);  // 40 8 8
  // net.add(conv2);
  Int8PoolingLayer pool2("pool2", "MAX", 1, 40, 8, 8);   // 40 4 4
  net.add(pool2);
  Int8FCLayer ip1("ip1", 1, 40, 4, 4, 1, 1, 500, true);  // 1, 1, 500
  net.add(ip1);
  Int8ReluLayer relu1("relu1", 1, 1, 1, 500);  // must set channel to 4 for int8 cudnn
  net.add(relu1);
  Int8FCLayer ip2("ip2", 1, 1, 1, 500, 1, 1, 10, true);  // 1, 1, 10
  net.add(ip2);
  // Int8SoftmaxLayer out("out", 1, 1, 1, 10);
  // net.add(out);

  cout << "read calibration\n";
  net.readCalibration("mnist2.table");  // read calibration after all layers are added
  cout << "read caffemodel\n";
  net.readCaffeModel("lenet_inn.caffemodel");  // read caffemodel after reading calibration
  net.feed(imagedata);

  // net.forward();
  // return 0;



  int count = net.topCount();
  vector<int8_t> result_int8(count);
  net.get(result_int8);
  for (int i = 0; i < count; ++i) {
    cout << int(result_int8[i]) << " ";
  } cout << endl;
  vector<float> result_fp32(count);
  net.get(result_fp32);
  for (int i = 0; i < count; ++i) {
    cout << result_fp32[i] << " ";
    if (i % 4 == 3) cout << "\n";
  } cout << endl;



  // measure time
  int iterations = 1000000;
  auto t1 = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < iterations; ++iter) {
    net.forward();
  }
  checkCudaErrors(cudaDeviceSynchronize());
  auto t2 = std::chrono::high_resolution_clock::now();

  LOG(INFO) << "run " << iterations << " times.";
  LOG(INFO) << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / iterations << " ms";
  printf("Iteration time: %f ms\n", 
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f / iterations);

  return 0;
}