#ifndef INT8_NET_H
#define INT8_NET_H

#include "int8_layer.h"

class Int8Net {
public:
  Int8Net() {}

  ~Int8Net() {}

  Int8Net& add(Int8Layer& layer);
  void feed(const vector<int8_t>& weight);
  void get(vector<int8_t>& top);
  void get(vector<float>& top);
  void forward();
  bool readCalibration(string table_name);
  void readCaffeModel(string model_name);
  int topCount();

  void setOutputLayer(string layer_name);
  void readPrototxt(string proto_name);

protected:

private:
  static bool readProtoFromBinary(const char* filepath, google::protobuf::Message* message);
  // calculate alpha for convolution layers
  void calculateScales();
  void getNextScaleLayer(map<string, string>& next_scale_layer);

  vector<Int8Layer*> layers_;
  std::map<string, int> layer_table_;
  std::map<string, float> calib_table_;

  string output_layer_;
};

#endif