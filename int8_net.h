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

protected:

private:
  static bool readProtoFromBinary(const char* filepath, google::protobuf::Message* message);
  // calculate alpha for convolution layers
  void calculateScales();
  void getNextScaleLayer(map<string, string>& next_scale_layer);

  vector<Int8Layer*> layers_;
  std::map<string, int> layer_table_;
  std::map<string, float> calib_table_;
};

Int8Net& Int8Net::add(Int8Layer& layer) {
  layers_.push_back(&layer);
  if (layers_.size() > 1) {
    int size = layers_.size();
    CHECK_EQ(layers_[size-2]->top_count(), layers_[size-1]->bottom_count()) << "last top size and bottom size do not match";
    layers_[size-1]->setBottomData(layers_[size-2]->getTopData());
  }
  layer_table_[layer.name()] = layers_.size() - 1;
  return *this;  // to allow net.add(x).add(x).add(x)
}

void Int8Net::feed(const vector<int8_t>& weight) {
  CHECK_GE(layers_.size(), 0) << "cannot feed, no layer added to the net";
  layers_[0]->feed(weight);
}

void Int8Net::get(vector<int8_t>& top) {
  CHECK_GE(layers_.size(), 0) << "cannot get, no layer added to the net";
  forward();
  layers_[layers_.size()-1]->get(top);
}

void Int8Net::get(vector<float>& top) {
  CHECK_GE(layers_.size(), 0) << "cannot get, no layer added to the net";
  forward();
  layers_[layers_.size()-1]->get(top);
}

void Int8Net::forward() {
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->forward();
  }
}

bool Int8Net::readCalibration(string table_name) {
  std::ifstream in(table_name);
  if(!in) {
    LOG(INFO) << "calibration file not found.";
    return false;
  }
  string name;
  float value;
  while (!in.eof()){
    in >> name >> value;
    LOG(INFO) << "\t" << name << " : " << value;
    calib_table_[name] = value;
  }
  in.close();
  calculateScales();
  return true;
}

void Int8Net::readCaffeModel(string model_name) {
  caffe::NetParameter caffemodel;
  bool status = readProtoFromBinary(model_name.c_str(), &caffemodel);
  if (!status) {
    LOG(INFO) << "caffemodel not found";
    exit(1);
  }
  map<string, int> layer_map;
  for (int i = 0; i < caffemodel.layer_size(); ++i){
    layer_map[caffemodel.layer(i).name()] = i;
  }
  for (int i = 0; i < layers_.size(); ++i) {
    string layer_name = layers_[i]->name();
    if (layer_map.count(layer_name) == 0) {
      continue;
    }
    int index = layer_map[layer_name];
    int blobs_size = caffemodel.layer(index).blobs_size();
    if (blobs_size == 0) {
      continue;
    }
    const caffe::LayerParameter& layer_param = caffemodel.layer(index);
    layers_[i]->readWeightFromModel(layer_param, calib_table_[layer_name + "_param_0"], calib_table_[layer_name]);  // read weight and bias for conv and fc layer 
  }
}

bool Int8Net::readProtoFromBinary(const char* filepath, google::protobuf::Message* message){
  std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
  if (!fs.is_open()) {
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

// calculate alpha for convolution layers
void Int8Net::calculateScales() {
  map<string, string> next_scale_layer;
  getNextScaleLayer(next_scale_layer);
  for (int i = 0; i < layers_.size(); ++i) {
    string name = layers_[i]->name();
    if (calib_table_.count(name) != 0) {  // for conv, ip
      if (next_scale_layer.count(name) != 0) {
        string next_name = next_scale_layer[name];
        LOG(INFO) << "[calculateScales] " << name << " : " << calib_table_[name] << " " << calib_table_[name + "_param_0"];
        LOG(INFO) << "                 " << next_name << " : " << calib_table_[next_name] << " " << calib_table_[next_name + "_param_0"];
        float alpha = calib_table_[next_name] / (calib_table_[name] * calib_table_[name + "_param_0"]);
        layers_[i]->setScales(alpha, calib_table_[next_name]);
      }
      else {
        LOG(INFO) << "[calculateScales] " << name << " : " << calib_table_[name] << " " << calib_table_[name + "_param_0"];
        float alpha = 1.0 / calib_table_[name + "_param_0"];
        layers_[i]->setScales(alpha, calib_table_[name]);
      }
    }
    else {  // for pool, relu
      LOG(INFO) << "[calculateScales] " << name << " , " << next_scale_layer[name];
      string next_name = next_scale_layer[name];
      layers_[i]->setScales(1.0, calib_table_[next_name]);
    }
  }
}

void Int8Net::getNextScaleLayer(map<string, string>& next_scale_layer) {  // ugly, need to be improved
  next_scale_layer["conv1"] = "conv2";
  next_scale_layer["conv2"] = "ip1";
  next_scale_layer["ip1"] = "ip2";

  next_scale_layer["data"] = "conv1";
  next_scale_layer["pool1"] = "conv2";
  next_scale_layer["pool2"] = "ip1";
  next_scale_layer["relu1"] = "ip2";

  next_scale_layer["ip2"] = "out";
}

#endif