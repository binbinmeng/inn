#ifndef INT8_NET_H
#define INT8_NET_H

#include "int8_layer.h"
#include <map>

#include <fstream>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include "caffe.pb.h"

class Int8Net {
public:
  Int8Net() {}

  ~Int8Net() {}

  bool readCalibration(string table_name) {
    std::ifstream in(table_name);
    if(!in) {
      cout << "calibration file not found." << endl;
      return false;
    }
    string name;
    float value;
    // cout << "Reading calibration table...\n";
    while (!in.eof()){
      in >> name >> value;
      cout << "\t" << name << " : " << value << "\n";
      calib_table_[name] = value;
    }
    in.close();
    calculateAlpha();
    return true;
  }

  Int8Net& add(Int8Layer& layer) {
    layers_.push_back(&layer);
    if (layers_.size() > 1) {
      int size = layers_.size();
      cout << "bottom count : " << layers_[size-1]->bottom_count() << " vs top count : " << layers_[size-2]->top_count() << endl;
      assert(layers_[size-1]->bottom_count() == layers_[size-2]->top_count());
      // if (layers_[size-1]->bottom_count() == layers_[size-2]->top_count()) {
      //   cout << "bottom count : " << layers_[size-1]->bottom_count() << " vs top count : " << layers_[size-2]->top_count() << endl;
      //   assert(layers_[size-1]->bottom_count() == layers_[size-2]->top_count());
      // }
      layers_[size-1]->setBottomData(layers_[size-2]->getTopData());
    }
    layer_table_[layer.name()] = layers_.size() - 1;
    return *this;  // to allow net.add(x).add(x).add(x)
  }

  void feed(const vector<int8_t>& weight) {
    assert(layers_.size() > 0);
    layers_[0]->feed(weight);
  }

  void get(vector<int8_t>& top) {
    forward();
    layers_[layers_.size()-1]->get(top);
  }

  void get(vector<float>& top) {
    forward();
    layers_[layers_.size()-1]->get(top);
  }

  void forward() {
    for (int i = 0; i < layers_.size(); ++i) {
      layers_[i]->forward();
    }
  }

  void readCaffeModel(string model_name) {
    caffe::NetParameter caffemodel;
    bool status = readProtoFromBinary(model_name.c_str(), &caffemodel);
    if (!status) {
      cout << "caffemodel not found" << endl;
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

protected:

private:
  vector<Int8Layer*> layers_;
  std::map<string, int> layer_table_;
  std::map<string, float> calib_table_;

  static bool readProtoFromBinary(const char* filepath, google::protobuf::Message* message)
  {
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
  void calculateAlpha() {
    map<string, string> next_scale_layer;
    getNextScaleLayer(next_scale_layer);
    for (int i = 0; i < layers_.size(); ++i) {
      string name = layers_[i]->name();
      if (calib_table_.count(name) != 0) {  // for conv, ip
        if (next_scale_layer.count(name) != 0) {
          string next_name = next_scale_layer[name];
          cout << "[calculateAlpha] " << name << " : " << calib_table_[name] << " " << calib_table_[name + "_param_0"] << endl;
          cout << "                 " << next_name << " : " << calib_table_[next_name] << " " << calib_table_[next_name + "_param_0"] << endl;
          float alpha = calib_table_[next_name] / (calib_table_[name] * calib_table_[name + "_param_0"]);
          layers_[i]->setAlphaAndBeta(alpha, 0.0);
          layers_[i]->setBiasScale(calib_table_[next_name]);
        }
        else {
          cout << "[calculateAlpha] " << name << " : " << calib_table_[name] << " " << calib_table_[name + "_param_0"] << endl;
          float alpha = 1.0 / calib_table_[name + "_param_0"];
          layers_[i]->setAlphaAndBeta(alpha, 0.0);
          layers_[i]->setBiasScale(calib_table_[name]);
        }
      }
      else {  // for pool, relu
        cout << "[calculateAlpha] " << name << " , " << next_scale_layer[name] << endl;
        string next_name = next_scale_layer[name];
        layers_[i]->setBiasScale(calib_table_[next_name]);
      }
    }
  }

  void getNextScaleLayer(map<string, string>& next_scale_layer) {
    next_scale_layer["conv1"] = "conv2";
    next_scale_layer["conv2"] = "ip1";
    next_scale_layer["ip1"] = "ip2";

    next_scale_layer["pool1"] = "conv2";
    next_scale_layer["pool2"] = "ip1";
    next_scale_layer["relu1"] = "ip2";

    next_scale_layer["ip2"] = "out";
  }
};

#endif