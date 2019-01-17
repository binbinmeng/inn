#ifndef INT8_DATA_LAYER_H
#define INT8_DATA_LAYER_H

#include "int8_layer.h"

class Int8DataLayer : public Int8Layer {
public:
	explicit Int8DataLayer(string name, int n, int c, int h, int w)
      : Int8Layer(name, n, c, h, w, c, h, w) {
    CreateCudnn();
    CreateCuda();
  }

  virtual ~Int8DataLayer() {
    FreeCudnn();
    FreeCuda();
  }

protected:

private:
  virtual void Forward();
  virtual void CreateCudnn();
  virtual void CreateCuda();
  virtual void FreeCudnn();
  virtual void FreeCuda();
};

#endif
