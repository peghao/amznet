//
// Created by tom on 2022/7/26.
//

#ifndef AMZNET_MODEL_H
#define AMZNET_MODEL_H

extern "C"
{
#include "tensor.h"

};

/*model基类*/
class model {
protected:
    tensor **params = nullptr;
    tensor **consts = nullptr;

public:
    virtual tensor *forward(tensor *x) = 0;
};

/*线性层*/
class linear : model {
public:
    linear(uint32_t input_features, uint32_t output_features);
    tensor *forward(tensor *x);
};


#endif //AMZNET_MODEL_H
