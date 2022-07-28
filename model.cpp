//
// Created by tom on 2022/7/26.
//
#include <stdlib.h>

#include "model.h"
#include "tensor.h"

linear::linear(uint32_t input_features, uint32_t output_features) {
//    this->params = (tensor **)malloc(2*sizeof(void*));
    this->params = (tensor **)new void*[2];

    size_t shape_W[2] = {input_features, output_features};
    size_t shape_b[2] = {output_features, 1};

    // this->params[0] = create(shape_W, 2);
    // this->params[1] = create(shape_b, 2);
    this->params[0] = range_tensor(shape_W, 2, -0.05, 0.05);
    this->params[1] = range_tensor(shape_b, 2, -0.05, 0.05);

    this->params[0]->requires_grad = true;
    this->params[1]->requires_grad = true;

}

tensor *linear::forward(tensor *x) {
    auto *W = this->params[0];
    auto *b = this->params[1];
    return add_distri(mul(x, W), b);
}