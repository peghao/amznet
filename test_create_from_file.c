#include "tensor.h"

int main()
{
    size_t shape_t1[] = {5, 1};
    size_t shape_t2[] = {5, 28, 28};

    tensor *t1 = create_from_file("./dataset/MNIST/raw/train-images-idx3-ubyte", 8, shape_t1, 2);
    tensor *t2 = create_from_file("./dataset/MNIST/raw/train-labels-idx1-ubyte", 16, shape_t2, 3);

    show("前五张图的标签：", t1);
    show("前五张图的数据：", t2);
}