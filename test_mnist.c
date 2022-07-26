// #define DEBUG
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#include "tensor.h"

int main()
{
    size_t train_imgs_shape[] = {5, 28*28};
    size_t train_labs_shape[] = {5, 1};
    tensor *train_imgs = create_from_file("../dataset/MNIST/raw/train-images-idx3-ubyte", 16, train_imgs_shape, sizeof(train_imgs_shape)/sizeof(size_t));
    tensor *train_labs = create_from_file("../dataset/MNIST/raw/train-labels-idx1-ubyte", 8, train_labs_shape, sizeof(train_labs_shape)/sizeof(size_t));

    // show(train_imgs);
    // show("train_labs:", train_labs);

    size_t W1_shape[] = {28*28, 100}, b1_shape[] = {100, 1}, 
           W2_shape[] = {100, 100}, b2_shape[] = {100, 1},
           W3_shape[] = {100, 10}, b3_shape[] = {10, 1};
    tensor *W1=range_tensor(W1_shape, 2, -0.05, 0.05), *b1=range_tensor(b1_shape, 2, -0.05, 0.05),
           *W2=range_tensor(W2_shape, 2, -0.05, 0.05), *b2=range_tensor(b2_shape, 2, -0.05, 0.05),
           *W3=range_tensor(W3_shape, 2, -0.05, 0.05), *b3=range_tensor(b3_shape, 2, -0.05, 0.05);
    W1->requires_grad = true; b1->requires_grad = true;
    W2->requires_grad = true; b2->requires_grad = true;
    W3->requires_grad = true; b3->requires_grad = true;

    for(int i=0; i<100; i++)
    {

        tensor *X1 = relu(linear(train_imgs, W1, b1));
        tensor *X2 = relu(linear(X1, W2, b2));
        tensor *net_out = linear(X2, W3, b3);

        tensor *predict_labs = softmax(net_out);
        
        tensor *loss = CrossEntropyLoss(predict_labs, train_labs, 10);
        printf("steps:%d, loss:%f\n", i, loss->data[0]/5);
        // show("loss is:", loss);

        backward(loss);

        release(loss);

        float lr = 0.00001; //学习率

        array_linear(W1->grad, -lr, W1->data, W1->data, W1->size);
        array_linear(W2->grad, -lr, W2->data, W2->data, W2->size);
        array_linear(W3->grad, -lr, W3->data, W3->data, W3->size);
        array_linear(b1->grad, -lr, b1->data, b1->data, b1->size);
        array_linear(b2->grad, -lr, b2->data, b2->data, b2->size);
        array_linear(b3->grad, -lr, b3->data, b3->data, b3->size);

    }

    printf("system paused!\n"); //停下来看看内存有没有泄露
    getchar();
}