#include "tensor.h"


tensor *linear(tensor *X, tensor *W, tensor *b)
{
    return add_broad(mul(X, W), b);
}

int main()
{
    size_t train_imgs_shape[] = {5, 28*28};
    size_t train_labs_shape[] = {5, 1};
    tensor *train_imgs = create_from_file("../dataset/MNIST/raw/train-images-idx3-ubyte", 16, train_imgs_shape, sizeof(train_imgs_shape)/sizeof(size_t));
    tensor *train_labs = create_from_file("../dataset/MNIST/raw/train-labels-idx1-ubyte", 8, train_labs_shape, sizeof(train_labs_shape)/sizeof(size_t));

    // show(train_imgs);
    show(train_labs);

    size_t W1_shape[] = {28*28, 100}, b1_shape[] = {100, 1}, 
           W2_shape[] = {100, 100}, b2_shape[] = {100, 1},
           W3_shape[] = {100, 10}, b3_shape[] = {10, 1};
    tensor *W1=range_tensor(W1_shape, 2, -0.05, 0.05), *b1=range_tensor(b1_shape, 2, -0.05, 0.05),
           *W2=range_tensor(W2_shape, 2, -0.05, 0.05), *b2=range_tensor(b2_shape, 2, -0.05, 0.05),
           *W3=range_tensor(W3_shape, 2, -0.05, 0.05), *b3=range_tensor(b3_shape, 2, -0.05, 0.05);
    W1->requires_grad = true; b1->requires_grad = true;
    W2->requires_grad = true; b2->requires_grad = true;
    W3->requires_grad = true; b3->requires_grad = true;

    tensor *X1 = relu(linear(train_imgs, W1, b1));
    tensor *X2 = relu(linear(X1, W2, b2));
    tensor *Y_hat = linear(X2, W3, b3);

    show(Y_hat);

    tensor *Y_hat_softed = softmax(Y_hat);
    show(Y_hat_softed);
    tensor *loss = CrossEntropyLoss(Y_hat_softed, train_labs, 10);
    // tensor *loss = sum_all(Y_hat);
    show(loss);

    backward(loss);

    show_grad(Y_hat_softed);
    show_grad(Y_hat);
    show_grad(b1);
}