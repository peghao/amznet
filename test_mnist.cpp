// #define DEBUG
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

extern "C"{
#include "linked_list.h"
}
#include "model.h"

class fcnet : Model
{
private:
    Linear model1 = Linear(28*28, 100);
    Linear model2 = Linear(100,100);
    Linear model3 = Linear(100,10);
public:
    tensor *forward(tensor *x)
    {
        tensor *x1 = relu(model1.forward(x));
        tensor *x2 = relu(model2.forward(x1));
        tensor *x3 = model3.forward(x2);
        return x3;
    }
};

#include <iostream>
void opt(tensor *t, float lr)
{
    linked_list *node_list = to_linked_list(t);
    for(linked_list *node = node_list; node != NULL; node=node->next)
    {
        tensor *W = (tensor *)node->p;
        if(W->op == NONE && W->requires_grad == true)
        {
            array_linear(W->grad, -lr, W->data, W->data, W->size);
        }
    }
    list_free(node_list);
}

#include <map>
#include <string.h>

class SGD{
private:
    std::map<void *, float *> grads_map;
    float lr = 0;
    float momentum = 0;
public:

    SGD(float lr=0.001, float momentum=0.9)
    {
        this->lr = lr;
        this->momentum = momentum;
    }

    void step(tensor *t){
        // printf("p=%f\n", this->momentum);
        linked_list *linked_graph = to_linked_list(t);
        for(linked_list *node=linked_graph; node!=NULL; node=node->next)
        {
            tensor *W = (tensor *)node->p;
            if(W->op != NONE || W->requires_grad != true) continue;

            auto dW_last = grads_map.find(node->p);
            if(dW_last == grads_map.end()){ //没有找到
                float *v0 = new float[W->size];
                array_constant(v0, W->size, 0.0f);
                grads_map.insert(std::pair<void *, float *>(W, v0));

                dW_last = grads_map.find(node->p);
            }

            /*更新梯度*/
            float *dw_1 = new float[W->size];
            float *dw_2 = new float[W->size];
            float *g_last = dW_last->second;
            float *g_now = W->grad;

            array_times_constant(g_last, dw_1, W->size, this->momentum);
            array_times_constant(g_now, dw_2, W->size, -this->lr);
            float *fixed_dw = new float[W->size];
            array_add(dw_1, dw_2, fixed_dw, W->size);

            memcpy(g_last, fixed_dw, W->size*sizeof(float));

            array_linear(g_last, 1.0f, W->data, W->data, W->size);

            delete[] dw_1;
            delete[] dw_2;
            delete[] fixed_dw;
        }
        list_free(linked_graph);
    }
};

void norm(float *X, size_t size)
{
    float min = array_min(X, size);
    float max = array_max(X, size);

    for(size_t i=0; i<size; i++)
        X[i] = (X[i]-min)/(max-min);
}

void norm_batched(tensor *X)
{
    size_t batch_size = width(X);
    for(size_t i=0; i<X->size/batch_size; i++)
        norm(X->data + i*batch_size, batch_size);
}

#define BATCHSIZE 128
#define EPOCHS UINT32_MAX
#define NUM_BATCHS 400


int main()
{
    size_t train_imgs_shape[] = {BATCHSIZE, 28*28};
    size_t train_labs_shape[] = {BATCHSIZE, 1};

    auto net = fcnet();
    auto opt = SGD(0.001f, 0.9f);

    tensor *train_imgs = NULL;
    tensor *train_labs = NULL;
    tensor *net_out = NULL;
    tensor *predict_labs = NULL;
    tensor *loss = NULL;

    for(int epoch=0, step=0; epoch<EPOCHS; epoch++)
    for(int i=0; i<NUM_BATCHS; i++, step++)
    {
        // if(step == 20) goto end;
        train_imgs = create_from_file((char *)"../dataset/MNIST/raw/train-images-idx3-ubyte", 16 + i*BATCHSIZE*28*28, train_imgs_shape, sizeof(train_imgs_shape)/sizeof(size_t));
        train_labs = create_from_file((char *)"../dataset/MNIST/raw/train-labels-idx1-ubyte", 8 + i*BATCHSIZE, train_labs_shape, sizeof(train_labs_shape)/sizeof(size_t));
        
        norm_batched(train_imgs); //把输入图片归一化

        net_out = net.forward(train_imgs);
        predict_labs = softmax(net_out);
        
        loss = CrossEntropyLoss(predict_labs, train_labs, 10);
        printf("epoch:%d, loss:%.12f, prograss:[%d/%d], step:%d\n", epoch, loss->data[0]/BATCHSIZE, i, NUM_BATCHS, step);

        backward(loss);
        // opt(loss, 0.0001);
        opt.step(loss);

        release(loss);

        tensor_free(train_imgs);
        tensor_free(train_labs);
    }

    end:
    return 0;
}