#ifndef __TENSOR__
#define __TENSOR__

#include <stddef.h> //size_t
#include <stdint.h>

#include "linked_list.h"
#include "raw_array.h"
#include "raw_matrix.h"

/*支持反向传播的“buildin”运算*/
enum operator_t //_t means tensor
{
    NONE = 0,
    MUL, //✅ 矩阵乘法操作
    SUM_ALL, //✅
    SUM, //✅
    SUM_2D, //✅
    ADD,
    ADD_C,
    ADD_BROAD, //✅
    ADD_DISTRI,
    // SUB_C,
    TIMES,
    TIMES_C, //✅
    TIMES_BROAD,
    DIV_BROAD, //✅
    LOG, //✅
    EXP, //✅
    TRANSPOSE,
    RELU,
    CONST // 运算过程中引入的常数张量
};


struct tensor
{
    float *data; //张量中的数据
    size_t size; //数组data中有多少个元素，也就是张量中分量的个数
    size_t *shape; //张量的形状，每个维度最多有MAX_ULONG个元素，已经很大很大了
    size_t dim; //张量的维数，即shape中元素的个数
    float *grad; //一个标量对自己的梯度，形状和自己是一样的。初始化应为NULL
    
    struct tensor *prev1; //指向计算图中上一个张量的指针，初始化应为NULL
    struct tensor *prev2;
    enum operator_t op; //表示本tensor是由什么计算符得来的。初始化应为NONE
    bool requires_grad; //是否需要梯度，不需要的话，反向传播的时候不会计算该tensor的梯度。默认初始化为不需要
    bool is_end; //是否是计算图的终点。默认初始化为 是
    uint8_t num_quotes;
};

typedef struct tensor tensor;

bool check_shape(tensor *t1, tensor *t2);
size_t width(tensor *t);
size_t height(tensor *t);
void show_shape(tensor *t);
void show(char *msg, tensor *t);
void show_grad(char *msg, tensor *t);
tensor *none();
tensor *create(size_t *shape, size_t dim);
tensor *constant(size_t *shape, size_t dim, float a);
tensor *range_tensor(size_t *shape, size_t dim, float start, float end);
tensor *rand_tensor(size_t *shape, size_t dim);
tensor *create_from_file(char *name, size_t offset, size_t *shape, size_t dim);
void tensor_free(tensor *t);
linked_list *to_linked_list(tensor *t);
void release(tensor *t);
tensor *times(tensor *t1, tensor *t2);
tensor *times_constant(tensor *t1, float c);
tensor *times_broad(tensor *t1, tensor *t2);
tensor *sum_all(tensor *t);
tensor *sum_2d(tensor *t1, size_t dim);
tensor *sum(tensor *t1, size_t dim);
tensor *dot(tensor *t1, tensor *t2);
tensor *add(tensor *t1, tensor *t2);
tensor *add_constant(tensor *t1, float c);
tensor *add_distri(tensor *t1, tensor *t2);
tensor *add_broad(tensor *t1, tensor *t2);
tensor *div_broad(tensor *t1, tensor *t2);
tensor *mul(tensor *t1, tensor *t2);
tensor *transpose(tensor *x);
tensor *log_t(tensor *x);
tensor *exp_t(tensor *x);
tensor *softmax(tensor *t);
tensor *BCELoss(tensor *X, tensor *Y);
tensor *onehot(tensor *t1, uint32_t num_classes);
tensor *CrossEntropyLoss(tensor *Y_hat, tensor *Y, uint32_t num_classes);
tensor *relu(tensor *x);
void backward(tensor *t);

#endif