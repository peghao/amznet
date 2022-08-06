/**
 * @file tensor.h
 * @brief 定义了张量及其各种运算，定义了反向传播函数
 * @date 2022-07-04
 */

#include <stdio.h>
#include <math.h> 
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>
#include <stdbool.h>

//把这个注释掉，计算过程中的调试信息会消失
// #define DEBUG

#include "raw_array.h"
#include "raw_matrix.h"
#include "linked_list.h"

#include "tensor.h"

/**
 * @brief 检查张量t1和t2的形状是不是一致
 * 
 * @param t1 
 * @param t2 
 * @return true 一致
 * @return false 不一致
 */
bool check_shape(tensor *t1, tensor *t2)
{
    if(t1->dim != t2->dim) return false;
    for(size_t i=0; i<t1->dim; i++)
        if(t1->shape[i] != t2->shape[i]) return false;
    return true;
}

bool check_shape_and_dim(tensor *t)
{
    //TODO: 检测t—>shape中所有元素相乘，是不是等于t->size
    return true;
}

/**
 * @brief 获取一个tensor最后一维的元素个数
 * 
 * @param t 
 * @return size_t 
 */
size_t width(tensor *t)
{
    // printf("width:%zu\n", (t->shape)[t->dim]);
    return (t->shape)[t->dim - 1]; //shape 的最后一维
}

size_t height(tensor *t)
{
    return (t->shape)[t->dim - 2]; //shape的倒数第二维
}

void show_shape(tensor *t)
{
    if(t == NULL) printf("show shape error: the input tensor t is NULL!"), exit(-1);
    printf("tensor: %p, size=%zu, dim=%zu, shape=", t, t->size, t->dim);
    printf("[");
    for(size_t i=0; i<t->dim; i++)
        printf("%zu, ", t->shape[i]);
    printf("\b]\n");
}

/**
 * @brief 输出一个tensor
 * 
 * @param msg 输出之前要显示的一小段提示信息，为NULL时不输出任何信息
 * @param t 要显示的tensor
 */
void show(char *msg, tensor *t)
{
    if(t == NULL) printf("show error: 输入tensor为NULL！\n"), exit(-1);
    if(msg != NULL) printf("\033[31m%s\033[0m\n", msg);

    show_shape(t);

    /*把一个tensor当作很多个矩阵，一个一个地show出来*/
    size_t matrix_size = height(t)*width(t);
    for(size_t i=0; i<(t->size / matrix_size); i++)
        matrix_show_raw(t->data + i*matrix_size, height(t), width(t));
    printf("\n");
}

/**
 * @brief 显示输入张量t的梯度，基本上是上面show()的翻版
 * 
 * @param msg 
 * @param t 
 */
void show_grad(char *msg, tensor *t)
{
    if(t == NULL)
    {
     printf("show_grad warnnig: 输入tensor为NULL！\n"); return;
       
    }
    if(t->grad == NULL) printf("show grad error: tensor %p grad is NULL!\n", t), exit(-1);
    if(msg != NULL) printf("\033[31m%s\033[0m\n", msg);
    show_shape(t);
    size_t matrix_size = height(t)*width(t);
    for(size_t i=0; i<(t->size / matrix_size); i++)
        matrix_show_raw(t->grad + i*matrix_size, height(t), width(t));
    printf("\n");
}

/**
 * @brief 把一个计算图显示出来，还没写好，不能用
 * 
 * @param t 
 */
void show_graph(tensor *t)
{
    if(t == NULL) return;
    show(NULL, t);
    if(t->grad != NULL) show_grad(NULL, t);
    show_graph(t->prev1);
    show_graph(t->prev2);
}


void warning()
{
    printf("汪汪!\n");
}

/**
 * @brief 创建一个空的张量，一个空的tensor只占据sizeof(tensor)大小的内存
 * 
 * @return tensor* 创建好的tensor
 */
tensor *none()
{
    tensor *t = (tensor *)malloc(sizeof(tensor));
    t->data = NULL;
    t->size = 0;
    t->shape = NULL;
    t->dim = 0;
    t->grad = NULL;
    t->prev1 = NULL;
    t->prev2 = NULL;
    t->op = NONE;
    t->requires_grad = false;
    t->is_end = true;
    t->num_quotes = 0;
    return t;
}

tensor *create(size_t *shape, size_t dim)
{
    /*检查给定的shape和dim是不是有效的*/
    // if(check_shape_and_dim(shape, dim) != true) printf("create error: bad shape and dim!"), exit(-1);      
    
    /*从shape和dim生成张量中数据的个数*/
    size_t size = 1;
    for(size_t i=0; i<dim; i++) size *= shape[i];
    
    /*创建并初始化张量*/
    tensor *t = none();
    t->size = size;
    t->data = (float *)malloc(size * sizeof(float)); //为张量的数据分配内存
    t->shape = (size_t*)memcpy(malloc(dim*sizeof(size_t)), shape, dim*sizeof(size_t)); //把传入的参数copy一份，把copy出来的shape用来初始化新创建的tensor。为什么要copy呢？因为传进来的shape可能是一个局部变量，也可能是应一个tensor的shape
    t->dim = dim;
    return t;
}

/**
 * @brief 创建一个新的tensor，并将改tensor中的数据全部用参数a来初始化
 * 
 * @param shape 新tensor的形状
 * @param dim 
 * @param a 用来初始化新张量的数
 * @return tensor* 创建好的张量
 */
tensor *constant(size_t *shape, size_t dim, float a)
{
    tensor *t = create(shape, dim);
    array_constant(t->data, t->size, a);
    return t;
}

tensor *range_tensor(size_t *shape, size_t dim, float start, float end)
{
    tensor *t = create(shape, dim);
    array_range(t->data, t->size, start, end);
    return t;
}

tensor *rand_tensor(size_t *shape, size_t dim)
{
    tensor *t = create(shape, dim);
    array_rand(t->data, t->size);
    return t;
} 

tensor *create_from_file(char *name, size_t offset, size_t *shape, size_t dim)
{
    /**
     * TODO:
     * 1. 设置错误提示，检查文件是不是存在、是不是够长
     * 2. 优化for循环读取速度
     */
    FILE *fp = fopen(name, "r");
    fseek(fp, offset, SEEK_SET);

    tensor *t = create(shape, dim);

    for(size_t i=0; i<t->size; i++)
    {
        uint8_t pixel = 0;
        fread(&pixel, 1, 1, fp);
        t->data[i] = (float)pixel;
    }
    fclose(fp);
    return t;
}

void tensor_free(tensor *t)
{
    free(t->data);
    free(t->shape);
    if(t->requires_grad == true && t->grad != NULL) free(t->grad);
    free(t);
}

linked_list *to_linked_list(tensor *t)
{
    /*将计算图t中的所有tensor添加到一个链表中（不采用递归）*/
    size_t num_append = 0;
    linked_list *node_list = list_create(t);
    linked_list *added_head = node_list;
    num_append = 1;

    for(;;)
    {
        int temp_num_append = 0;
        for(int i=0; i<num_append; i++)
        {
            tensor *temp_node = list_index(added_head, i)->p;
            tensor *temp_prev1 = temp_node->prev1, *temp_prev2 = temp_node->prev2;
            if(temp_prev1 != NULL) if(append_no_repeat(node_list, temp_prev1) == true) temp_num_append++;
            if(temp_prev2 != NULL) if(append_no_repeat(node_list, temp_prev2) == true) temp_num_append++;
        }
        if(temp_num_append == 0) break;
        added_head = list_index(added_head, num_append);
        num_append = temp_num_append;
    }

    return node_list;
}

/*递归释放计算图可能会double free，所以采用逻辑复杂的非递归方法*/
void release(tensor *t)
{
    linked_list *node_list = to_linked_list(t);

    #ifdef DEBUG
    printf("node list len:%lu\n", list_len(node_list));
    #endif

    /*遍历链表中的所有元素，就相当于遍历了整个计算图*/
    for(linked_list *node_iterator=node_list; node_iterator != NULL; node_iterator = node_iterator->next)
    {
        tensor *temp_tensor = node_iterator->p; //to be free
        if(temp_tensor->op != NONE)
        {
            tensor_free(temp_tensor);
            #ifdef DEBUG
            printf("freed:%p\n", temp_tensor);
            #endif
        }
        /*对需要计算梯度的tensor的梯度清零，相当于pytorch里的zero_grad()*/
        if(temp_tensor->op == NONE && temp_tensor->requires_grad == true && temp_tensor->grad != NULL)
        {
            array_constant(temp_tensor->grad, temp_tensor->size, 0.0f);
            // show_grad("zero grad:", temp_tensor);
        }
    }
    list_free(node_list);
}

/*对张量t1和t2做逐元素乘法，这里t1和t2必须是两个形状相同的张量*/
tensor *times(tensor *t1, tensor *t2)
{
    if(t1 == NULL || t2 == NULL)
        printf("times error: t1:%p or t2:%p is NULL!", t1, t2), exit(-1);
    if(check_shape(t1, t2) == false)
        printf("times error: t1和t2形状不同！\n"), show_shape(t1), show_shape(t2), exit(-1);
    
    tensor *t = create(t1->shape, t1->dim);
    array_times(t1->data, t2->data, t->data, t->size);

    t->prev1 = t1;
    t->prev2 = t2;
    t->requires_grad = t1->requires_grad == true || t2->requires_grad == true ? true : false;
    t->op = TIMES;
    t->is_end = true;

    t1->is_end = false;
    t2->is_end = false;
    t1->num_quotes++;
    t2->num_quotes++;
    return t;
}

tensor *times_constant(tensor *t1, float c)
{
    //TODO: 检查是否为NULL
    tensor *t = create(t1->shape, t1->dim);
    array_times_constant(t1->data, t->data, t1->size, c);

    size_t shape_t2[] = {1,1};
    tensor *t2 = create(shape_t2, 2);
    t2->data[0] = c;
    t2->op = CONST;

    t->prev1 = t1;
    t->prev2 = t2;
    t->requires_grad = t1->requires_grad;
    t->op = TIMES_C;
    t->is_end = true;

    t1->is_end = false;
    t2->is_end = false;

    t1->num_quotes++;
    t2->num_quotes++;
    return t;
}

tensor *times_broad(tensor *t1, tensor *t2)
{
    size_t t1_rows = t1->size/width(t1); //t1的行数
    //TODO: 检查t2是不是列向量
    if(t2->size != t1_rows) printf("times_broad error: t1 and t2 不能以广播方式相加！"), exit(-1);
    
    tensor *t = create(t1->shape, t1->dim);

    array_times_broad(t1->data, t2->data, t->data, t1_rows, width(t1));

    t->prev1 = t1;
    t->prev2 = t2;
    t->requires_grad = t1->requires_grad == true || t2->requires_grad == true ? true : false;
    t->op = TIMES_BROAD;
    t->is_end = true;

    t1->is_end = false;
    t2->is_end = false;
    return t;
}

tensor *sum_all(tensor *t)
{
    if(t == NULL) printf("sum error: the input tensor t=%p is NULL!", t), exit(-1);
    size_t shape[] = {1,1};
    size_t dim = 2;
    tensor *sum_t = create(shape, dim);
    array_sum(t->data, t->size, sum_t->data);
    
    sum_t->prev1 = t;
    sum_t->prev2 = NULL;
    sum_t->requires_grad = t->requires_grad;
    sum_t->op = SUM_ALL;
    sum_t->is_end = true;
    
    t->is_end = false;
    t->num_quotes++;
    return sum_t;
}

tensor *sum_2d(tensor *t1, size_t dim)
{
    size_t shape[2];
    if(dim == 0)
    {
        shape[0] = 1;
        shape[1] = t1->shape[1];
    }else if(dim == 1)
    {
        shape[0] = t1->shape[0];
        shape[1] = 1;
    }
    tensor *t = constant(shape, 2, 0); //全0的

    size_t num_row = height(t1);
    size_t row_size = width(t1);

    if(dim == 0)
    {
        for(size_t i=0; i<num_row; i++)
        {
            size_t offset = i*row_size;
            array_add(t1->data+offset , t->data, t->data, row_size);
        }
    }else if(dim == 1)
    {
        for(size_t i=0; i<num_row; i++)
        {
            size_t offset = i*row_size;
            array_sum(t1->data+offset, row_size, t->data+i); //can size_t convert to pointer saftly? size_t转成指针类型的时候会出问题吗？
        }
    }


    
    t->prev1 = t1;
    t->prev2 = NULL;
    t->requires_grad = t1->requires_grad;
    t->op = SUM_2D;
    t->is_end = true;
    
    t1->is_end = false;
    t1->num_quotes++;
    return t;
}


tensor *sum(tensor *t1, size_t dim)
{
    if(t1 == NULL) printf("sum error: the input tensor t1 is NULL!\n"), exit(-1);
    if(dim >= t1->dim) printf("sum error: the sum dim must less than input tensor dim! the input tensor t1 has dim:%zu, but the given sum dim is:%zu\n", t1->dim, dim), exit(-1);

    if(t1->dim == 2) return sum_2d(t1, dim);

    size_t sum_shape[t1->dim - 1];
    size_t sum_dim = t1->dim -1;
    size_t head_size=1, tail_size=1, dim_size=t1->shape[dim];
    for(size_t i=0; i<t1->dim; i++)
    {
        if(i<dim)
            head_size *= t1->shape[i], sum_shape[i] = t1->shape[i];
        else if(i>dim)
            tail_size *= t1->shape[i], sum_shape[i-1] = t1->shape[i];
    }
    printf("head:%zu, tail:%zu\n", head_size, tail_size);

    tensor *t = create(sum_shape, sum_dim);
    for(size_t k=0; k<head_size; k++)
    for(size_t i=0; i<dim_size; i++)
    {
        // t_k += t1_k_i
        size_t offset = k*dim_size*tail_size;
        array_add(t1->data+offset+i*tail_size, t->data+k*tail_size, t->data+k*tail_size, tail_size);
    }

    size_t shape_t2[] = {1,2};
    tensor *t2 = create(shape_t2, 2);
    t2->op = CONST;
    ((size_t *)(t2->data))[0] = dim; //将size_t类型的dim以二进制的方式储存在t2中，以后使用的时候，也必须以同样的方式“解包”出来
    
    t->prev1 = t1;
    t->prev2 = t2;
    t->requires_grad = t1->requires_grad;
    t->op = SUM;
    t->is_end = true;
    
    t1->is_end = false;
    t2->is_end = false;

    t1->num_quotes++;
    t2->num_quotes++;

    return t;
}

tensor *dot(tensor *t1, tensor *t2)
{
    return sum(times(t1,t2), -1);
}

/*对输入t1和t2做逐元素加法*/
tensor *add(tensor *t1, tensor *t2)
{
    if(t1 == NULL || t2 == NULL) printf("add error: 输入张量t1=%p或t2=%p为NULL！", t1, t2), exit(-1);
    if(check_shape(t1, t2) != true) printf("add error: 输入张量的形状应该相同！但是t1和t2的形状是：\n"), show_shape(t1), show_shape(t2), exit(-1);
    tensor *t = create(t1->shape, t1->dim);
    array_add(t1->data, t2->data, t->data, t->size);
    t->prev1 = t1;
    t->prev2 = t2;
    t->requires_grad = t1->requires_grad == true || t2->requires_grad == true ? true : false;
    t->op = ADD;
    t->is_end = true;

    t1->is_end = false;
    t2->is_end = false;

    t1->num_quotes++;
    t2->num_quotes++;

    return t;
}

tensor *add_constant(tensor *t1, float c)
{
    //TODO: 检查是否为NULL
    tensor *t = create(t1->shape, t1->dim);
    array_add_constant(t1->data, t->data, t1->size, c);

    t->prev1 = t1;
    t->prev2 = NULL;
    t->requires_grad = t1->requires_grad;
    t->op = ADD_C;
    t->is_end = true;

    t1->is_end = false;
    return t;
}

tensor *add_distri(tensor *t1, tensor *t2)
{
    if(t1->dim > 2)
        printf("add_distri error: 目前只支持2d向量的加法！\n"), exit(-1);
    if(t2->dim !=2 || width(t2) != 1)
        printf("add_distri error: t2不是列向量！\n"), exit(-1);
    if(height(t2) != width(t1))
        printf("add_distri error: t1和t2不能以广播方式相加！\n"), exit(-1);

    size_t m = height(t1), n = width(t1);
 
    tensor *t = create(t1->shape, t1->dim);

    for(size_t i=0; i<m; i++)
    {
        array_add(t1->data + i*n, t2->data, t->data + i*n, n);
    }
    t->prev1 = t1;
    t->prev2 = t2;
    t->requires_grad = t1->requires_grad == true || t2->requires_grad == true ? true : false;
    t->op = ADD_DISTRI;
    t->is_end = true;

    t1->is_end = false;
    t2->is_end = false;
    t1->num_quotes++;
    t2->num_quotes++;

    return t;
}

tensor *add_broad(tensor *t1, tensor *t2)
{
    size_t num_rows = t1->size/width(t1); //t1的行数
    // //检查t2是不是列向量
    if(t2->dim != 2 || width(t2) != 1) printf("add_broad error: t2不是列向量！t2 shape:\n"), show_shape(t2), exit(-1);
    if(t2->size != num_rows) printf("add_broad error: t1 and t2 不能以广播方式相加！\n"), exit(-1);
    
    tensor *t = create(t1->shape, t1->dim);

    size_t m=height(t), n=width(t), k=t->size/(m*n);
    for(size_t i=0; i<k; i++)
    for(size_t j=0; j<m; j++)
    {
        array_add_constant(t1->data + i*m*n + j*n, t->data + i*m*n + j*n, n, t2->data[j]);
    }

    t->prev1 = t1;
    t->prev2 = t2;
    t->requires_grad = t1->requires_grad == true || t2->requires_grad == true ? true : false;
    t->op = ADD_BROAD;
    t->is_end = true;

    t1->is_end = false;
    t2->is_end = false;
    t1->num_quotes++;
    t2->num_quotes++;
    return t;
}

tensor *div_broad(tensor *t1, tensor *t2)
{
    // size_t t1_rows = t1->size/width(t1); //t1的行数
    // //TODO: 检查t2是不是列向量
    // if(t2->size != t1_rows) printf("div_broad error: t1 and t2 不能以广播方式相除！"), exit(-1);
    
    tensor *t = create(t1->shape, t1->dim);

    size_t m=height(t), n=width(t), k=t->size/(m*n);
    for(size_t i=0; i<k; i++)
    for(size_t j=0; j<m; j++)
    {
        array_times_constant(t1->data + i*m*n + j*n, t->data + i*m*n + j*n, n, 1.0/(t2->data[j]));
    }

    t->prev1 = t1;
    t->prev2 = t2;
    t->requires_grad = t1->requires_grad == true || t2->requires_grad == true ? true : false;
    t->op = DIV_BROAD;
    t->is_end = true;

    t1->is_end = false;
    t2->is_end = false;

    t1->num_quotes++;
    t2->num_quotes++;

    return t;
}

/*矩阵乘法*/
tensor *mul(tensor *t1, tensor *t2)
{
    if(t1 == NULL || t2 == NULL)
        printf("mul error: the input tensor t1=%p or t2=%p is NULL!\n", t1, t2), exit(-1);
    if(t1->dim == 1 || t2->dim == 1)
        printf("mul error: only tensors of dim greater than 1 can be mul, but the input tensor t1 has dim %zu, and t2 has dim %zu.\n", t1->dim, t2->dim), exit(-1);
    if(t1->dim < 2)
        printf("mul error: expect the input tensor t1 dim >= 2, but got t1 dim=%zu.\n", t1->dim), exit(-1);
    if(t2->dim > 2)
        printf("mul error: only support matrix mul, the dim of input tensor t2 shouzu be equals to 2, but got t2 dim=%zu.\n", t2->dim), exit(-1);
    if(width(t1) != height(t2))
        printf("mul error: shape not match!\n"), show_shape(t1), show_shape(t2), exit(-1);

    /*计算m1与m2乘积的形状*/
    size_t *shape = malloc(t1->dim * sizeof(size_t));
    memcpy(shape, t1->shape, t1->dim * sizeof(size_t));
    shape[t1->dim - 2] = height(t1);
    shape[t1->dim - 1] = width(t2);

    tensor *t = create(shape, t1->dim); //创建用于储存计算结果的张量
    free(shape);

    // array_mul_tensor(t1->data, t2->data, t, t1->size, t2->size, t->size, )

    matrix *m1 = matrix_none();
    matrix *m2 = matrix_none();
    matrix *m = matrix_none();
    matrix_init(m2, t2->data, height(t2), width(t2));

    // void array_mul_tensor(float *A, float *B, )
    
    for(size_t i=0,m1_size = height(t1)*width(t1), m_size=height(t)*width(t); i<(t1->size / m1_size); i++)
    {
        matrix_init(m1, (t1->data + i*m1_size), height(t1), width(t1));
        matrix_init(m,  (t ->data + i*m_size),  height(t),  width(t));
        matrix_mul(m1, m2, m);
    }
    
    free(m1), free(m2), free(m);

    t->prev1 = t1;
    t->prev2 = t2;
    t->requires_grad = t1->requires_grad == true || t2->requires_grad == true ? true : false;
    t->op = MUL;
    t->is_end = true;

    t1->is_end = false;
    t2->is_end = false;
    t1->num_quotes++;
    t2->num_quotes++;
    return t;
}

tensor *transpose(tensor *x)
{
    size_t *shape_y = memcpy(malloc((x->dim) * sizeof(size_t)), x->shape, x->dim * sizeof(size_t)); //copy x shape to y shape
    size_t temp = shape_y[x->dim-2];shape_y[x->dim - 2] = shape_y[x->dim-1];shape_y[x->dim-1] = temp; //交换y的最后两维
    tensor *y = create(shape_y, x->dim);

    size_t matrix_size = height(x)*width(x);
    size_t num_matrix = x->size/matrix_size; //x中包含多少个矩阵
    for(size_t i=0; i<num_matrix; i++)
    {
        size_t offset = i*matrix_size;
        matrix *x_i = matrix_none(); matrix_init(x_i, x->data+offset, height(x), width(x));
        matrix *y_i = matrix_none(); matrix_init(y_i, y->data+offset, height(y), width(y));
        matrix_transpose(x_i, y_i);
    }

    
    free(shape_y);

    return y;
}

/*为了和math.h中的sin区分，这里使用sin_t, t表示tensor的意思*/
// tensor *sin_t(tensor *x)
// {
//     tensor *y = create(x->shape, x->dim);
//     array_sin(x->data, y->data, x->size);
//     return y;
// }

// tensor *cos_t(tensor *x)
// {
//     tensor *y = create(x->shape, x->dim);
//     array_cos(x->data, y->data, x->size);
//     return y;
// }

tensor *log_t(tensor *x)
{
    tensor *y = create(x->shape, x->dim);
    array_log(x->data, y->data, x->size);
    y->prev1 = x;
    y->prev2 = NULL;
    y->requires_grad = x->requires_grad;
    y->op = LOG;
    y->is_end = true;
    
    x->is_end = false;
    x->num_quotes++;
    return y;
}

tensor *exp_t(tensor *x)
{
    tensor *y = create(x->shape, x->dim);
    array_exp(x->data, y->data, x->size);
    y->prev1 = x;
    y->prev2 = NULL;
    y->requires_grad = x->requires_grad;
    y->op = EXP;
    y->is_end = true;
    
    x->is_end = false;
    x->num_quotes += 1;
    return y;
}

tensor *softmax(tensor *t)
{
    tensor *t_exp = exp_t(t);
    tensor *t_exp_sum = sum(t_exp, t->dim - 1);
    return div_broad(t_exp, t_exp_sum); //对最后一维做softmax
}

tensor *BCELoss(tensor *X, tensor *Y)
{
    //-(Y @ log(X).T + (1-Y) @ log(1-X).T).sum()
    return times_constant(sum_all(add(mul(Y, transpose(log_t(X))), mul(add_constant(times_constant(Y, -1), 1), transpose(log_t(add_constant(times_constant(X, -1), 1)))))), -1);
}

tensor *onehot(tensor *t1, uint32_t num_classes)
{
    if(t1 == NULL)
        printf("onehot error: the input tensor is NULL!\n"), exit(-1);
    if(t1->dim != 2 || width(t1) != 1)
        printf("onehot error: shape not match!\n"), exit(-1);
    if(array_min(t1->data, t1->size) < 0.0)
        printf("onehot error: 输入tensor的分量应该全是正数！\n"), exit(-1);
    // unsigned int t1_max = (unsigned int)array_max(t1->data, t1->size);
    if(num_classes >= UINT_MAX)
        printf("onehot error: too big!\n"), exit(-1);

    size_t shape[2] = {t1->shape[0], (size_t)num_classes};
    tensor *t = constant(shape, 2, 0.0);
    for(size_t i=0, row_size=width(t); i<t1->size; i++)
    {
        size_t offset = i*row_size;
        t->data[offset + (unsigned long)(t1->data[i])] = 1.0;
    }

    return t;
}

tensor *CrossEntropyLoss(tensor *Y_hat, tensor *Y, uint32_t num_classes)
{
    return times_constant(sum_all(times(log_t(Y_hat), onehot(Y, num_classes))), -1);
}

tensor *relu(tensor *x)
{
    tensor *y = create(x->shape, x->dim);
    array_relu(x->data, y->data, x->size);

    y->prev1 = x;
    y->prev2 = NULL;
    y->requires_grad = x->requires_grad;
    y->op = RELU;
    y->is_end = true;
    
    x->is_end = false;
    x->num_quotes++;
    return y;
}

/*通过递归的方式反向传播计算梯度*/
void backward(tensor *t)
{
    /**
     * @brief 递归停止条件:
     * 1. t==NULL: t->prev==NULL时停止
     * 2. t->op==NULL: 遇到计算图最前端的tensor时停止
     * 3. t->requires_grad == false: 一个节点不需要求梯度时停止
     */
    if(t == NULL || t->op == NONE || t->requires_grad == false || t->num_quotes != 0)
        return;

    // printf("----------------------------------------------------------------\n");

    /*初始化工作*/
    if(t->is_end == true)
    {
        if(t->size != 1)
            printf("backward error: 只支持对标量反向传播！\n"), exit(-1);
        
        if(isnan(t->data[0]) == true)
            printf("backward error: try to backward a NaN tensor!\n"), exit(-1);

        if(t->grad != NULL) warning(), free(t->grad);
        
        t->grad = (float *)malloc(1 * sizeof(float));
        *(t->grad) = 1.0;

        #ifdef DEBUG
        printf("grad init down.\n");
        #endif
    }
    /*初始化梯度*/
    if(t->prev1->requires_grad == true && t->prev1->grad == NULL)
    {
        // t->prev1->grad = constant_matrix(1,t->prev1->size, 0);
        t->prev1->grad = (float *)malloc(t->prev1->size * sizeof(float));
        array_constant(t->prev1->grad, t->prev1->size, 0.0);
        #ifdef DEBUG
        printf("prev1 grad init down!\n");
        #endif
    }
    if(t->prev2 != NULL && t->prev2->requires_grad == true && t->prev2->grad == NULL)
    {
        t->prev2->grad = (float *)malloc(t->prev2->size * sizeof(float));
        array_constant(t->prev2->grad, t->prev2->size, 0.0);
        #ifdef DEBUG
        printf("prev2 grad init down!\n");
        #endif
    }

    matrix *G = matrix_none();
    matrix_init(G, t->grad, 1, t->size);

    #ifdef DEBUG
    show_grad("t grad::", t);
    #endif

    if(t->op == MUL) //这里面算梯度的过程写的很不好看，以后得修饰一下
    {
        #ifdef DEBUG
        printf("\033[31m MUL!\033[0m\n");
        #endif
    	/**
         * 假设C = [A1,...Ak] × B, 这里A1:m×s, B:s×n, C:k×m×n
         */
        tensor *A = t->prev1;
        tensor *B = t->prev2;
        size_t m = height(A);
        size_t s = width(A);
        size_t n = width(B);
        size_t k = t->prev1->size / (m*s);
            
        /*计算prev1的梯度*/
        if(t->prev1->requires_grad == true)
        {
            for(size_t i=0; i<k*m; i++)
            for(size_t j=0; j<s; j++)
            {
                A->grad[i*s+j] += array_dot(t->grad+i*n, B->data+j*n, n);
            }

            t->prev1->num_quotes--;
        }

        /*计算prev2的梯度*/
        if(t->prev2->requires_grad == true)
        {
            matrix *grad_y_B = matrix_constant(s, n, 0.0f);
            for(size_t j=0; j<k*m; j++)
            for(size_t i=0; i<s; i++)
                array_linear(G->data+j*n, A->data[j*s+i], grad_y_B->data+i*n, grad_y_B->data+i*n, n);

            array_add(grad_y_B->data, B->grad, B->grad, B->size);
            matrix_release(grad_y_B);

            t->prev2->num_quotes--;
        }
    }else if (t->op == SUM_ALL)
    {
        if(t->prev1->requires_grad == true)
        {
            #ifdef DEBUG
            printf("\033[31m SUM_ALL!\033[0m\n");
            #endif
            /*假设Y = SUM(A) = SUM(t->prev1), A:m×n*/
            tensor *A = t->prev1;

            array_add_constant(A->grad, A->grad, A->size, G->data[0]);

            t->prev1->num_quotes--;
        }
    }else if(t->op == SUM)
    {
        if(t->prev1->requires_grad == true)
        {
            #ifdef DEBUG
            printf("\033[31m SUM!\033[0m\n");
            #endif

            size_t h=1, d=0 , t_=1; //h:size_head, t:size_tail
            size_t dim = ((size_t *)(t->prev2->data))[0]; //note: 获取个数据这么怪？参考SUM的实现

            d = t->prev1->shape[dim];
            for(size_t i=0; i<t->prev1->dim; i++)
            {
                if(i<dim)
                    h *= t->prev1->shape[i];
                else if(i>dim)
                    t_ *= t->prev1->shape[i];
            }

            printf("size_head:%zu, size_dim:%zu, size_tail:%zu\n", h,d,t_);

            matrix *U_d_T = matrix_create(1, d);
            array_constant(U_d_T->data, d, 1.0);

            matrix *I_h = matrix_unitary(h);
            matrix *I_t = matrix_unitary(t_);

            matrix *grad_Y_X_T = matrix_create(h*t_, h*d*t_);
            matrix *temp = matrix_create(h, h*d);
            kronecker_product(I_h, U_d_T, temp);
            kronecker_product(temp, I_t, grad_Y_X_T);

            matrix *grad_y_X = matrix_create(1, t->prev1->size);
            matrix_mul(G, grad_Y_X_T, grad_y_X);

            array_add(grad_y_X->data, t->prev1->grad, t->prev1->grad, t->prev1->size);

            matrix_release(U_d_T);
            matrix_release(I_h);
            matrix_release(I_t);
            matrix_release(grad_Y_X_T);
            matrix_release(temp);        
            matrix_release(grad_y_X);

            t->prev1->num_quotes--;
        }
    }else if(t->op == SUM_2D)
    {
        if(t->prev1->requires_grad == true)
        {
            
            #ifdef DEBUG
            printf("\033[31m SUM_2D!\033[0m\n");
            #endif
            size_t m=height(t->prev1), n=width(t->prev1);
            if(t->shape[0] == 1) //对列（第0维）求和
            {
                matrix *U_1_m = matrix_create(1, m);
                array_constant(U_1_m->data, m, 1.0);
                matrix *I_n = matrix_unitary(n);
                matrix *grad_Y_X_T = matrix_create(n, m*n);
                kronecker_product(U_1_m, I_n, grad_Y_X_T);

                matrix *grad_y_X = matrix_create(1, m*n);
                matrix_mul(G, grad_Y_X_T, grad_y_X);

                array_add(grad_y_X->data, t->prev1->grad, t->prev1->grad, m*n);

                matrix_release(U_1_m);
                matrix_release(I_n);
                matrix_release(grad_Y_X_T);
                matrix_release(grad_y_X);
            }else if(t->shape[1] == 1) //对行（第一维）求和
            {
                matrix *I_m = matrix_unitary(m);
                matrix *U_1_n = matrix_create(1, n);
                array_constant(U_1_n->data, n, 1.0);

                matrix *grad_Y_X_T = matrix_create(m, m*n);
                kronecker_product(I_m, U_1_n, grad_Y_X_T);
                matrix *grad_y_X = matrix_create(1, m*n);
                matrix_mul(G, grad_Y_X_T, grad_y_X);

                array_add(grad_y_X->data, t->prev1->grad, t->prev1->grad, m*n);

                matrix_release(I_m);
                matrix_release(U_1_n);
                matrix_release(grad_Y_X_T);
                matrix_release(grad_y_X);
            }else printf("backward error: SUM_2D bad shape!\n"), exit(-1);
            t->prev1->num_quotes--;
        }

    }
    else if(t->op == EXP)
    {
        if(t->prev1->requires_grad == true)
        {
            
            #ifdef DEBUG
            printf("\033[31m EXP!\033[0m\n");
            #endif
            /*假设Y=exp(X)*/
            float *grad_y_X = (float *)malloc(t->size * sizeof(float));
            array_times(G->data, t->data, grad_y_X, t->size);

            array_add(grad_y_X, t->prev1->grad, t->prev1->grad, t->size);
            // free(grad_Y_X_DIAG);
            free(grad_y_X);

            t->prev1->num_quotes--;
        }
    }else if(t->op == ADD_BROAD)
    {
        /**
         * 假设Y=A+b, 其中A:k*m*n, b:m*1, Y:k*m*n
         * 
         */
        if(t->prev1->requires_grad == true)
        {
            
            #ifdef DEBUG
            printf("\033[31m ADD_BROAD_1!\033[0m\n");
            #endif
            float *grad_y_A = G->data;
            array_add(grad_y_A, t->prev1->grad, t->prev1->grad, t->size);
            t->prev1->num_quotes--;
        }
        if(t->prev2->requires_grad == true)
        {
            #ifdef DEBUG
            printf("\033[31m ADD_BROAD_2!\033[0m\n");
            #endif
            size_t m=height(t), n=width(t), k=t->size/(m*n);
            matrix *U_k_1 = matrix_create(k, 1);
            array_constant(U_k_1->data, k, 1.0);
            matrix *U_n_1 = matrix_create(n, 1);
            array_constant(U_n_1->data, n, 1.0);
            matrix *I_m = matrix_unitary(m);
            matrix *temp = matrix_create(k*m, m);
            kronecker_product(U_k_1, I_m, temp);

            matrix *grad_Y_b_T = matrix_create(k*m*n, m);
            kronecker_product(temp, U_n_1, grad_Y_b_T);
            matrix *grad_y_b = matrix_create(1, m);
            matrix_mul(G, grad_Y_b_T, grad_y_b);

            array_add(grad_y_b->data, t->prev2->grad, t->prev2->grad, t->prev2->size);

            matrix_release(U_k_1);
            matrix_release(U_n_1);
            matrix_release(I_m);
            matrix_release(temp);
            matrix_release(grad_Y_b_T);
            matrix_release(grad_y_b);
            
            t->prev2->num_quotes--;
        }
    }else if(t->op == TIMES_C)
    {
        /*假设Y = c*X */
        if(t->prev1->requires_grad == true)
        {
            #ifdef DEBUG
            printf("\033[31m TIMES_C!\033[0m\n");
            #endif
            tensor *Y = t, *X=t->prev1;
            float c = t->prev2->data[0];
            // float *grad_Y_X_DIAG = (float *)malloc(t->size);
            // array_constant(grad_Y_X_DIAG, t->size, c);

            float *grad_y_X = (float *)malloc(X->size * sizeof(float));
            array_times_constant(G->data, grad_y_X, Y->size, c);
            array_add(grad_y_X, X->grad, X->grad, X->size);

            free(grad_y_X);

            t->prev1->num_quotes--;
        }
    }else if(t->op == DIV_BROAD)
    {
        /**
         * 假设Y = A➗b 其中A:k*m*n, b:m*1, Y:k*m*n
         * 
         */
        tensor *Y=t, *A=t->prev1, *b=t->prev2;

        size_t m=height(t), n=width(t), k=t->size/(m*n);

        if(t->prev1->requires_grad == true)
        {
            
            #ifdef DEBUG
            printf("\033[31m DIV_BROAD_1!\033[0m\n");
            #endif
            float *grad_y_A = (float *)malloc(A->size * sizeof(float));
            // float *b_i = (float *)malloc(m);

            for(size_t j=0; j<k; j++)
            for(size_t i=0; i<m; i++)
            {
                array_times_constant(G->data + j*m*n + i*n, grad_y_A + j*m*n + i*n, n, 1.0/(b->data[i]));
            }
            array_add(grad_y_A, A->grad, A->grad, A->size);
            free(grad_y_A);
            // free(b_i);

            t->prev1->num_quotes--;
        }
        if(t->prev2->requires_grad == true)
        {
            
            #ifdef DEBUG
            printf("\033[31m DIV_BROAD_2!\033[0m\n");
            #endif
            float *grad_y_b = (float *)malloc(m*sizeof(float));
            float *A_k_m = (float *)malloc(n * sizeof(float));
            for(size_t i=0; i<m; i++)
            {
                float grad_y_b_i = 0;
                for(size_t j=0; j<k; j++)
                {
                    array_times_constant(A->data + j*m*n + i*n, A_k_m, n, -1.0/(b->data[i] * b->data[i]));
                    grad_y_b_i += array_dot(G->data + j*m*n + i*n, A_k_m, n);
                }
                grad_y_b[i] = grad_y_b_i;
            }

            array_add(grad_y_b, b->grad, b->grad, m);
            free(grad_y_b);
            free(A_k_m);
            t->prev2->num_quotes--;
        }
    }else if(t->op == LOG)
    {
        if(t->prev1->requires_grad == true)
        {
            
            #ifdef DEBUG
            printf("\033[31m LOG!\033[0m\n");
            #endif
            float *grad_Y_X_DIAG = (float *)malloc(t->size * sizeof(float));
            array_div_constant(t->prev1->data, grad_Y_X_DIAG, t->size, 1.0);

            float *grad_y_X = (float *)malloc(t->prev1->size * sizeof(float));
            array_times(G->data, grad_Y_X_DIAG, grad_y_X, t->size);

            array_add(grad_y_X, t->prev1->grad, t->prev1->grad, t->size);

            free(grad_Y_X_DIAG);
            free(grad_y_X);

            t->prev1->num_quotes--;
        }
    }else if(t->op == RELU)
    {
        if(t->prev1->requires_grad == true)
        {
            
            #ifdef DEBUG
            printf("\033[31m RELU!\033[0m\n");
            #endif
            float *grad_Y_X_T_DIAG = malloc((t->size)*sizeof(float));
            for(size_t i=0; i<t->size; i++)
                grad_Y_X_T_DIAG[i] = t->data[i]>0. ? 1. : 0.;
            float *grad_y_X = malloc((t->size)*sizeof(float));
            array_times(G->data, grad_Y_X_T_DIAG, grad_y_X, t->size);

            array_add(grad_y_X, t->prev1->grad, t->prev1->grad, t->size);

            free(grad_Y_X_T_DIAG);
            free(grad_y_X);

            t->prev1->num_quotes--;
        }        
    }
    else if(t->op == TIMES)
    {
        if(t->prev1->requires_grad == true)
        {
            
            #ifdef DEBUG
            printf("\033[31m TIMES_1!\033[0m\n");
            #endif
            float *grad_Y_X_T_DIAG = t->prev2->data;
            float *grad_y_X = malloc((t->size)*sizeof(size_t));

            array_times(G->data, grad_Y_X_T_DIAG, grad_y_X, t->size);
            
            array_add(grad_y_X, t->prev1->grad, t->prev1->grad, t->size);

            free(grad_y_X);

            t->prev1->num_quotes--;
        }
        if(t->prev2->requires_grad == true)
        {
            
            #ifdef DEBUG
            printf("\033[31m TIMES_2!\033[0m\n");
            #endif
            float *grad_Y_X_T_DIAG = malloc((t->size)*sizeof(size_t));
            memcpy(grad_Y_X_T_DIAG, t->prev1->data, (t->size)*sizeof(size_t));
            float *grad_y_X = malloc((t->size)*sizeof(size_t));

            array_times(G->data, grad_Y_X_T_DIAG, grad_y_X, t->size);
            
            array_add(grad_y_X, t->prev2->grad, t->prev2->grad, t->size);

            free(grad_Y_X_T_DIAG);
            free(grad_y_X);
            t->prev2->num_quotes--;
        }
    }
    else if(t->op == ADD_DISTRI)
    {
        if(t->prev1->requires_grad == true)
        {
            array_add(G->data, t->prev1->grad, t->prev1->grad, t->prev1->size);

            t->prev1->num_quotes--;
        }

        if(t->prev2->requires_grad == true)
        {
            size_t m=height(t), n=width(t);
            matrix *U_m_1 = matrix_constant(m, 1, 1.0);
            matrix *I_n = matrix_unitary(n);
            matrix *grad_Y_X_T = matrix_create(m*n, n);
            kronecker_product(U_m_1, I_n, grad_Y_X_T);

            matrix *grad_y_X = matrix_create(1, m*n);
            matrix_mul(G, grad_Y_X_T, grad_y_X);

            array_add(grad_y_X->data, t->prev2->grad, t->prev2->grad, t->prev2->size);

            matrix_release(U_m_1);
            matrix_release(I_n);
            matrix_release(grad_Y_X_T);
            matrix_release(grad_y_X);

            t->prev2->num_quotes--;
        }
    }
    else
    {
        printf("backward error: unknow operator!\n");
        exit(-1);
    }

    free(G);

    backward(t->prev1);
    backward(t->prev2);
    
}
