/**
 * @brife: 测试sum_2d函数的正向和反向过程，检验其计算结果是否正确
 * 测试项目，分别对以下矩阵的第0维和第1维求和，并反向计算梯度：
 * 1. 1*1矩阵
 * 2. 1*n矩阵
 * 3. m*1矩阵
 * 4. m*n矩阵
 * 
 * @date 2022-07-13
 */

#include "tensor.h"

int main(int argc, char const *argv[])
{
    size_t shape_t1[] = {1,1};
    size_t shape_t2[] = {1,4};
    size_t shape_t3[] = {3,1};
    size_t shape_t4[] = {3,4};

    tensor *t1 = create(shape_t1, 2); t1->data[0] = 1;
    tensor *t2 = range_tensor(shape_t2, 2, -0.5, 0.5);
    tensor *t3 = range_tensor(shape_t3, 2, -0.5, 0.5);
    tensor *t4 = range_tensor(shape_t4, 2, -0.5, 0.5);

    t1->requires_grad = true;
    t2->requires_grad = true;
    t3->requires_grad = true;
    t4->requires_grad = true;

    tensor *s1_1 = sum_2d(t1,0);
    tensor *s2_1 = sum_2d(t2,0);
    tensor *s3_1 = sum_2d(t3,0);
    tensor *s4_1 = sum_2d(t4,0);
    tensor *s1_2 = sum_2d(t1,1);
    tensor *s2_2 = sum_2d(t2,1);
    tensor *s3_2 = sum_2d(t3,1);
    tensor *s4_2 = sum_2d(t4,1);


    show("tensor t1:",t1);
    show("tensor t2:",t2);
    show("tensor t3:",t3);
    show("tensor t4:",t4);

    backward(sum_all(s1_1));
    backward(sum_all(s2_1));
    backward(sum_all(s3_1));
    backward(sum_all(s4_1));
    
    backward(sum_all(s1_2));
    backward(sum_all(s2_2));
    backward(sum_all(s3_2));
    backward(sum_all(s4_2));

    show_grad("t1 grad:", t1);
    show_grad("t2 grad:", t2);
    show_grad("t3 grad:", t3);
    show_grad("t4 grad:", t4);

    return 0;
}
