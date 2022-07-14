/**
 * @brife: 测试sum_all函数的正向和反向过程，检验其计算结果是否正确
 * 测试项目：
 * 1. 1*1矩阵
 * 2. 1*n矩阵
 * 3. m*1矩阵
 * 4. m*n矩阵
 * 5. k*m*n张量
 * 
 * @date 2022-07-12
 */

#include "tensor.h"

int main(int argc, char const *argv[])
{
    size_t shape_t1[] = {1,1};
    size_t shape_t2[] = {1,4};
    size_t shape_t3[] = {3,1};
    size_t shape_t4[] = {3,4};
    size_t shape_t5[] = {2,3,4};

    tensor *t1 = create(shape_t1, 2); t1->data[0] = 1;
    tensor *t2 = range_tensor(shape_t2, 2, -0.5, 0.5);
    tensor *t3 = range_tensor(shape_t3, 2, -0.5, 0.5);
    tensor *t4 = range_tensor(shape_t4, 2, -0.5, 0.5);
    tensor *t5 = range_tensor(shape_t5, 3, -0.5, 0.5);

    t1->requires_grad = true;
    t2->requires_grad = true;
    t3->requires_grad = true;
    t4->requires_grad = true;
    t5->requires_grad = true;

    tensor *s1 = sum_all(t1);
    tensor *s2 = sum_all(t2);
    tensor *s3 = sum_all(t3);
    tensor *s4 = sum_all(t4);
    tensor *s5 = sum_all(t5);


    show("tensor t1:",t1);
    show("tensor t2:",t2);
    show("tensor t3:",t3);
    show("tensor t4:",t4);
    show("tensor t5:",t5);

    backward(s1);
    backward(s2);
    backward(s3);
    backward(s4);
    backward(s5);

    show_grad("t1 grad:", t1);
    show_grad("t2 grad:", t2);
    show_grad("t3 grad:", t3);
    show_grad("t4 grad:", t4);
    show_grad("t5 grad:", t5);
    return 0;
}
