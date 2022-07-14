/**
 * NOTE:sum函数在mnist数据集上用不到，可以不用急着测试.
 * 
 * @brife: 测试sum函数的正向和反向过程，检验其计算结果是否正确。
 * 因为sum()在传入2维tensor的时候调用的是sum_2d()处理的，所以我们这里只测试高维的tensor
 * 测试项目：
 * 1. 对一个1*2*3张量的第0维求和
 * 2. 分别对一个2*2*3张量的第0、1、2维求和
 * @date 2022-07-14
 */

#include "tensor.h"

int main(int argc, char const *argv[])
{
    size_t shape_t1[] = {1,2,3};
    size_t shape_t2[] = {2,2,3};

    tensor *t1 = range_tensor(shape_t1, 3, -0.5, 0.5);
    tensor *t2 = range_tensor(shape_t2, 3, -0.5, 0.5);

    t1->requires_grad = true;
    t2->requires_grad = true;

    return 0;
}
