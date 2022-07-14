#include "tensor.h"

#define DEBUG

int main()
{
    size_t shape_1[] = {5, 1};
    size_t dim_1 = sizeof(shape_1)/sizeof(size_t);
    tensor *t1 = range_tensor(shape_1, dim_1, 0, 2);
    // t1->requires_grad = true;

    printf("tensor create down.\n");

    tensor *onehoted = onehot(t1);
    tensor *t = sum_all(onehoted);

    show(t1);
    show(onehoted);
    show(t);

    // backward(t);

    // printf("t1 grad:\n");
    // show_grad(t1);
}