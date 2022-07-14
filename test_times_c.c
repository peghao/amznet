#include "tensor.h"

#define DEBUG

int main()
{
    size_t shape_1[] = {2,2,3};
    size_t dim_1 = sizeof(shape_1)/sizeof(size_t);

    tensor *t1 = range_tensor(shape_1, dim_1, -1, 1);
    
    t1->requires_grad = true;

    
    tensor *t = sum_all(times_constant(t1, 2));

    show("tensor t1:", t1);
    show("2*t1:", t->prev1);

    backward(t);

    show_grad("t1 grad:", t1);
}