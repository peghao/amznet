#include "tensor.h"

int main()
{
    size_t shape_t1[] = {2,2,3};
    size_t shape_t2[] = {3,2};
    size_t dim_t1 = 3;
    size_t dim_t2 = 2;
    tensor *t1 = range_tensor(shape_t1, dim_t1, -1, 1);
    tensor *t2 = range_tensor(shape_t2, dim_t2, -1, 1);
    
    t1->requires_grad = true;
    t2->requires_grad = true;
    
    tensor *s = sum_all(mul(t1, t2));
    show("tensor t1:", t1);
    show("tensor t2:", t2);
    show("t1✖t2:", s->prev1);
    show("tensor s:", s);

    backward(s);

    show_grad("t1 grad:", t1);
    show_grad("t2 grad:", t2);
}