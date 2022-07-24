#include "tensor.h"

int main()
{
    size_t shape[] = {2,3};
    tensor *t1 = constant(shape, 2, 0.5);

    tensor *t = softmax(t1);

    show("t:", t);

    release(t);
}