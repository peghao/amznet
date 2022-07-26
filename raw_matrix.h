#ifndef __RAW_MATRIX__
#define __RAW_MATRIX__

#include <stddef.h>

struct matrix
{
    float *data;
    size_t height;
    size_t width;
};

typedef struct matrix matrix;

matrix *matrix_none();
void matrix_init(matrix *M, float *data, size_t height, size_t width);
matrix *matrix_create(size_t height, size_t width);
void matrix_release(matrix *M);
matrix *matrix_constant(size_t height, size_t width, float c);
float *matrix_index(matrix *M, size_t i, size_t j);
matrix *matrix_unitary(size_t n);
void matrix_show_raw(float *data, size_t height, size_t width);
void matrix_show(matrix *M);
void matrix_add(matrix *M1, matrix *M2, matrix *SUM);
void matrix_mul(matrix *M1, matrix *M2, matrix *M);
void matrix_transpose(matrix *M, matrix *M_T);
void kronecker_product(matrix *m1, matrix *m2, matrix *m);

#endif