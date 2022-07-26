#include <stdio.h>
#include <stdlib.h>

#include "raw_array.h"
#include "raw_matrix.h"

matrix *matrix_none()
{
    matrix *M = (matrix *)malloc(sizeof(matrix));
    M->data = NULL;
    M->height = 0;
    M->width = 0;
    return M;
}

void matrix_init(matrix *M, float *data, size_t height, size_t width)
{
    M->data = data;
    M->height = height;
    M->width = width;
}

matrix *matrix_create(size_t height, size_t width)
{
    matrix *M = matrix_none();
    matrix_init(M, (float *)malloc(height*width*sizeof(float)), height, width);
    return M;
}

void matrix_release(matrix *M)
{
    free(M->data);
    free(M);
}

matrix *matrix_constant(size_t height, size_t width, float c)
{
    matrix *M = matrix_create(height, width);
    array_constant(M->data, height*width, c);
    return M;
}

/*这个函数可以改成宏定义或者inline函数，可能会提升速度*/
float *matrix_index(matrix *M, size_t i, size_t j)
{
    return &(M->data[(M->width)*i + j]);
}

matrix *matrix_unitary(size_t n)
{
    matrix *I = matrix_create(n, n);
    array_constant(I->data, n*n, 0.0);
    for(size_t i=0; i<n; i++)
        *matrix_index(I, i, i) = 1.0;
    return I;
}

void matrix_show_raw(float *data, size_t height, size_t width)
{
    matrix *m = matrix_none();
    matrix_init(m, data, height, width);
    for(size_t i=0; i<height; i++)
    {
        for(size_t j=0; j<width; j++)
            printf("%12.6f,", *matrix_index(m, i,j));
        printf("\b \n");
    }
    free(m);
}

void matrix_show(matrix *M)
{
    printf("matrix:%p, height:%zu, width:%zu:\n", M, M->height, M->width);
    matrix_show_raw(M->data, M->height, M->width);
}

void matrix_add(matrix *M1, matrix *M2, matrix *SUM)
{
    if(M1->height != M2->height || M1->width != M2->width)
        printf("matrix add error: /*TODO*/\n"), exit(-1);
    array_add(M1->data, M2->data, SUM->data, M1->height * M1->width);
}

void matrix_mul(matrix *M1, matrix *M2, matrix *M)
{
    /*假设M1是m×n矩阵， M2是n×s矩阵*/
    size_t m=M1->height, n=M1->width, s=M2->width;
    // struct matrix *M = matrix_create(m, s);
    for(size_t i=0; i<m; i++)
    for(size_t j=0; j<s; j++)
    {
        float sum = 0;
        for(size_t k=0; k<n; k++)
            sum += (*matrix_index(M1, i, k)) * (*matrix_index(M2, k, j));
        *matrix_index(M, i, j) = sum;
    }
}

void matrix_transpose(matrix *M, matrix *M_T)
{
    for(size_t i=0; i<M_T->height; i++)
    for(size_t j=0; j<M_T->width; j++)
        *matrix_index(M_T, i, j) = *matrix_index(M, j, i);
}

void  kronecker_product(matrix *m1, matrix *m2, matrix *m)
{
    for(size_t i=0; i<m1->height; i++)
    for(size_t j=0; j<m1->width; j++)
    for(size_t p=0; p<m2->height; p++)
    for(size_t q=0; q<m2->width; q++)
        *matrix_index(m, i*m2->height + p, j*m2->width + q) = *matrix_index(m1, i, j) * *matrix_index(m2, p, q);
}