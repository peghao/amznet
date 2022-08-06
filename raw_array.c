#ifndef __RAW_ARRAY__
#define __RAW_ARRAY__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifdef ACC_AVX
#include <immintrin.h>
#include <avx2intrin.h>
#endif

#include "raw_array.h"

void array_show(float *t, size_t size)
{
    for(size_t i=0; i<size; i++)
        printf("%f, ", t[i]);
    printf("\b \n");
}

#ifdef ACC_AVX
void array_constant_avx(float *t, size_t size, float a)
{
    // #pragma omp parallel for num_threads(4)
    // for(size_t i=0; i<size; i++)
    //     t[i] = a;

    #define BATCHSIZE 8
    size_t num_batch = size/BATCHSIZE;
    size_t mod = size % BATCHSIZE;
    
    #pragma omp parallel for num_threads(4)
    for(size_t i=0; i<num_batch; i++)
    {
        size_t offset = i*BATCHSIZE;
        __m256 t_vec = _mm256_set1_ps(a);
        _mm256_storeu_ps((t+offset), t_vec);
    }
    for(size_t i=0, offset=size-mod; i<mod; i++)
    {
        t[offset+i] = a;
    }
}
#endif

void array_constant(float *t, size_t size, float a)
{
    // #pragma omp parallel for num_threads(4)
    for(size_t i=0; i<size; i++)
        t[i] = a;
}

void array_range(float *t, size_t steps, float start, float end)
{
    float delta = (end - start)/(steps-1);
    for(size_t i=0; i<steps; i++)
        t[i] = start + i*delta;
}

void array_rand(float *t, size_t size)
{
    for(size_t i=0; i<size; i++)
        t[i] = (rand()/(float)RAND_MAX - 0.5)*0.01;
}

void array_add(float *t1, float *t2, float *t, size_t size)
{
    for(size_t i=0; i<size; i++)
        t[i] = t1[i] + t2[i];
}

void array_add_constant(float *t1, float *t, size_t size, float c)
{
    for(size_t i=0; i<size; i++)
        t[i] = t1[i] + c;
}

void array_add_broad(float *t1, float *t2, float *t, size_t num_row, size_t row_size)
{
    for(size_t i=0; i<num_row; i++){
        array_add_constant(t1+i*row_size, t+i*row_size, row_size, t2[i]);
    }
}

void array_times(float *t1, float *t2, float *t, size_t size)
{
    for(size_t i=0; i<size; i++)
        t[i] = t1[i] * t2[i];
}

void array_times_multi_threads(float *t1, float *t2, float *t, size_t size)
{
    #pragma omp parallel for num_threads(4)
    for(size_t i=0; i<size; i++)
        t[i] = t1[i] * t2[i];
}

#ifdef ACC_AVX
void array_times_avx(float const *t1, float *t2, float *t, size_t size)
{
    #define BATCHSIZE 8
    size_t num_batch = size/BATCHSIZE;
    size_t mod = size % BATCHSIZE;
    // size_t offset = 0;
    #pragma omp parallel for num_threads(4)
    for(size_t i=0; i<num_batch; i++)
    {
        size_t offset = i*BATCHSIZE;
        __m256 t1_vec = _mm256_loadu_ps((t1+offset));
        __m256 t2_vec = _mm256_loadu_ps((t2+offset));
        __m256 t_vec = _mm256_mul_ps(t1_vec, t2_vec);
        _mm256_storeu_ps((t+offset), t_vec);

    }
    for(size_t i=0, offset=size-mod; i<mod; i++)
    {
        t[offset+i] = t1[offset+i] * t2[offset+i];
    }
}
#endif

void array_times_constant_original(float *t1, float *t, size_t size, float c)
{
    #pragma omp parallel for num_threads(4)
    for(size_t i=0; i<size; i++)
        t[i] = t1[i] * c;
}

#ifdef ACC_AVX
void array_times_constant_avx(float *t1, float *t, size_t size, float c)
{
    #define BATCHSIZE 8
    size_t num_batch = size/BATCHSIZE;
    size_t mod = size % BATCHSIZE;
    __m256 a_vec = _mm256_set1_ps(c);
    #pragma omp parallel for num_threads(4)
    for(size_t i=0; i<num_batch; i++)
    {
        size_t offset = i*BATCHSIZE;
        __m256 t1_vec = _mm256_loadu_ps((t1+offset));
        __m256 t_vec = _mm256_mul_ps(t1_vec, a_vec);
        _mm256_storeu_ps((t+offset), t_vec);
    }
    for(size_t i=0, offset=size-mod; i<mod; i++)
        t[offset+i] = t1[offset+i] * c;
}
#endif

void array_times_constant(float *t1, float *t, size_t size, float c)
{
    for(size_t i=0; i<size; i++)
        t[i] = t1[i] * c;
}

void array_times_broad(float *t1, float *t2, float *t, size_t num_row, size_t row_size)
{
    for(size_t i=0; i<num_row; i++)
        array_times_constant(t1+i*row_size, t+i*row_size, row_size, t2[i]);
}

/*t = a*x + b*/
void array_linear(float *x, float a, float *b, float *t, size_t size)
{
    for(size_t i=0; i<size; i++)
        t[i] = a*x[i] + b[i];
}

float array_dot(float *t1, float *t2, size_t size)
{
    float t = 0.0;
    for(size_t i=0; i<size; i++)
        t += t1[i]*t2[i];
    return t;
}

void array_div_constant(float *t1, float *t, size_t size, float c)
{
    for(size_t i=0; i<size; i++)
        t[i] = c/t1[i];
}

void array_div_broad(float *t1, float *t2, float *t, size_t num_row, size_t row_size)
{
    for(size_t i=0; i<num_row; i++)
        array_div_constant(t1+i*row_size, t+i*row_size, row_size, t2[i]);
}

// void array_div_inverse(float *t1, float *t, size_t size)
// {
//     for(size_t i=0; i<size; i++)
//         t[i] = 1.0/t1[i];
// }

void array_sum(float *t, size_t size, float *sum)
{
    *sum = 0;
    for(size_t i=0; i<size; i++)
        *sum += t[i];
}

float array_sum_rt(float *t, size_t size)
{
    float sum = 0.0f;
    for(size_t i=0; i<size; i++)
        sum += t[i];
    return sum;
}
// void array_sin(float *x, float *y, size_t size)
// {
//     for(size_t i=0; i<size; i++)
//         y[i] = sinf(x[i]);
// }

// void array_cos(float *x, float *y, size_t size)
// {
//     for(size_t i=0; i<size; i++)
//         y[i] = cosf(x[i]);
// }

void array_log(float *x, float *y, size_t size)
{
    for(size_t i=0; i<size; i++)
        y[i] = logf(x[i]); //logf: float32的log
}

// void array_pow(float *x1, float *x2, float *y, size_t size)
// {
//     for(size_t i=0; i<size; i++)
//         y[i] = powf(x1[i], x2[i]);
// }

void array_exp(float *x, float *y, size_t size)
{
    for(size_t i=0; i<size; i++)
        y[i] = expf(x[i]);
}

float array_max(float *t1, size_t size)
{
    float max = -INFINITY;
    for(size_t i=0; i<size; i++)
        max = t1[i] > max ? t1[i] : max;
    return max;
}

float array_min(float *t1, size_t size)
{
    register float min = +INFINITY; //register变量会比普通变量快吗？
    for(size_t i=0; i<size; i++)
        min = t1[i] < min ? t1[i] : min;
    return min;
}

void array_relu(float *t1, float *t, size_t size)
{
    for(size_t i=0; i<size; i++)
        t[i] = t1[i]<=0.0 ? 0.0 : t1[i];
}

#endif