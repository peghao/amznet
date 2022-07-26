#ifndef __RAW_ARRAY__
#define __RAW_ARRAY__

void array_show(float *t, size_t size);
void array_constant(float *t, size_t size, float a);
void array_range(float *t, size_t steps, float start, float end);
void array_rand(float *t, size_t size);
void array_add(float *t1, float *t2, float *t, size_t size);
void array_add_constant(float *t1, float *t, size_t size, float c);
void array_add_broad(float *t1, float *t2, float *t, size_t num_row, size_t row_size);
void array_times(float *t1, float *t2, float *t, size_t size);
void array_times_constant(float *t1, float *t, size_t size, float c);
void array_times_broad(float *t1, float *t2, float *t, size_t num_row, size_t row_size);
void array_linear(float *x, float a, float *b, float *t, size_t size);
float array_dot(float *t1, float *t2, size_t size);
void array_div_constant(float *t1, float *t, size_t size, float c);
void array_div_broad(float *t1, float *t2, float *t, size_t num_row, size_t row_size);
void array_sum(float *t, size_t size, float *sum);
void array_log(float *x, float *y, size_t size);
void array_exp(float *x, float *y, size_t size);
float array_max(float *t1, size_t size);
float array_min(float *t1, size_t size);
void array_relu(float *t1, float *t, size_t size);

#endif