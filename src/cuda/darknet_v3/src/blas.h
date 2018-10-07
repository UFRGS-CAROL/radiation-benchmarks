#ifndef BLAS_H
#define BLAS_H
#include "darknet.h"

void flatten(real_t *x, int size, int layers, int batch, int forward);
void pm(int M, int N, real_t *A);
real_t *random_matrix(int rows, int cols);
void time_random_matrix(int TA, int TB, int m, int k, int n);
void reorg_cpu(real_t *x, int w, int h, int c, int batch, int stride,
		int forward, real_t *out);

void test_blas();

void inter_cpu(int NX, real_t *X, int NY, real_t *Y, int B, real_t *OUT);
void deinter_cpu(int NX, real_t *X, int NY, real_t *Y, int B, real_t *OUT);
void mult_add_into_cpu(int N, real_t *X, real_t *Y, real_t *Z);

void const_cpu(int N, real_t ALPHA, real_t *X, int INCX);
void pow_cpu(int N, real_t ALPHA, real_t *X, int INCX, real_t *Y, int INCY);
void mul_cpu(int N, real_t *X, int INCX, real_t *Y, int INCY);

int test_gpu_blas();
void shortcut_cpu(int batch, int w1, int h1, int c1, real_t *add, int w2,
		int h2, int c2, real_t s1, real_t s2, real_t *out);

void mean_cpu(real_t *x, int batch, int filters, int spatial, real_t *mean);
void variance_cpu(real_t *x, real_t *mean, int batch, int filters, int spatial,
		real_t *variance);

void scale_bias(real_t *output, real_t *scales, int batch, int n, int size);
void backward_scale_cpu(real_t *x_norm, real_t *delta, int batch, int n,
		int size, real_t *scale_updates);
void mean_delta_cpu(real_t *delta, real_t *variance, int batch, int filters,
		int spatial, real_t *mean_delta);
void variance_delta_cpu(real_t *x, real_t *delta, real_t *mean,
		real_t *variance, int batch, int filters, int spatial,
		real_t *variance_delta);
void normalize_delta_cpu(real_t *x, real_t *mean, real_t *variance,
		real_t *mean_delta, real_t *variance_delta, int batch, int filters,
		int spatial, real_t *delta);
void l2normalize_cpu(real_t *x, real_t *dx, int batch, int filters,
		int spatial);

void smooth_l1_cpu(int n, real_t *pred, real_t *truth, real_t *delta,
		real_t *error);
void l2_cpu(int n, real_t *pred, real_t *truth, real_t *delta, real_t *error);
void l1_cpu(int n, real_t *pred, real_t *truth, real_t *delta, real_t *error);
void logistic_x_ent_cpu(int n, real_t *pred, real_t *truth, real_t *delta,
		real_t *error);
void softmax_x_ent_cpu(int n, real_t *pred, real_t *truth, real_t *delta,
		real_t *error);
void weighted_sum_cpu(real_t *a, real_t *b, real_t *s, int num, real_t *c);
void weighted_delta_cpu(real_t *a, real_t *b, real_t *s, real_t *da, real_t *db,
		real_t *ds, int n, real_t *dc);

void softmax(real_t *input, int n, real_t temp, int stride, real_t *output);
void softmax_cpu(real_t *input, int n, int batch, int batch_offset, int groups,
		int group_offset, int stride, real_t temp, real_t *output);
void upsample_cpu(real_t *in, int w, int h, int c, int batch, int stride,
		int forward, real_t scale, real_t *out);

#ifdef GPU
#include "cuda.h"
#include "tree.h"
void constrain_gpu(int N, real_t ALPHA, real_t * X, int INCX, cudaStream_t st);

void axpy_gpu(int N, real_t ALPHA, real_t * X, int INCX, real_t * Y, int INCY,
		cudaStream_t st);
void axpy_gpu_offset(int N, real_t ALPHA, real_t * X, int OFFX, int INCX, real_t * Y, int OFFY, int INCY,
		cudaStream_t st);
void copy_gpu(int N, real_t * X, int INCX, real_t * Y, int INCY,
		cudaStream_t st);
void copy_gpu_offset(int N, real_t * X, int OFFX, int INCX, real_t * Y, int OFFY, int INCY,
		cudaStream_t st);
void add_gpu(int N, real_t ALPHA, real_t * X, int INCX,
		cudaStream_t st);
void supp_gpu(int N, real_t ALPHA, real_t * X, int INCX,
		cudaStream_t st);
void mask_gpu(int N, real_t * X, real_t mask_num, real_t * mask, real_t val,
		cudaStream_t st);
void scale_mask_gpu(int N, real_t * X, real_t mask_num, real_t * mask, real_t scale,
		cudaStream_t st);
void const_gpu(int N, real_t ALPHA, real_t *X, int INCX,
		cudaStream_t st);
void pow_gpu(int N, real_t ALPHA, real_t *X, int INCX, real_t *Y, int INCY,
		cudaStream_t st);
void mul_gpu(int N, real_t *X, int INCX, real_t *Y, int INCY,
		cudaStream_t st);

void mean_gpu(real_t *x, int batch, int filters, int spatial, real_t *mean,
		cudaStream_t st);
void variance_gpu(real_t *x, real_t *mean, int batch, int filters, int spatial, real_t *variance,
		cudaStream_t st);
void normalize_gpu(real_t *x, real_t *mean, real_t *variance, int batch, int filters, int spatial,
		cudaStream_t st);
void l2normalize_gpu(real_t *x, real_t *dx, int batch, int filters, int spatial,
		cudaStream_t st);

void normalize_delta_gpu(real_t *x, real_t *mean, real_t *variance, real_t *mean_delta, real_t *variance_delta, int batch, int filters, int spatial, real_t *delta,
		cudaStream_t st);

void fast_mean_delta_gpu(real_t *delta, real_t *variance, int batch, int filters, int spatial, real_t *mean_delta,
		cudaStream_t st);
void fast_variance_delta_gpu(real_t *x, real_t *delta, real_t *mean, real_t *variance, int batch, int filters, int spatial, real_t *variance_delta,
		cudaStream_t st);

void fast_variance_gpu(real_t *x, real_t *mean, int batch, int filters, int spatial, real_t *variance,
		cudaStream_t st);
void fast_mean_gpu(real_t *x, int batch, int filters, int spatial, real_t *mean,
		cudaStream_t st);
void shortcut_gpu(int batch, int w1, int h1, int c1, real_t *add, int w2, int h2, int c2, real_t s1, real_t s2, real_t *out,
		cudaStream_t st);
void scale_bias_gpu(real_t *output, real_t *biases, int batch, int n, int size,
		cudaStream_t st);
void backward_scale_gpu(real_t *x_norm, real_t *delta, int batch, int n, int size, real_t *scale_updates,
		cudaStream_t st);
void scale_bias_gpu(real_t *output, real_t *biases, int batch, int n, int size,
		cudaStream_t st);
void add_bias_gpu(real_t *output, real_t *biases, int batch, int n, int size,
		cudaStream_t st);
void backward_bias_gpu(real_t *bias_updates, real_t *delta, int batch, int n, int size,
		cudaStream_t st);

void logistic_x_ent_gpu(int n, real_t *pred, real_t *truth, real_t *delta, real_t *error,
		cudaStream_t st);
void softmax_x_ent_gpu(int n, real_t *pred, real_t *truth, real_t *delta, real_t *error,
		cudaStream_t st);
void smooth_l1_gpu(int n, real_t *pred, real_t *truth, real_t *delta, real_t *error,
		cudaStream_t st);
void l2_gpu(int n, real_t *pred, real_t *truth, real_t *delta, real_t *error,
		cudaStream_t st);
void l1_gpu(int n, real_t *pred, real_t *truth, real_t *delta, real_t *error,
		cudaStream_t st);
void wgan_gpu(int n, real_t *pred, real_t *truth, real_t *delta, real_t *error,
		cudaStream_t st);
void weighted_delta_gpu(real_t *a, real_t *b, real_t *s, real_t *da, real_t *db, real_t *ds, int num, real_t *dc,
		cudaStream_t st);
void weighted_sum_gpu(real_t *a, real_t *b, real_t *s, int num, real_t *c,
		cudaStream_t st);
void mult_add_into_gpu(int num, real_t *a, real_t *b, real_t *c,
		cudaStream_t st);
void inter_gpu(int NX, real_t *X, int NY, real_t *Y, int B, real_t *OUT,
		cudaStream_t st);
void deinter_gpu(int NX, real_t *X, int NY, real_t *Y, int B, real_t *OUT,
		cudaStream_t st);

void reorg_gpu(real_t *x, int w, int h, int c, int batch, int stride, int forward, real_t *out,
		cudaStream_t st);

void softmax_gpu(real_t *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, real_t temp, real_t *output,
		cudaStream_t st);
void adam_update_gpu(real_t *w, real_t *d, real_t *m, real_t *v, real_t B1, real_t B2, real_t eps, real_t decay, real_t rate, int n, int batch, int t,
		cudaStream_t st);
void adam_gpu(int n, real_t *x, real_t *m, real_t *v, real_t B1, real_t B2, real_t rate, real_t eps, int t,
		cudaStream_t st);

void flatten_gpu(real_t *x, int spatial, int layers, int batch, int forward, real_t *out,
		cudaStream_t st);
void softmax_tree(real_t *input, int spatial, int batch, int stride, real_t temp, real_t *output, tree hier,
		cudaStream_t st);
void upsample_gpu(real_t *in, int w, int h, int c, int batch, int stride, int forward, real_t scale, real_t *out,
		cudaStream_t st);

#endif
#endif
