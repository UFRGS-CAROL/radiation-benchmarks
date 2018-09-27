#ifndef BLAS_H
#define BLAS_H
#include "darknet.h"
#include "type.h"

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

void axpy_gpu(int N, real_t_device ALPHA, real_t_device * X, int INCX, real_t_device * Y, int INCY);
void axpy_gpu_offset(int N, real_t_device ALPHA, real_t_device * X, int OFFX, int INCX, real_t_device * Y, int OFFY, int INCY);
void copy_gpu(int N, real_t_device * X, int INCX, real_t_device * Y, int INCY);
void copy_gpu_offset(int N, real_t_device * X, int OFFX, int INCX, real_t_device * Y, int OFFY, int INCY);
void add_gpu(int N, real_t_device ALPHA, real_t_device * X, int INCX);
void supp_gpu(int N, real_t_device ALPHA, real_t_device * X, int INCX);
void mask_gpu(int N, real_t_device * X, real_t_device mask_num, real_t_device * mask, real_t_device val);
void scale_mask_gpu(int N, real_t_device * X, real_t_device mask_num, real_t_device * mask, real_t_device scale);
void const_gpu(int N, real_t_device ALPHA, real_t_device *X, int INCX);
void pow_gpu(int N, real_t_device ALPHA, real_t_device *X, int INCX, real_t_device *Y, int INCY);
void mul_gpu(int N, real_t_device *X, int INCX, real_t_device *Y, int INCY);
void mean_gpu(real_t_device *x, int batch, int filters, int spatial, real_t_device *mean);
void variance_gpu(real_t_device *x, real_t_device *mean, int batch, int filters, int spatial, real_t_device *variance);
void normalize_gpu(real_t_device *x, real_t_device *mean, real_t_device *variance, int batch, int filters, int spatial);
void l2normalize_gpu(real_t_device *x, real_t_device *dx, int batch, int filters, int spatial);
void normalize_delta_gpu(real_t_device *x, real_t_device *mean, real_t_device *variance, real_t_device *mean_delta, real_t_device *variance_delta, int batch, int filters, int spatial, real_t_device *delta);
void fast_mean_delta_gpu(real_t_device *delta, real_t_device *variance, int batch, int filters, int spatial, real_t_device *mean_delta);
void fast_variance_delta_gpu(real_t_device *x, real_t_device *delta, real_t_device *mean, real_t_device *variance, int batch, int filters, int spatial, real_t_device *variance_delta);
void fast_variance_gpu(real_t_device *x, real_t_device *mean, int batch, int filters, int spatial, real_t_device *variance);
void fast_mean_gpu(real_t_device *x, int batch, int filters, int spatial, real_t_device *mean);
void shortcut_gpu(int batch, int w1, int h1, int c1, real_t_device *add, int w2, int h2, int c2, real_t_device s1, real_t_device s2, real_t_device *out);
void scale_bias_gpu(real_t_device *output, real_t_device *biases, int batch, int n, int size);
void backward_scale_gpu(real_t_device *x_norm, real_t_device *delta, int batch, int n, int size, real_t_device *scale_updates);
void scale_bias_gpu(real_t_device *output, real_t_device *biases, int batch, int n, int size);
void add_bias_gpu(real_t_device *output, real_t_device *biases, int batch, int n, int size);
void backward_bias_gpu(real_t_device *bias_updates, real_t_device *delta, int batch, int n, int size);
void logistic_x_ent_gpu(int n, real_t_device *pred, real_t_device *truth, real_t_device *delta, real_t_device *error);
void softmax_x_ent_gpu(int n, real_t_device *pred, real_t_device *truth, real_t_device *delta, real_t_device *error);
void smooth_l1_gpu(int n, real_t_device *pred, real_t_device *truth, real_t_device *delta, real_t_device *error);
void l2_gpu(int n, real_t_device *pred, real_t_device *truth, real_t_device *delta, real_t_device *error);
void l1_gpu(int n, real_t_device *pred, real_t_device *truth, real_t_device *delta, real_t_device *error);
void wgan_gpu(int n, real_t_device *pred, real_t_device *truth, real_t_device *delta, real_t_device *error);
void weighted_delta_gpu(real_t_device *a, real_t_device *b, real_t_device *s, real_t_device *da, real_t_device *db, real_t_device *ds, int num, real_t_device *dc);
void weighted_sum_gpu(real_t_device *a, real_t_device *b, real_t_device *s, int num, real_t_device *c);
void mult_add_into_gpu(int num, real_t_device *a, real_t_device *b, real_t_device *c);
void inter_gpu(int NX, real_t_device *X, int NY, real_t_device *Y, int B, real_t_device *OUT);
void deinter_gpu(int NX, real_t_device *X, int NY, real_t_device *Y, int B, real_t_device *OUT);
void reorg_gpu(real_t_device *x, int w, int h, int c, int batch, int stride, int forward, real_t_device *out);
void softmax_gpu(real_t_device *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, real_t_device temp, real_t_device *output);
void adam_update_gpu(real_t_device *w, real_t_device *d, real_t_device *m, real_t_device *v, real_t_device B1, real_t_device B2, real_t_device eps, real_t_device decay, real_t_device rate, int n, int batch, int t);
void adam_gpu(int n, real_t_device *x, real_t_device *m, real_t_device *v, real_t_device B1, real_t_device B2, real_t_device rate, real_t_device eps, int t);
void flatten_gpu(real_t_device *x, int spatial, int layers, int batch, int forward, real_t_device *out);
void softmax_tree(real_t_device *input, int spatial, int batch, int stride, real_t_device temp, real_t_device *output, tree hier);
void upsample_gpu(real_t_device *in, int w, int h, int c, int batch, int stride, int forward, real_t_device scale, real_t_device *out);
void constrain_gpu(int N, real_t_device ALPHA, real_t_device * X, int INCX);

#endif
#endif
