#ifndef BLAS_H
#define BLAS_H

#ifdef GPU
#include "cuda.h"
#endif

void reorg(float *x, int size, int layers, int batch, int forward);
void pm(int M, int N, float *A);
float *random_matrix(int rows, int cols);
void time_random_matrix(int TA, int TB, int m, int k, int n);

void test_blas();

void const_cpu(int N, float ALPHA, float *X, int INCX);
void constrain_ongpu(int N, float ALPHA, float * X, int INCX, cudaStream_t stream);
void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY);

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
void test_gpu_blas();
void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2,
		int c2, float *out);

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial,
		float *variance);
void normalize_cpu(float *x, float *mean, float *variance, int batch,
		int filters, int spatial);

void scale_bias(float *output, float *scales, int batch, int n, int size);
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size,
		float *scale_updates);
void mean_delta_cpu(float *delta, float *variance, int batch, int filters,
		int spatial, float *mean_delta);
void variance_delta_cpu(float *x, float *delta, float *mean, float *variance,
		int batch, int filters, int spatial, float *variance_delta);
void normalize_delta_cpu(float *x, float *mean, float *variance,
		float *mean_delta, float *variance_delta, int batch, int filters,
		int spatial, float *delta);

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta,
		float *error);
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error);
void weighted_sum_cpu(float *a, float *b, float *s, int num, float *c);

#ifdef GPU
void axpy_ongpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY, cudaStream_t stream);
void axpy_ongpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY, cudaStream_t stream);
void copy_ongpu(int N, float * X, int INCX, float * Y, int INCY, cudaStream_t stream);
void copy_ongpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY, cudaStream_t stream);
void scal_ongpu(int N, float ALPHA, float * X, int INCX, cudaStream_t stream);
void supp_ongpu(int N, float ALPHA, float * X, int INCX, cudaStream_t stream);
void mask_ongpu(int N, float * X, float mask_num, float * mask, cudaStream_t stream);
void const_ongpu(int N, float ALPHA, float *X, int INCX, cudaStream_t stream);
void pow_ongpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY, cudaStream_t stream);
void mul_ongpu(int N, float *X, int INCX, float *Y, int INCY, cudaStream_t stream);
void fill_ongpu(int N, float ALPHA, float * X, int INCX, cudaStream_t stream);

void mean_gpu(float *x, int batch, int filters, int spatial, float *mean, cudaStream_t stream);
void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance, cudaStream_t stream);
void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial, cudaStream_t stream);

void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta, cudaStream_t stream);

void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta, cudaStream_t stream);
void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta, cudaStream_t stream);

void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance, cudaStream_t stream);
void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean, cudaStream_t stream);
void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float *out, cudaStream_t stream);
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size, cudaStream_t stream);
void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates, cudaStream_t stream);
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size, cudaStream_t stream);
void add_bias_gpu(float *output, float *biases, int batch, int n, int size, cudaStream_t stream);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size, cudaStream_t stream);

void smooth_l1_gpu(int n, float *pred, float *truth, float *delta, float *error, cudaStream_t stream);
void l2_gpu(int n, float *pred, float *truth, float *delta, float *error, cudaStream_t stream);
void weighted_delta_gpu(float *a, float *b, float *s, float *da, float *db, float *ds, int num, float *dc, cudaStream_t stream);
void weighted_sum_gpu(float *a, float *b, float *s, int num, float *c, cudaStream_t stream);
void mult_add_into_gpu(int num, float *a, float *b, float *c, cudaStream_t stream);

void reorg_ongpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out, cudaStream_t stream);

#endif
#endif
