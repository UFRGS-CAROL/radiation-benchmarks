#ifndef CUDA_H
#define CUDA_H

#include "darknet.h"

#ifdef GPU

void check_error(cudaError_t status);
cublasHandle_t blas_handle(unsigned char use_tensor_cores);
int *cuda_make_int_array(int *x, size_t n);
void cuda_random(real_t *x_gpu, size_t n);
real_t cuda_compare(real_t *x_gpu, real_t *x, size_t n, char *s);
dim3 cuda_gridsize(size_t n);

#ifdef CUDNN
cudnnHandle_t cudnn_handle((unsigned char use_tensor_cores));
#endif

#endif
#endif
