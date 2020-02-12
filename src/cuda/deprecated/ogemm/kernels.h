/*
 * kernels.h
 *
 *  Created on: 05/10/2018
 *      Author: fernando
 */

#ifndef KERNELS_H_
#define KERNELS_H_


#include <cuda_fp16.h>


// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

using namespace nvcuda;

__device__ __forceinline__ void fma__(const double a, const double b,
    float& c) {
  c = __fmaf_rn(__double2float_rn(a), __double2float_rn(b), c);
}

__device__ __forceinline__ void fma__(const float a, const float b,
    float& c) {
  c = __fmaf_rn(a, b, c);
}


__device__ __forceinline__ void fma__(const double a, const double b,
    double& c) {
  c = __fma_rn(a, b, c);
}

template<typename real_t, typename half_real_t>
__device__ void saxpy(real_t a, real_t *b, real_t *c, half_real_t *c_inc) {
  fma__(a, b[0], c[0]);
  fma__(a, b[1], c[1]);
  fma__(a, b[2], c[2]);
  fma__(a, b[3], c[3]);
  fma__(a, b[4], c[4]);
  fma__(a, b[5], c[5]);
  fma__(a, b[6], c[6]);
  fma__(a, b[7], c[7]);
  fma__(a, b[8], c[8]);
  fma__(a, b[9], c[9]);
  fma__(a, b[10], c[10]);
  fma__(a, b[11], c[11]);
  fma__(a, b[12], c[12]);
  fma__(a, b[13], c[13]);
  fma__(a, b[14], c[14]);
  fma__(a, b[15], c[15]);

  fma__(a, b[0], c_inc[0]);
  fma__(a, b[1], c_inc[1]);
  fma__(a, b[2], c_inc[2]);
  fma__(a, b[3], c_inc[3]);
  fma__(a, b[4], c_inc[4]);
  fma__(a, b[5], c_inc[5]);
  fma__(a, b[6], c_inc[6]);
  fma__(a, b[7], c_inc[7]);
  fma__(a, b[8], c_inc[8]);
  fma__(a, b[9], c_inc[9]);
  fma__(a, b[10], c_inc[10]);
  fma__(a, b[11], c_inc[11]);
  fma__(a, b[12], c_inc[12]);
  fma__(a, b[13], c_inc[13]);
  fma__(a, b[14], c_inc[14]);
  fma__(a, b[15], c_inc[15]);
}

//DMR KERNEL
template<typename real_t, typename half_real_t>
__global__ void sgemm_kernel(half_real_t* C_inc, real_t *C, const real_t *A, const real_t *B, int m,
    int n, int k, int lda, int ldb, int ldc, real_t alpha, real_t beta) {

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int ibx = blockIdx.x * 64;
  const int iby = blockIdx.y * 16;

  const int idt = ty * 16 + tx;

  B += tx + __mul24(iby + ty, ldb);
  A += ibx + idt;
  C += ibx + idt + __mul24(iby, ldc);

  C_inc += ibx + idt + __mul24(iby, ldc);

  const real_t *Bend = B + k;

  real_t Cb[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  half_real_t Cb_inc[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

  m = 2 * lda;
  n = 3 * lda;

  do {
    //float Ab[4] = {A[0], A[lda], A[2*lda], A[3*lda]};
    real_t Ab[4] = { A[0], A[lda], A[m], A[n] };
    __shared__ real_t Bb[16][17];
    Bb[tx][ty + 0] = B[0];
    Bb[tx][ty + 4] = B[4 * ldb];
    Bb[tx][ty + 8] = B[8 * ldb];
    Bb[tx][ty + 12] = B[12 * ldb];

    __syncthreads();

    A += 4 * lda;
    saxpy(Ab[0], &Bb[0][0], Cb, Cb_inc);

    Ab[0] = A[0];
    saxpy(Ab[1], &Bb[1][0], Cb, Cb_inc);

    Ab[1] = A[lda];
    saxpy(Ab[2], &Bb[2][0], Cb, Cb_inc);

    Ab[2] = A[m];
    saxpy(Ab[3], &Bb[3][0], Cb, Cb_inc);
    Ab[3] = A[n];

    A += 4 * lda;
    saxpy(Ab[0], &Bb[4][0], Cb, Cb_inc);

    Ab[0] = A[0];
    saxpy(Ab[1], &Bb[5][0], Cb, Cb_inc);

    Ab[1] = A[lda];
    saxpy(Ab[2], &Bb[6][0], Cb, Cb_inc);

    Ab[2] = A[m];
    saxpy(Ab[3], &Bb[7][0], Cb, Cb_inc);

    Ab[3] = A[n];

    A += 4 * lda;
    saxpy(Ab[0], &Bb[8][0], Cb, Cb_inc);

    Ab[0] = A[0];
    saxpy(Ab[1], &Bb[9][0], Cb, Cb_inc);

    Ab[1] = A[lda];
    saxpy(Ab[2], &Bb[10][0], Cb, Cb_inc);

    Ab[2] = A[m];
    saxpy(Ab[3], &Bb[11][0], Cb, Cb_inc);

    Ab[3] = A[n];

    A += 4 * lda;
    saxpy(Ab[0], &Bb[12][0], Cb, Cb_inc);

    saxpy(Ab[1], &Bb[13][0], Cb, Cb_inc);

    saxpy(Ab[2], &Bb[14][0], Cb, Cb_inc);

    saxpy(Ab[3], &Bb[15][0], Cb, Cb_inc);

    B += 16;

    __syncthreads();
  } while (B < Bend);

  half_real_t alpha_inc = half_real_t(alpha);
  half_real_t beta_inc = half_real_t(beta);
#pragma unroll 16
  for (int i = 0; i < 16; i++, C += ldc, C_inc += ldc) {
    C[0] = alpha * Cb[i] + beta * C[0];
    C_inc[0] =  alpha_inc * Cb_inc[i] + beta_inc * C_inc[0];
  }
}





#endif /* KERNELS_H_ */
