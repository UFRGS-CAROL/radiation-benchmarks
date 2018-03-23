/*
 * cudaKernels.h
 *
 *  Created on: Mar 23, 2018
 *      Author: carol
 */

#ifndef CUDAKERNELS_H_
#define CUDAKERNELS_H_

void im2col_ongpu(float *im, int channels, int height, int width, int ksize,
		int stride, int pad, float *data_col);

void col2im_ongpu(float *data_col, int channels, int height, int width,
		int ksize, int stride, int pad, float *data_im);

void gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA, float *A_gpu,
		int lda, float *B_gpu, int ldb, float BETA, float *C_gpu, int ldc);

#endif /* CUDAKERNELS_H_ */
