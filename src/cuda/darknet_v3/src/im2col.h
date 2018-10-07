#ifndef IM2COL_H
#define IM2COL_H

#include "type.h"

void im2col_cpu(real_t* data_im, int channels, int height, int width, int ksize,
		int stride, int pad, real_t* data_col);

#ifdef GPU

void im2col_gpu(real_t *im,
		int channels, int height, int width,
		int ksize, int stride, int pad,real_t *data_col,
		cudaStream_t st);

#endif
#endif
