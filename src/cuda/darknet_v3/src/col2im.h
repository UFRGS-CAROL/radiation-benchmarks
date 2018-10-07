#ifndef COL2IM_H
#define COL2IM_H

#include "type.h"

void col2im_cpu(real_t* data_col, int channels, int height, int width,
		int ksize, int stride, int pad, real_t* data_im);

#ifdef GPU
void col2im_gpu(real_t *data_col,
		int channels, int height, int width,
		int ksize, int stride, int pad, real_t *data_im,
		cudaStream_t st);
#endif
#endif
