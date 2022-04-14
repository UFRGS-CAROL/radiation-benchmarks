//headers hog.cu

#ifndef HOG_H
#define HOG_H

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/gpumat.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "opencv2/gpu/device/common.hpp"
#include "opencv2/gpu/device/reduce.hpp"

#include "opencv2/gpu/device/functional.hpp"
#include "opencv2/gpu/device/warp_shuffle.hpp"


#include <vector>
#include <memory>
#include <iosfwd>

using namespace cv;
using namespace gpu;
using namespace device;

void set_up_constants(int nbins, int block_stride_x, int block_stride_y,
					  int nblocks_win_x, int nblocks_win_y);

void compute_hists(int nbins, int block_stride_x, int blovck_stride_y,
				   int height, int width, const cv::gpu::PtrStepSzf& grad,
				   const cv::gpu::PtrStepSzb& qangle, float sigma, float* block_hists);

void normalize_hists(int nbins, int block_stride_x, int block_stride_y,
					 int height, int width, float* block_hists, float threshold);

void classify_hists(int win_height, int win_width, int block_stride_y,
					int block_stride_x, int win_stride_y, int win_stride_x, int height,
					int width, float* block_hists, float* coefs, float free_coef,
					float threshold, unsigned char* labels);

void compute_confidence_hists(int win_height, int win_width, int block_stride_y, int block_stride_x,
				   int win_stride_y, int win_stride_x, int height, int width, float* block_hists,
				   float* coefs, float free_coef, float threshold, float *confidences);

void extract_descrs_by_rows(int win_height, int win_width, int block_stride_y, int block_stride_x,
							int win_stride_y, int win_stride_x, int height, int width, float* block_hists,
							cv::gpu::PtrStepSzf descriptors);
void extract_descrs_by_cols(int win_height, int win_width, int block_stride_y, int block_stride_x,
							int win_stride_y, int win_stride_x, int height, int width, float* block_hists,
							cv::gpu::PtrStepSzf descriptors);

void compute_gradients_8UC1(int nbins, int height, int width, const cv::gpu::PtrStepSzb& img,
							float angle_scale, cv::gpu::PtrStepSzf grad, cv::gpu::PtrStepSzb qangle, bool correct_gamma);
void compute_gradients_8UC4(int nbins, int height, int width, const cv::gpu::PtrStepSzb& img,
							float angle_scale, cv::gpu::PtrStepSzf grad, cv::gpu::PtrStepSzb qangle, bool correct_gamma);

void resize_8UC1(const cv::gpu::PtrStepSzb& src, cv::gpu::PtrStepSzb dst);
void resize_8UC4(const cv::gpu::PtrStepSzb& src, cv::gpu::PtrStepSzb dst);

#endif
