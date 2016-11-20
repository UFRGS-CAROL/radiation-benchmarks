/*
 * hog.cuh
 *
 *  Created on: Aug 17, 2016
 *      Author: carol
 */

#ifndef HOG_CUH_
#define HOG_CUH_

	void set_up_constants_ext(int nbins, int block_stride_x, int block_stride_y,
			int nblocks_win_x, int nblocks_win_y);

	void compute_hists_ext(int nbins, int block_stride_x, int blovck_stride_y,
			int height, int width, const cv::gpu::PtrStepSzf& grad,
			const cv::gpu::PtrStepSzb& qangle, float sigma, float* block_hists);

	int normalize_hists_ext(int nbins, int block_stride_x, int block_stride_y,
			int height, int width, float* block_hists, float threshold);

	int classify_hists_ext(int win_height, int win_width, int block_stride_y,
			int block_stride_x, int win_stride_y, int win_stride_x, int height,
			int width, float* block_hists, float* coefs, float free_coef,
			float threshold, unsigned char* labels);

	void compute_confidence_hists_ext(int win_height, int win_width,
			int block_stride_y, int block_stride_x, int win_stride_y,
			int win_stride_x, int height, int width, float* block_hists,
			float* coefs, float free_coef, float threshold, float *confidences);

	void extract_descrs_by_rows_ext(int win_height, int win_width,
			int block_stride_y, int block_stride_x, int win_stride_y,
			int win_stride_x, int height, int width, float* block_hists,
			cv::gpu::PtrStepSzf descriptors);
	void extract_descrs_by_cols_ext(int win_height, int win_width,
			int block_stride_y, int block_stride_x, int win_stride_y,
			int win_stride_x, int height, int width, float* block_hists,
			cv::gpu::PtrStepSzf descriptors);

	void compute_gradients_8UC1_ext(int nbins, int height, int width,
			const cv::gpu::PtrStepSzb& img, float angle_scale,
			cv::gpu::PtrStepSzf grad, cv::gpu::PtrStepSzb qangle,
			bool correct_gamma);
	void compute_gradients_8UC4_ext(int nbins, int height, int width,
			const cv::gpu::PtrStepSzb& img, float angle_scale,
			cv::gpu::PtrStepSzf grad, cv::gpu::PtrStepSzb qangle,
			bool correct_gamma);

	int resize_8UC1(const cv::gpu::PtrStepSzb& src, cv::gpu::PtrStepSzb dst);
	int resize_8UC4(const cv::gpu::PtrStepSzb& src, cv::gpu::PtrStepSzb dst);


#endif /* HOG_CUH_ */
