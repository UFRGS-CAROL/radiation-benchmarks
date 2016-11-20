/*
 * HogDescriptor.h
 *
 *  Created on: Aug 17, 2016
 *      Author: carol
 */

#ifndef HOGDESCRIPTOR_H_
#define HOGDESCRIPTOR_H_
#include "opencv2/gpu/gpu.hpp"


using namespace std;
class HOGConfidence{
public:
   double scale;
   vector<cv::Point> locations;
   vector<double> confidences;
   vector<double> part_scores[4]; 
};

class HogDescriptor {
protected:
	void computeBlockHistograms(const cv::gpu::GpuMat& img);
	void computeGradient(const  cv::gpu::GpuMat& img, cv::gpu::GpuMat& grad, cv::gpu::GpuMat& qangle);

	double getWinSigma() const;
	bool checkDetectorSize() const;

	static int numPartsWithin(int size, int part_size, int stride);
	static cv::Size numPartsWithin(cv::Size size, cv::Size part_size, cv::Size stride);

	// Coefficients of the separating plane
	float free_coef;
	cv::gpu::GpuMat detector;

	// Results of the last classification step
	cv::gpu::GpuMat labels, labels_buf;
	cv::Mat labels_host;

	// Results of the last histogram evaluation step
	cv::gpu::GpuMat block_hists, block_hists_buf;

	// Gradients conputation results
	cv::gpu::GpuMat grad, qangle, grad_buf, qangle_buf;

	// returns subbuffer with required size, reallocates buffer if nessesary.
	static cv::gpu::GpuMat getBuffer(const cv::Size& sz, int type, cv::gpu::GpuMat& buf);
	static cv::gpu::GpuMat getBuffer(int rows, int cols, int type, cv::gpu::GpuMat& buf);

	std::vector<cv::gpu::GpuMat> image_scales;

public:
    enum { DEFAULT_WIN_SIGMA = -1 };
    enum { DEFAULT_NLEVELS = 64 };
    enum { DESCR_FORMAT_ROW_BY_ROW, DESCR_FORMAT_COL_BY_COL };
	cv::Size win_size;
	cv::Size block_size;
	cv::Size block_stride;
	cv::Size cell_size;
	int nbins;
	double win_sigma;
	double threshold_L2hys;
	bool gamma_correction;
	int nlevels;
	HogDescriptor(cv::Size win_size = cv::Size(64, 128), cv::Size block_size = cv::Size(16, 16),
			cv::Size block_stride = cv::Size(8, 8), cv::Size cell_size = cv::Size(8, 8),
			int nbins = 9, double win_sigma = DEFAULT_WIN_SIGMA,
			double threshold_L2hys = 0.2, bool gamma_correction = true,
			int nlevels = DEFAULT_NLEVELS);
	size_t getDescriptorSize() const;
	size_t getBlockHistogramSize() const;

	void setSVMDetector(const vector<float>& detector);

	static vector<float> getDefaultPeopleDetector();
	static vector<float> getPeopleDetector48x96();
	static vector<float> getPeopleDetector64x128();

	void detect(const cv::gpu::GpuMat& img, vector<cv::Point>& found_locations,
			double hit_threshold = 0, cv::Size win_stride = cv::Size(), cv::Size padding =
					cv::Size());

	void detectMultiScale(const cv::gpu::GpuMat& img, vector<cv::Rect>& found_locations,
			double hit_threshold = 0, cv::Size win_stride = cv::Size(), cv::Size padding =
					cv::Size(), double scale0 = 1.05, int group_threshold = 2);

	void computeConfidence(const cv::gpu::GpuMat& img, vector<cv::Point>& hits,
			double hit_threshold, cv::Size win_stride, cv::Size padding,
			vector<cv::Point>& locations, vector<double>& confidences);

	void computeConfidenceMultiScale(const cv::gpu::GpuMat& img,
			vector<cv::Rect>& found_locations, double hit_threshold,
			cv::Size win_stride, cv::Size padding, vector<HOGConfidence> &conf_out,
			int group_threshold);

	void getDescriptors(const cv::gpu::GpuMat& img, cv::Size win_stride, cv::gpu::GpuMat& descriptors,
			int descr_format = DESCR_FORMAT_COL_BY_COL);

};

#endif /* HOGDESCRIPTOR_H_ */
