//Hog_class.h

#ifndef HOG_CLASS_H
#define HOG_CLASS_H

#include "hog.h"
using namespace std;

struct HOGConfidence
{
   double scale;
   vector<Point> locations;
   vector<double> confidences;
   vector<double> part_scores[4];
};

class HOGDescriptor
{
    enum { DEFAULT_WIN_SIGMA = -1 };
    enum { DEFAULT_NLEVELS = 64 };
    enum { DESCR_FORMAT_ROW_BY_ROW, DESCR_FORMAT_COL_BY_COL };

    HOGDescriptor(Size win_size=Size(64, 128), Size block_size=Size(16, 16),
                  Size block_stride=Size(8, 8), Size cell_size=Size(8, 8),
                  int nbins=9, double win_sigma=DEFAULT_WIN_SIGMA,
                  double threshold_L2hys=0.2, bool gamma_correction=true,
                  int nlevels=DEFAULT_NLEVELS);

    size_t getDescriptorSize() const;
    size_t getBlockHistogramSize() const;

    void setSVMDetector(const vector<float>& detector);

    static vector<float> getDefaultPeopleDetector();
    static vector<float> getPeopleDetector48x96();
    static vector<float> getPeopleDetector64x128();

    void detect(const GpuMat& img, vector<Point>& found_locations,
                double hit_threshold=0, Size win_stride=Size(),
                Size padding=Size());

    void detectMultiScale(const GpuMat& img, vector<Rect>& found_locations,
                          double hit_threshold=0, Size win_stride=Size(),
                          Size padding=Size(), double scale0=1.05,
                          int group_threshold=2);

    void computeConfidence(const GpuMat& img, vector<Point>& hits, double hit_threshold,
                                                Size win_stride, Size padding, vector<Point>& locations, vector<double>& confidences);

    void computeConfidenceMultiScale(const GpuMat& img, vector<Rect>& found_locations,
                                                                    double hit_threshold, Size win_stride, Size padding,
                                                                    vector<HOGConfidence> &conf_out, int group_threshold);

    void getDescriptors(const GpuMat& img, Size win_stride,
                        GpuMat& descriptors,
                        int descr_format=DESCR_FORMAT_COL_BY_COL);

    Size win_size;
    Size block_size;
    Size block_stride;
    Size cell_size;
    int nbins;
    double win_sigma;
    double threshold_L2hys;
    bool gamma_correction;
    int nlevels;

protected:
    void computeBlockHistograms(const GpuMat& img);
    void computeGradient(const GpuMat& img, GpuMat& grad, GpuMat& qangle);

    double getWinSigma() const;
    bool checkDetectorSize() const;

    static int numPartsWithin(int size, int part_size, int stride);
    static Size numPartsWithin(Size size, Size part_size, Size stride);

    // Coefficients of the separating plane
    float free_coef;
    GpuMat detector;

    // Results of the last classification step
    GpuMat labels, labels_buf;
    Mat labels_host;

    // Results of the last histogram evaluation step
    GpuMat block_hists, block_hists_buf;

    // Gradients conputation results
    GpuMat grad, qangle, grad_buf, qangle_buf;

    // returns subbuffer with required size, reallocates buffer if nessesary.
    static GpuMat getBuffer(const Size& sz, int type, GpuMat& buf);
    static GpuMat getBuffer(int rows, int cols, int type, GpuMat& buf);

    std::vector<GpuMat> image_scales;
};

#endif
