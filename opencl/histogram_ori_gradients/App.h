/*
 * App.h
 *
 *  Created on: Sep 19, 2015
 *      Author: fernando
 */

#ifndef APP_H_
#define APP_H_
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
//my libs
#include "../../cuda/histogram_ori_gradients/Helpful.h"

using namespace std;
using namespace cv;
int iteractions = 5; //global loop iteration

class App {
public:
	App(CommandLineParser& cmd);
	void run();
	void handleKey(char key);
	void hogWorkBegin();
	void hogWorkEnd();
	string hogWorkFps() const;
	void workBegin();
	void workEnd();
	string workFps() const;
	string message() const;

// This function test if gpu_rst matches cpu_rst.
// If the two vectors are not equal, it will return the difference in vector size
// Else if will return
// (total diff of each cpu and gpu rects covered pixels)/(total cpu rects covered pixels)
	double checkRectSimilarity(Size sz, std::vector<Rect>& cpu_rst,
			std::vector<Rect>& gpu_rst);
private:
	App operator=(App&);

	//Args args;
	bool running;
	bool make_gray;
	double scale;
	double resize_scale;
	int win_width;
	int win_stride_width, win_stride_height;
	int gr_threshold;
	int nlevels;
	double hit_threshold;
	bool gamma_corr;

	int64 hog_work_begin;
	double hog_work_fps;
	int64 work_begin;
	double work_fps;

	string img_source;
	string vdo_source;
	string output;
	int camera_id;
	bool write_once;
};

App::App(CommandLineParser& cmd) {
	/*cout << "\nControls:\n" << "\tESC - exit\n"
			<< "\tm - change mode GPU <-> CPU\n"
			<< "\tg - convert image to gray or not\n"
			<< "\to - save output image once, or switch on/off video save\n"
			<< "\t1/q - increase/decrease HOG scale\n"
			<< "\t2/w - increase/decrease levels count\n"
			<< "\t3/e - increase/decrease HOG group threshold\n"
			<< "\t4/r - increase/decrease hit threshold\n" << endl;*/

	make_gray = cmd.has("gray");
	resize_scale = cmd.get<double>("s");
	vdo_source = cmd.get < string > ("v");
	img_source = cmd.get < string > ("i");
	output = cmd.get < string > ("o");
	camera_id = cmd.get<int>("c");

	win_width = 48;
	win_stride_width = 8;
	win_stride_height = 8;
	gr_threshold = 1;
	nlevels = 80;
	hit_threshold = 1.4;
	scale = 1.05;
	gamma_corr = true;
	write_once = false;
	cout << "Video " << vdo_source << endl;
	/*cout << "Group threshold: " << gr_threshold << endl;
	cout << "Levels number: " << nlevels << endl;
	cout << "Win width: " << win_width << endl;
	cout << "Win stride: (" << win_stride_width << ", " << win_stride_height
			<< ")\n";
	cout << "Hit threshold: " << hit_threshold << endl;
	cout << "Gamma correction: " << gamma_corr << endl;*/
	//cout << endl;
}

void App::run() {
	//running = true;
	VideoWriter video_writer;

	Size win_size(win_width, win_width * 2);
	Size win_stride(win_stride_width, win_stride_height);

	// Create HOG descriptors and detectors here

	HOGDescriptor hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9, 1, -1,
			HOGDescriptor::L2Hys, 0.2, gamma_corr,
			cv::HOGDescriptor::DEFAULT_NLEVELS);
	hog.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());

	VideoCapture vc;
	UMat frame;

	if (vdo_source != "") {
		vc.open(vdo_source.c_str());
		if (!vc.isOpened())
			throw runtime_error(
					string("can't open video file: " + vdo_source));
		vc >> frame;
	} 
		/*else if (camera_id != -1) {
		vc.open(camera_id);
		if (!vc.isOpened()) {
			stringstream msg;
			msg << "can't open camera: " << camera_id;
			throw runtime_error(msg.str());
		}
		vc >> frame;
	}*/
	else {
		imread(img_source).copyTo(frame);
		if (frame.empty())
			throw runtime_error(
					string("can't open image file: " + img_source));
	}

	UMat img_aux, img;
	Mat img_to_show;

	// Iterate all iterations
	for(int j = 0; j < iteractions; j++){
		frame.copyTo(img_aux);

		// Resize image
		if (abs(scale - 1.0) > 0.001) {
			Size sz((int) ((double) img_aux.cols / resize_scale),
					(int) ((double) img_aux.rows / resize_scale));
			resize(img_aux, img, sz);
		} else
			img = img_aux;
		img.copyTo(img_to_show);
		hog.nlevels = nlevels;
		vector < Rect > found;

		// Perform HOG classification
		double time = mysecond();
		hog.detectMultiScale(img, found, hit_threshold, win_stride,
				Size(0, 0), scale, gr_threshold);

		cout << "Total time: " << mysecond() - time << endl;


	}
}

#endif /* APP_H_ */
