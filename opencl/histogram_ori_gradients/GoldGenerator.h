/*
 * GoldGenerator.h
 *
 *  Created on: Oct 5, 2015
 *      Author: fernando
 */

#ifndef GOLDGENERATOR_H_
#define GOLDGENERATOR_H_

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
#ifdef LOGS
#include "log_helper.h"
#endif
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
	vdo_source = ""; //cmd.get < string > ("v");
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
//for gold verification---------
	/*vector < vector<int> > gold;
	//ifstream input_file(img_source.c_str());

	if (!input_file.is_open()) {
		throw runtime_error(string("can't open image file: " + output));
	}*/

//------------------------------
	VideoWriter video_writer;

	Size win_size(win_width, win_width * 2);
	Size win_stride(win_stride_width, win_stride_height);

	// Create HOG descriptors and detectors here
	try {
		HOGDescriptor hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9, 1,
				-1, HOGDescriptor::L2Hys, 0.2, gamma_corr,
				cv::HOGDescriptor::DEFAULT_NLEVELS);
		hog.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());

		VideoCapture vc;
		UMat frame;

		if (vdo_source != "") {
			vc.open(vdo_source.c_str());
			if (!vc.isOpened()) {
				throw runtime_error(
						string("can't open video file: " + vdo_source));
			}
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
			if (frame.empty()) {
				throw runtime_error(
						string("can't open image file: " + img_source));
			}
		}

		UMat img_aux, img;
		Mat img_to_show;


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
		hog.detectMultiScale(img, found, hit_threshold, win_stride, Size(0, 0),
				scale, gr_threshold);

		cout << "Total time: " << mysecond() - time << endl;
		// Draw positive classified windows
		/*for (size_t i = 0; i < found.size(); i++) {
			Rect r = found[i];
			rectangle(img_to_show, r.tl(), r.br(), Scalar(0, 255, 0), 3);
		}
		if (img_source != ""){     // wirte image
		 write_once = false;
		 imwrite(output, img_to_show);
		 } */
		//record the gold data----------------------------------------------
		cout << "Gold generated with success\n";
		//save the data on the *.data file
		ofstream output_file;
		output_file.open(string(output + string(".data")).c_str());

		output_file << make_gray << ",";
		output_file << scale << ",";
		output_file << gamma_corr << ",";
		output_file << gr_threshold << ",";
		output_file << win_width << ",";
		output_file << hit_threshold << ",";
		output_file << nlevels << endl;

		if (output_file.is_open()) {
			for (size_t i = 0; i < found.size(); i++) {
				// Draw positive classified windows
				Rect r = found[i];
				rectangle(img_to_show, r.tl(), r.br(), CV_RGB(0, 255, 0), 3);
				//new approach
				output_file << r.height << "," << r.width << "," << r.x << ","
						<< r.y << "," << r.br().x << "," << r.br().y << endl;
			}
			output_file.close();
		} else {
			throw runtime_error(
					string("can't create output file: " + output));
		}

		//save the output
		cvtColor(img_to_show, img, CV_BGRA2BGR);
		imwrite(string("output_") + output, img_to_show);

	} catch (cv::Exception &e) {
		char *str = const_cast<char*>(e.what());
		string err_msg(str);
		throw runtime_error(string("error: " + err_msg));
	}
}

#endif /* APP_H_ */

#endif /* GOLDGENERATOR_H_ */
