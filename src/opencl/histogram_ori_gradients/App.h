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
int iteractions = 1000000; //default value

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

	void write_output_image(vector<Rect> found, Mat img_to_show, string output);

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
	make_gray = cmd.has("gray");
	resize_scale = cmd.get<double>("s");
	vdo_source = ""; //cmd.get < string > ("v");
	img_source = cmd.get < string > ("i");
	output = cmd.get < string > ("o");
	camera_id = cmd.get<int>("c");

	//adjusted parameters
	win_width = 48;
	win_stride_width = 8;
	win_stride_height = 8;
	gr_threshold = 1;
	nlevels = 100;
	hit_threshold = 0.9;
	scale = 1.05;
	gamma_corr = true;
	write_once = false;

}

/** this method write the final image, if required **/
void App::write_output_image(vector<Rect> found, Mat img_to_show, string output) {
	// Draw positive classified windows
	for (size_t i = 0; i < found.size(); i++) {
		Rect r = found[i];
		rectangle(img_to_show, r.tl(), r.br(), Scalar(0, 255, 0), 3);
	}
	imwrite(output, img_to_show);
}

/**
This method performs HOG classification
if LOGS is set with TRUE every iteration information 
about execution will be saved as well corrupted outputs.
Only the rectangle coordinates are recorded in case of errors.
**/

void App::run() {
	//for gold verification---------
	vector < vector<int> > gold;
	ifstream input_file(output.c_str());
	//init logs
	//====================================
#ifdef LOGS
	char test_info[90];
	int image_size = atoi(img_source.c_str()[0]);
	snprintf(test_info, 90, "gold %dx image", image_size);
	start_log_file((char*)"openclHOG", test_info);
#endif
	//====================================
	if (!input_file.is_open()) {
#ifdef LOGS
		log_error_detail((char*)"Cant open gold file.");
		end_log_file();
#endif
		throw runtime_error(string("can't open image file: " + output));
	}

	//open gold values
	//====================================
	string line;

	if (getline(input_file, line)) {
		vector < string > sep_line = split(line, ',');
		if (sep_line.size() != 7) {
#ifdef LOGS	
			log_error_detail((char*)"Wrong parameters on gold file.");
			end_log_file();
#endif
			throw runtime_error(
					string("wrong parameters on gold file: " + output));
		}
		//vector<int> header_out;
		(this->make_gray = (bool) atoi(sep_line[0].c_str()));
		(this->scale = atof(sep_line[1].c_str()));
		(this->gamma_corr = (bool) atoi(sep_line[2].c_str()));
		(this->gr_threshold = atoi(sep_line[3].c_str()));
		(win_width = atoi(sep_line[4].c_str()));
		(this->hit_threshold = atof(sep_line[5].c_str()));
		(this->nlevels = atoi(sep_line[6].c_str()));
		//data.push_back(header_out);
	}

	while (getline(input_file, line)) {
		vector < string > sep_line = split(line, ',');
		vector<int> values;
		for (int i = 0; i < GOLD_LINE_SIZE; i++) {
			values.push_back(atoi(sep_line[i].c_str()));
		}
		gold.push_back(values);
	}

	//====================================


	//set win size and win stride for HOG
	//====================================
	Size win_size(win_width, win_width * 2);
	Size win_stride(win_stride_width, win_stride_height);
	//====================================

	try {
	// Create HOG descriptors and detectors here
	//====================================
		HOGDescriptor hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9, 1,
				-1, HOGDescriptor::L2Hys, 0.2, gamma_corr,
				cv::HOGDescriptor::DEFAULT_NLEVELS);
		hog.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());
	//====================================

		UMat frame;
	//open source img
		imread(img_source).copyTo(frame);
		if (frame.empty()) {
#ifdef LOGS
			log_error_detail((char*)"Cant open matrix frame.");
			end_log_file();
#endif
			throw runtime_error(string("can't open image file: " + img_source));
		}

		UMat img_aux, img;
		Mat img_to_show;

		// Iterate all iterations
		for (int j = 0; j < iteractions; j++) {
			frame.copyTo(img_aux);

			// Resize image
			if (abs(scale - 1.0) > 0.001) {
				Size sz((int) ((double) img_aux.cols / resize_scale),
						(int) ((double) img_aux.rows / resize_scale));
				resize(img_aux, img, sz);
			} else
				img = img_aux;
			//make a copy every iteration
			img.copyTo(img_to_show);
			//set hog levels
			hog.nlevels = nlevels;
			vector < Rect > found;

			// Perform HOG classification
#ifdef LOGS
			start_iteration();
#endif
			double time = mysecond();
			hog.detectMultiScale(img, found, hit_threshold, win_stride,
					Size(0, 0), scale, gr_threshold);
#ifdef LOGS
			end_iteration();
#endif
			cout << "Total time: " << mysecond() - time << endl;
	//output verification
	//====================================
			time = mysecond();

			unsigned long int error_counter = 0;

#ifdef LOGS
			//if the numbers of rects found is different from gold, there is some errors
			if(found.size() != gold.size()) {
				char message[120];
				snprintf(message, 120, "Rectangles found: %lu (gold has %lu).", found.size(), gold.size());
				log_error_detail(message);
				error_counter++;
			}
#endif
			//check if rects are different
			for (size_t s = 0; s < found.size(); s++) {
				Rect r = found[s];
				vector<int> vf(GOLD_LINE_SIZE, 0);
				vf[0] = r.height;
				vf[1] = r.width;
				vf[2] = r.x;
				vf[3] = r.y;
				vf[4] = r.br().x;
				vf[5] = r.br().y;


				bool diff = set_countains(vf, gold);
				if (diff) error_counter++;
			}
#ifdef LOGS
			//algorithm stops if there are more than 500 iterations with errors
			if (error_counter) { 
				for(size_t g = 0; g < found.size(); g++) {
					Rect r = found[g];
					char str[150];
					snprintf(str, 150, "%d,%d,%d,%d,%d,%d", r.height, r.width, r.x,	r.y, r.br().x, r.br().y); log_error_detail(str);
				}
			}
			cout << "Verification time " << mysecond() - time << endl;
			log_error_count(error_counter);
		}
#endif
	//====================================

	} catch (cv::Exception &e) {
		char *str = const_cast<char*>(e.what());
		string err_msg(str);
#ifdef LOGS
		log_error_detail(str);
		end_log_file();
#endif
		throw runtime_error(string("error: " + err_msg));
	}
#ifdef LOGS
	end_log_file();
#endif
}
#endif /* APP_H_ */

