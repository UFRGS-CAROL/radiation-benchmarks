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
	//running = true;
//for gold verification---------
	vector < vector<int> > gold;
	ifstream input_file(output.c_str());
	//================== Init logs
#ifdef LOGS
	char test_info[90];
	snprintf(test_info, 90, "HOG GOLD TEXT FILE");
	start_log_file("HOG", test_info);
#endif
	//====================================
	if (!input_file.is_open()) {
#ifdef LOGS
		log_error_detail("Cant open gold file");
		end_log_file();
#endif
		throw runtime_error(string("can't open image file: " + output));
	}
	//get file data
	string line;

	if (getline(input_file, line)) {
		vector<string> sep_line = split(line, ',');
		if (sep_line.size() != 7) {
#ifdef LOGS
			log_error_detail("wrong parameters on gold file");
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
		vector<string> sep_line = split(line, ',');
		vector<int> values;
		for (int i = 0; i < GOLD_LINE_SIZE; i++) {
			values.push_back(atoi(sep_line[i].c_str()));
		}
		gold.push_back(values);
	}
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
#ifdef LOGS
				log_error_detail("Cant open matrix Frame");
				end_log_file();
#endif
				throw runtime_error(
						string("can't open image file: " + img_source));
			}
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
			img.copyTo(img_to_show);
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
			// Draw positive classified windows
			/*for (size_t i = 0; i < found.size(); i++) {
			 Rect r = found[i];
			 rectangle(img_to_show, r.tl(), r.br(), Scalar(0, 255, 0), 3);
			 }
			 if (img_source != ""){     // wirte image
			 write_once = false;
			 imwrite(output, img_to_show);
			 } */
			//verify the output----------------------------------------------
			ostringstream error_detail;
			time = mysecond();
			size_t gold_iterator = 0;
			bool any_error = false;

			vector < vector<int> > data;
			for (size_t s = 0; s < found.size(); s++) {
				Rect r = found[s];
				int vf[8];
				vf[0] = r.height;
				vf[1] = r.width;
				vf[2] = r.x;
				vf[3] = r.y;
				vf[4] = r.tl().x;
				vf[5] = r.tl().y;
				vf[6] = r.br().x;
				vf[7] = r.br().y;
				vector<int> values = gold[gold_iterator];

				data.push_back(
						vector<int>(vf, (vf + sizeof(vf) / sizeof(int))));

				if ((vf[0] != values[0]) || (vf[1] != values[1])
						|| (vf[2] != values[2]) || (vf[3] != values[3])
						|| (vf[4] != values[4]) || (vf[5] != values[5])
						|| (vf[6] != values[6]) || (vf[7] != values[7])) {
					error_detail << "SDC: " << s << ", Height: " << vf[0]
							<< ", width: " << vf[1] << ", X: " << vf[2]
							<< ", Y: " << vf[3] << endl;
#ifdef LOGS
					char *str = const_cast<char*>(error_detail.str().c_str());
					log_error_detail(str);
#endif
					any_error = true;
					//s--;
				}
				if (gold_iterator < gold.size())
					gold_iterator++;
			}
			dump_output(j, "./output", any_error, data);
			cout << "Verification time " << mysecond() - time << endl;
		}
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

