/*
 * App.h
 *
 *  Created on: Sep 9, 2015
 *      Author: fernando
 */

#ifndef APP_H_
#define APP_H_

//new classes
#include "Args.h"

#define MAX_ERROR 0.001
#include <list>
#include <unistd.h>

#include "Helpful.h"
#ifdef LOGS
#include "log_helper.h"
#endif

int iteractions = 1; // global loop iteracion

class App {
public:
	App(const Args& s);
	void run();

	void handleKey(char key);

	void hogWorkBegin();
	void hogWorkEnd();
	string hogWorkFps() const;

	void workBegin();
	void workEnd();
	string workFps() const;

	string message() const;
	friend ostream &operator <<(ostream &os, const App &app);

private:
	App operator=(App&);

	Args args;
	bool running;

	bool use_gpu;
	bool make_gray;
	double scale;
	int gr_threshold;
	int nlevels;
	double hit_threshold;
	bool gamma_corr;

	int64 hog_work_begin;
	double hog_work_fps;

	int64 work_begin;
	double work_fps;
};

ostream &operator <<(ostream &os, const App &app) {
	os << "running " << app.running << endl;
	os << "make gray " << app.use_gpu << endl;
	os << "scale " << app.scale << endl;
	os << "gr threshold " << app.gr_threshold << endl;
	os << "nlevels " << app.nlevels << endl;
	os << "hit threshold " << app.hit_threshold << endl;
	os << "gamma corr " << app.gamma_corr << endl;
	return os;
}

App::App(const Args& s) {
	cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

	args = s;
//	cout << "\nControls:\n" << "\tESC - exit\n"
//			<< "\tm - change mode GPU <-> CPU\n"
//			<< "\tg - convert image to gray or not\n"
//			<< "\t1/q - increase/decrease HOG scale\n"
//			<< "\t2/w - increase/decrease levels count\n"
//			<< "\t3/e - increase/decrease HOG group threshold\n"
//			<< "\t4/r - increase/decrease hit threshold\n" << endl;

	use_gpu = true;
	make_gray = args.make_gray;
	scale = args.scale;
	gr_threshold = args.gr_threshold;
	nlevels = args.nlevels;

	if (args.hit_threshold_auto)
		args.hit_threshold = args.win_width == 48 ? 1.4 : 0.;
	hit_threshold = args.hit_threshold;

	gamma_corr = args.gamma_corr;

	if (args.win_width != 64 && args.win_width != 48)
		args.win_width = 64;

//	cout << "Scale: " << scale << endl;
//	if (args.resize_src)
//		cout << "Resized source: (" << args.width << ", " << args.height
//				<< ")\n";
//	cout << "Group threshold: " << gr_threshold << endl;
//	cout << "Levels number: " << nlevels << endl;
//	cout << "Win width: " << args.win_width << endl;
//	cout << "Win stride: (" << args.win_stride_width << ", "
//			<< args.win_stride_height << ")\n";
//	cout << "Hit threshold: " << hit_threshold << endl;
//	cout << "Gamma correction: " << gamma_corr << endl;
//	cout << endl;
}

void fault_injection(Mat *src, int max_change) {
	cout << max_change;
	while (max_change--) {
		int rand_row = rand() % src->rows;
		int rand_col = rand() % src->cols;
		Vec4b& rgba = src->at < Vec4b > (rand_row, rand_col);
		rgba[0] = UCHAR_MAX;
		rgba[1] = saturate_cast < uchar
				> ((float(src->rows)) / ((float) src->cols) * UCHAR_MAX);
		rgba[2] = saturate_cast < uchar
				> ((float(src->cols)) / ((float) src->rows) * UCHAR_MAX);
		rgba[3] = saturate_cast < uchar > (0.5 * (rgba[1] + rgba[2]));
	}
}

void App::run() {
//	Mat gold = imread(args.dst_video);
	vector < vector<int> > gold;
	ifstream input_file(args.dst_video.c_str());
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
		throw runtime_error(string("can't open image file: " + args.dst_video));
	}
	//get file data
	string line;

	if (getline(input_file, line)) {
		vector < string > sep_line = split(line, ',');
		if (sep_line.size() != 7) {
#ifdef LOGS
			log_error_detail("wrong parameters on gold file");
			end_log_file();
#endif
			throw runtime_error(
					string("wrong parameters on gold file: " + args.dst_video));
		}
		//vector<int> header_out;
		(this->make_gray = (bool) atoi(sep_line[0].c_str()));
		(this->scale = atof(sep_line[1].c_str()));
		(this->gamma_corr = (bool) atoi(sep_line[2].c_str()));
		(this->gr_threshold = atoi(sep_line[3].c_str()));
		(args.win_width = atoi(sep_line[4].c_str()));
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

	Size win_size(args.win_width, args.win_width * 2); //(64, 128) or (48, 96)
	Size win_stride(args.win_stride_width, args.win_stride_height);
	int fault_size = 100000;
// Create HOG descriptors and detectors here
	vector<float> detector;
	if (win_size == Size(64, 128))
		detector = cv::gpu::HOGDescriptor::getPeopleDetector64x128();
	else
		detector = cv::gpu::HOGDescriptor::getPeopleDetector48x96();

	try {

		//gpu --------------------------------------------
		cv::gpu::HOGDescriptor gpu_hog(win_size, Size(16, 16), Size(8, 8),
				Size(8, 8), 9, cv::gpu::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2,
				gamma_corr, cv::gpu::HOGDescriptor::DEFAULT_NLEVELS);
		//====================================================
		//CPU ------------------------------------------------
		cv::HOGDescriptor cpu_hog(win_size, Size(16, 16), Size(8, 8),
				Size(8, 8), 9, 1, -1, HOGDescriptor::L2Hys, 0.2, gamma_corr,
				cv::HOGDescriptor::DEFAULT_NLEVELS);
		//====================================================

		gpu_hog.setSVMDetector(detector);
		cpu_hog.setSVMDetector(detector);
		for (int i = 0; i < iteractions; i++) {
			Mat frame;

			frame = imread(args.src);
			if (frame.empty()) {
#ifdef LOGS
				log_error_detail("Cant open matrix Frame");
				end_log_file();
#endif
				throw runtime_error(
						string("can't open image file: " + args.src));
			}
			//fault injection
			fault_size /= (((i + 1) == iteractions) ? fault_size : (i + 1));
			fault_injection(&frame, fault_size / (i + 1));
			//--------------------
			Mat img_aux, img, img_to_show;
			gpu::GpuMat gpu_img;

			if (use_gpu)
				cvtColor(frame, img_aux, CV_BGR2BGRA);
			else
				frame.copyTo(img_aux);

			img = img_aux;
			img_to_show = img;

			gpu_hog.nlevels = nlevels;
			cpu_hog.nlevels = nlevels;

			vector < Rect > found;

			// Perform HOG classification
			double time;
#ifdef LOGS
			start_iteration();
#endif
			//	if (use_gpu) {
			gpu_img.upload(img);
			time = mysecond();
			gpu_hog.detectMultiScale(gpu_img, found, hit_threshold, win_stride,
					Size(0, 0), scale, gr_threshold);
			time = mysecond() - time;
			/*	} else {
			 time = mysecond();
			 cpu_hog.detectMultiScale(img, found, hit_threshold, win_stride,
			 Size(0, 0), scale, gr_threshold);
			 time = mysecond() - time;
			 }*/
#ifdef LOGS
			end_iteration();
#endif
			cout << "Iteration: " << i << " Time: " << time << " ";
			// Draw positive classified windows
//			for (size_t t = 0; t < found.size(); t++) {
//				Rect r = found[t];
//				rectangle(img_to_show, r.tl(), r.br(), CV_RGB(0, 255, 0), 3);
//			}
			//verify the output----------------------------------------------
			ostringstream error_detail;
			time = mysecond();
			//size_t gold_iterator = 0;
			//bool any_error = false;

//-----------------Lucas Aproach
			bool log_all_rectangles = false;
			bool stop_logging = false;

#ifdef LOGS
			int rectangles_logged = 0;

			if(found.size() != gold.size()) {
				if(found.size() < gold.size()) // log all rectangles to check which were missed
				log_all_rectangles = true;
				char message[120];
				snprintf(message, 120, "Rectangles found: %lu (gold has %lu).", found.size(), gold.size());
				log_error_detail(message);
				if(found.size() > 500) { // inform that only 500 rectangles will be logged
					char msg[100];
					snprintf(msg, 100, "Unreasonable to log all %lu rectangles. Logging the first 500 only.", found.size());
					log_error_detail(msg);
				}
				//corrupted = true;
			}

#endif

//------------------------------

			//vector < vector<int> > data;
			for (size_t s = 0; s < found.size(); s++) {
				Rect r = found[s];
				vector<int> vf(GOLD_LINE_SIZE,0);
				vf[0] = r.height;
				vf[1] = r.width;
				vf[2] = r.x;
				vf[3] = r.y;
				vf[4] = r.br().x;
				vf[5] = r.br().y;

				//vector<int> vector_found(vf, (vf + sizeof(vf) / sizeof(int)));
				//data.push_back(vector_found);
				bool diff = set_countains(vf, gold);

				if ((diff || log_all_rectangles) && !stop_logging) {
					/*
					 error_detail << "SDC: " << s << " Height: " << vf[0]
					 << " width: " << vf[1] << " X: " << vf[2] << " Y: "
					 << vf[3] << endl;
					 #ifdef LOGS
					 char *str = const_cast<char*>(error_detail.str().c_str());
					 log_error_detail(str);
					 #endif
					 any_error = true;
					 //s--;
					 * */
#ifdef LOGS
					char str[150];
					snprintf(str, 150, "%d,%d,%d,%d,%d,%d", r.height, r.width, r.x,
							r.y, r.br().x, r.br().y);
					log_error_detail(str);
					rectangles_logged++;
					if(rectangles_logged > 500)
					stop_logging = true;
#endif
				}
				//if (gold_iterator < gold.size())
				//	gold_iterator++;
				//rectangle(img_to_show, r.tl(), r.br(), CV_RGB(0, 255, 0), 3);
			}
			cout << "Verification time " << mysecond() - time << endl;
			//dump_output(i, "./output", any_error, data);
			stringstream ss;
			ss << fault_size / (i + 1);
			imwrite(ss.str() + "_out.jpg", img_to_show);
		}
		//===============================================================
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
