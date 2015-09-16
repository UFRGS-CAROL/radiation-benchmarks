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
#include <sys/time.h>
#define MAX_ERROR 0.0000000001
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
	double mysecond();

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

App::App(const Args& s) {
	cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

	args = s;
	cout << "\nControls:\n" << "\tESC - exit\n"
			<< "\tm - change mode GPU <-> CPU\n"
			<< "\tg - convert image to gray or not\n"
			<< "\t1/q - increase/decrease HOG scale\n"
			<< "\t2/w - increase/decrease levels count\n"
			<< "\t3/e - increase/decrease HOG group threshold\n"
			<< "\t4/r - increase/decrease hit threshold\n" << endl;

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

	cout << "Scale: " << scale << endl;
	if (args.resize_src)
		cout << "Resized source: (" << args.width << ", " << args.height
				<< ")\n";
	cout << "Group threshold: " << gr_threshold << endl;
	cout << "Levels number: " << nlevels << endl;
	cout << "Win width: " << args.win_width << endl;
	cout << "Win stride: (" << args.win_stride_width << ", "
			<< args.win_stride_height << ")\n";
	cout << "Hit threshold: " << hit_threshold << endl;
	cout << "Gamma correction: " << gamma_corr << endl;
	cout << endl;
}

void App::run() {
	Mat gold = imread(args.dst_video);
	//================== Init logs
#ifdef LOGS
	char test_info[90];
	snprintf(test_info, 90, "size:%d", k);
	start_log_file("HOG", test_info);
#endif
	//====================================

	Size win_size(args.win_width, args.win_width * 2); //(64, 128) or (48, 96)
	Size win_stride(args.win_stride_width, args.win_stride_height);

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
			if (frame.empty())
				throw runtime_error(
						string("can't open image file: " + args.src));

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

			vector<Rect> found;

			// Perform HOG classification
			double time;
			if (use_gpu) {
				gpu_img.upload(img);
				time = mysecond();
				gpu_hog.detectMultiScale(gpu_img, found, hit_threshold,
						win_stride, Size(0, 0), scale, gr_threshold);
				time = mysecond() - time;
			} else {
				time = mysecond();
				cpu_hog.detectMultiScale(img, found, hit_threshold, win_stride,
						Size(0, 0), scale, gr_threshold);
				time = mysecond() - time;
			}

			cout << "Iteration: " << i << "Time: " << time << endl;
			// Draw positive classified windows
			for (size_t i = 0; i < found.size(); i++) {
				Rect r = found[i];
				rectangle(img_to_show, r.tl(), r.br(), CV_RGB(0, 255, 0), 3);
			}

			//verify the output----------------------------------------------
			ostringstream error_detail;
			int x_gold, x_frame;
			for(int j = 0; j < gold.rows; j++){
				for(int k = 0; k < gold.cols; k++){
					x_gold = gold.at<int>(j, k);
					x_frame = frame.at<int>(j, k);
					if((fabs((x_frame - x_gold) / x_frame) > MAX_ERROR) || (fabs((x_frame - x_gold) / x_gold) > MAX_ERROR)){
						error_detail << "p: [" << j << "," << k << "], r: " << x_frame << ", e: " << x_gold;
#ifdef LOGS
						log_error_detail(error_detail.c_str);
#endif
					}
				}
				cout << endl;
			}
			//===============================================================
		}
	} catch (cv::Exception &e) {
		const char *err_msg = e.what();

	}
}

inline void App::hogWorkBegin() {
	hog_work_begin = getTickCount();
}

inline void App::hogWorkEnd() {
	int64 delta = getTickCount() - hog_work_begin;
	double freq = getTickFrequency();
	hog_work_fps = freq / delta;
}

inline string App::hogWorkFps() const {
	stringstream ss;
	ss << hog_work_fps;
	return ss.str();
}

inline void App::workBegin() {
	work_begin = getTickCount();
}

inline void App::workEnd() {
	int64 delta = getTickCount() - work_begin;
	double freq = getTickFrequency();
	work_fps = freq / delta;
}

inline string App::workFps() const {
	stringstream ss;
	ss << work_fps;
	return ss.str();
}

double App::mysecond() {
	struct timeval tp;
	struct timezone tzp;
	gettimeofday(&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

#endif /* APP_H_ */
