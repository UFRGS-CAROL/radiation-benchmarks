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
	running = true;
	cv::VideoWriter output_video;
	const string NAME = "output.avi";


	Size win_size(args.win_width, args.win_width * 2); //(64, 128) or (48, 96)
	Size win_stride(args.win_stride_width, args.win_stride_height);

	// Create HOG descriptors and detectors here
	vector<float> detector;
	if (win_size == Size(64, 128))
		detector = cv::gpu::HOGDescriptor::getPeopleDetector64x128();
	else
		detector = cv::gpu::HOGDescriptor::getPeopleDetector48x96();

	cv::gpu::HOGDescriptor gpu_hog(win_size, Size(16, 16), Size(8, 8),
			Size(8, 8), 9, cv::gpu::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2,
			gamma_corr, cv::gpu::HOGDescriptor::DEFAULT_NLEVELS);
	cv::HOGDescriptor cpu_hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9,
			1, -1, HOGDescriptor::L2Hys, 0.2, gamma_corr,
			cv::HOGDescriptor::DEFAULT_NLEVELS);
	gpu_hog.setSVMDetector(detector);
	cpu_hog.setSVMDetector(detector);

	//executes while there is an input to show
//	while (running) {
		VideoCapture vc;
		Mat frame;

		if (args.src_is_video) {
			vc.open(args.src.c_str());
			if (!vc.isOpened())
				throw runtime_error(
						string("can't open video file: " + args.src));
			vc >> frame;
		}
//		else if (args.src_is_camera) {
//			vc.open(args.camera_id);
//			if (!vc.isOpened()) {
//				stringstream msg;
//				msg << "can't open camera: " << args.camera_id;
//				throw runtime_error(msg.str());
//			}
//			vc >> frame;
//		}
		//if it is an image
		else {
			frame = imread(args.src);
			if (frame.empty())
				throw runtime_error(
						string("can't open image file: " + args.src));
		}

		Mat img_aux, img, img_to_show;
		gpu::GpuMat gpu_img;

		// Iterate over all frames
		while (/*running && */!frame.empty()) {
//			workBegin();

			// Change format of the image
			if (make_gray)
				cvtColor(frame, img_aux, CV_BGR2GRAY);
			else if (use_gpu)
				cvtColor(frame, img_aux, CV_BGR2BGRA);
			else
				frame.copyTo(img_aux);

			// Resize image
			Size S = Size(args.width, args.height);
			if (args.resize_src)
				resize(img_aux, img, S);
			else
				img = img_aux;
			img_to_show = img;

			gpu_hog.nlevels = nlevels;
			cpu_hog.nlevels = nlevels;

			vector<Rect> found;

			// Perform HOG classification
//			hogWorkBegin();
			if (use_gpu) {
				gpu_img.upload(img);
				gpu_hog.detectMultiScale(gpu_img, found, hit_threshold,
						win_stride, Size(0, 0), scale, gr_threshold);
			} else
				cpu_hog.detectMultiScale(img, found, hit_threshold, win_stride,
						Size(0, 0), scale, gr_threshold);
//			hogWorkEnd();

			// Draw positive classified windows
			for (size_t i = 0; i < found.size(); i++) {
				Rect r = found[i];
				rectangle(img_to_show, r.tl(), r.br(), CV_RGB(0, 255, 0), 3);
			}

			//gold creation---------------------------------------------------
			int ex = static_cast<int>(vc.get(CV_CAP_PROP_FOURCC));
			output_video.open(NAME, ex, vc.get(CV_CAP_PROP_FPS), S, true);
		    if (!output_video.isOpened())
		    {
		        cout  << "Could not open the output video for write: " << endl;
		        return;
		    }
		    output_video << img_to_show;
		    //----------------------------------------------------------------

//			cout << "FPS(hog only): " << hogWorkFps() << "FPS total "
//					<< workFps() << endl;
//			if (use_gpu)
//				putText(img_to_show, "Mode: GPU", Point(5, 25),
//						FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
//			else
//				putText(img_to_show, "Mode: CPU", Point(5, 25),
//						FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
//			putText(img_to_show, "FPS (HOG only): " + hogWorkFps(),
//					Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0),
//					2);
//			putText(img_to_show, "FPS (total): " + workFps(), Point(5, 105),
//					FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
//			imshow("opencv_gpu_hog", img_to_show);

			if (args.src_is_video || args.src_is_camera) {
				//output
				vc >> frame;
			}

//			workEnd();
//
//			if (args.write_video) {
//				cout << "passou aqui\n";
//				if (!video_writer.isOpened()) {
//					video_writer.open(args.dst_video,
//							CV_FOURCC('x', 'v', 'i', 'd'), args.dst_video_fps,
//							img_to_show.size(), true);
//					if (!video_writer.isOpened())
//						throw std::runtime_error("can't create video writer");
//				}
//
//				if (make_gray)
//					cvtColor(img_to_show, img, CV_GRAY2BGR);
//				else
//					cvtColor(img_to_show, img, CV_BGRA2BGR);
//
//				video_writer << img;
//			}
//			handleKey((char) waitKey(3));
		}
//	}
}

void App::handleKey(char key) {
	switch (key) {
	case 27:
		running = false;
		break;
	case 'm':
	case 'M':
		use_gpu = !use_gpu;
		cout << "Switched to " << (use_gpu ? "CUDA" : "CPU") << " mode\n";
		break;
	case 'g':
	case 'G':
		make_gray = !make_gray;
		cout << "Convert image to gray: " << (make_gray ? "YES" : "NO") << endl;
		break;
	case '1':
		scale *= 1.05;
		cout << "Scale: " << scale << endl;
		break;
	case 'q':
	case 'Q':
		scale /= 1.05;
		cout << "Scale: " << scale << endl;
		break;
	case '2':
		nlevels++;
		cout << "Levels number: " << nlevels << endl;
		break;
	case 'w':
	case 'W':
		nlevels = max(nlevels - 1, 1);
		cout << "Levels number: " << nlevels << endl;
		break;
	case '3':
		gr_threshold++;
		cout << "Group threshold: " << gr_threshold << endl;
		break;
	case 'e':
	case 'E':
		gr_threshold = max(0, gr_threshold - 1);
		cout << "Group threshold: " << gr_threshold << endl;
		break;
	case '4':
		hit_threshold += 0.25;
		cout << "Hit threshold: " << hit_threshold << endl;
		break;
	case 'r':
	case 'R':
		hit_threshold = max(0.0, hit_threshold - 0.25);
		cout << "Hit threshold: " << hit_threshold << endl;
		break;
	case 'c':
	case 'C':
		gamma_corr = !gamma_corr;
		cout << "Gamma correction: " << gamma_corr << endl;
		break;
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

#endif /* APP_H_ */
