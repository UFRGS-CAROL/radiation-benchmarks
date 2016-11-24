/*
 * GoldGenerator.h
 *
 *  Created on: Sep 16, 2015
 *      Author: fernando
 */

#ifndef GOLDGENERATOR_H_
#define GOLDGENERATOR_H_
//new classes
#include "Args_Generator.h"
#include "Helpful.h"

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
	/*if (args.dst_video.empty())
		throw runtime_error(
				string("No output path. [--dst_data <path>] # output data\n"));*/
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

	VideoCapture vc;
	Mat frame;
	
	vector <string> dataset;
	string dataset_line;
	ifstream images(args.src.c_str());
	if(images.is_open()) {
		while(getline(images, dataset_line)) {
			dataset.push_back(dataset_line);
		}
	}
	else
		throw runtime_error(string("error opening dataset text file"));

	for (int index = 0; index < dataset.size(); index++) {

	frame = imread(dataset[index]);
	if (frame.empty())
		throw runtime_error(string("can't open image file: " + args.src));

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
	cout << "Generating gold for " << dataset[index].c_str() << "\n";
	if (use_gpu) {
		gpu_img.upload(img);
		gpu_hog.detectMultiScale(gpu_img, found, hit_threshold, win_stride,
				Size(0, 0), scale, gr_threshold);
	} else {
		cpu_hog.detectMultiScale(img, found, hit_threshold, win_stride,
				Size(0, 0), scale, gr_threshold);
	}
	//cout << "Gold generated with success\n";
	// Draw positive classified windows (OLD)
	//save the data on the *.data file

	vector<string> split_current_line = split(dataset[index], '/');
	string gold_set = split_current_line[split_current_line.size() - 3].c_str();
	string gold_video = split_current_line[split_current_line.size() - 2].c_str();
	string gold_frame = split_current_line[split_current_line.size() - 1].c_str();
	gold_set.append("_" + gold_video + "_" + gold_frame + ".data");

	ofstream output_file;

	string data_path("/home/carol/radiation-benchmarks/data/histogram_ori_gradients/");
	data_path.append(gold_set);
	output_file.open(data_path.c_str());

	output_file << args.make_gray << ",";
	output_file << args.scale << ",";
	output_file << args.gamma_corr << ",";
	output_file << args.gr_threshold << ",";
	output_file << args.win_width << ",";
	output_file << args.hit_threshold << ",";
	output_file << args.nlevels << endl;

	if (output_file.is_open()) {
		for (size_t i = 0; i < found.size(); i++) {
			Rect r = found[i];
			//rectangle(img_to_show, r.tl(), r.br(), CV_RGB(0, 255, 0), 3);
			//new approach
			output_file << r.height << "," << r.width << "," << r.x << ","
					<< r.y << "," << r.br().x << "," << r.br().y << endl;
		}
		output_file.close();
	} else {
		throw runtime_error(
				string("can't create output file: " + args.dst_video));
	}

	// Draw positive classified windows
	for (size_t i = 0; i < found.size(); i++) {
		Rect r = found[i];
		rectangle(img_to_show, r.tl(), r.br(), CV_RGB(0, 255, 0), 3);
	}

	//save the output
	cvtColor(img_to_show, img, CV_BGRA2BGR);
	imwrite(string("output_") + dataset[index], img);
	}//txt loop
}

#endif /* GOLDGENERATOR_H_ */
