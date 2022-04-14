/*
 * Args.h
 *
 *  Created on: Sep 9, 2015
 *      Author: fernando
 */

#ifndef ARGS_H_
#define ARGS_H_

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cstdlib>
#include "opencv2/highgui/highgui.hpp"

using namespace std;

class Args {
public:
	Args();
	static Args read(int argc, char** argv);
	bool help_showed;
	string src;
	bool src_is_video;
	bool src_is_camera;
	int camera_id;

	bool write_video;
	string dst_video;
	double dst_video_fps;

	bool make_gray;

	bool resize_src;
	int width, height;

	double scale;
	int nlevels;
	int gr_threshold;

	double hit_threshold;
	bool hit_threshold_auto;

	int win_width;
	int win_stride_width, win_stride_height;
	bool gamma_corr;
	int iterations;

	void printHelp() {
		cout << "Histogram of Oriented Gradients descriptor and detector sample.\n"
				<< "\nUsage: hog_gpu\n"
				<< "  (<image>|--video <vide>|--camera <camera_id>) # frames source\n"
				<< "  [--make_gray <true/false>] # convert image to gray one or not\n"
				<< "  [--resize_src <true/false>] # do resize of the source image or not\n"
				<< "  [--width <int>] # resized image width\n"
				<< "  [--height <int>] # resized image height\n"
				<< "  [--hit_threshold <double>] # classifying plane distance threshold (0.0 usually)\n"
				<< "  [--scale <double>] # HOG window scale factor\n"
				<< "  [--nlevels <int>] # max number of HOG window scales\n"
				<< "  [--win_width <int>] # width of the window (48 or 64)\n"
				<< "  [--win_stride_width <int>] # distance by OX axis between neighbour wins\n"
				<< "  [--win_stride_height <int>] # distance by OY axis between neighbour wins\n"
				<< "  [--gr_threshold <int>] # merging similar rects constant\n"
				<< "  [--gamma_correct <int>] # do gamma correction or not\n"
				<< "  [--write_video <bool>] # write video or not\n"
				<< "  [--dst_video <path>] # output video path\n"
				<< "  [--dst_video_fps <double>] # output video fps\n";
		help_showed = true;
	}

	friend ostream &operator <<(ostream &os, const Args &dt);
};


Args::Args() {
	help_showed = false;
	src_is_video = false;
	src_is_camera = false;
	camera_id = 0;

	write_video = false;
	dst_video_fps = 24.;

	make_gray = false;

	resize_src = false;
	width = 640;
	height = 480;

	scale = 1.05;
	nlevels = 13;
	gr_threshold = 8;
	hit_threshold = 1.4;
	hit_threshold_auto = true;

	win_width = 48;
	win_stride_width = 8;
	win_stride_height = 8;
        iterations = 0;
	gamma_corr = true;
}

Args Args::read(int argc, char** argv) {
	Args args;
	for (int i = 1; i < argc; i++) {
		if (string(argv[i]) == "--make_gray")
			args.make_gray = (string(argv[++i]) == "true");
		else if (string(argv[i]) == "--resize_src")
			args.resize_src = (string(argv[++i]) == "true");
		else if (string(argv[i]) == "--iterations")
		        args.iterations = atoi(argv[++i]);
		else if (string(argv[i]) == "--width")
			args.width = atoi(argv[++i]);
		else if (string(argv[i]) == "--height")
			args.height = atoi(argv[++i]);
		else if (string(argv[i]) == "--hit_threshold") {
			args.hit_threshold = atof(argv[++i]);
			args.hit_threshold_auto = false;
		} else if (string(argv[i]) == "--scale")
			args.scale = atof(argv[++i]);
		else if (string(argv[i]) == "--nlevels")
			args.nlevels = atoi(argv[++i]);
		else if (string(argv[i]) == "--win_width")
			args.win_width = atoi(argv[++i]);
		else if (string(argv[i]) == "--win_stride_width")
			args.win_stride_width = atoi(argv[++i]);
		else if (string(argv[i]) == "--win_stride_height")
			args.win_stride_height = atoi(argv[++i]);
		else if (string(argv[i]) == "--gr_threshold")
			args.gr_threshold = atoi(argv[++i]);
		else if (string(argv[i]) == "--gamma_correct")
			args.gamma_corr = (string(argv[++i]) == "true");
		else if (string(argv[i]) == "--write_video")
			args.write_video = (string(argv[++i]) == "true");
		else if (string(argv[i]) == "--dst_data")
			args.dst_video = argv[++i];
		else if (string(argv[i]) == "--dst_video_fps")
			args.dst_video_fps = atof(argv[++i]);
		else if (string(argv[i]) == "--help")
			args.printHelp();
		else if (string(argv[i]) == "--video") {
			args.src = argv[++i];
			args.src_is_video = true;
		} else if (string(argv[i]) == "--camera") {
			args.camera_id = atoi(argv[++i]);
			args.src_is_camera = true;
		} else if (args.src.empty())
			args.src = argv[i];
		else
			throw runtime_error((string("unknown key: ") + argv[i]));
	}
	return args;
}

ostream &operator<<(ostream &os, const Args &args){
		os << "Scale: " << args.scale << endl;
		if (args.resize_src)
			os << "Resized source: (" << args.width << ", " << args.height
					<< ")\n";
		os << "Group threshold: " << args.gr_threshold << endl;
		os << "Levels number: " << args.nlevels << endl;
		os << "Win width: " << args.win_width << endl;
		os << "Win stride: (" << args.win_stride_width << ", "
				<< args.win_stride_height << ")\n";
		os << "Hit threshold: " << args.hit_threshold << endl;
		os << "Gamma correction: " << args.gamma_corr << endl;
		os << endl;
		return os;
}

#endif /* ARGS_H_ */
