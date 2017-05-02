/*
 * log_processing.cpp
 *
 *  Created on: 30/04/2017
 *      Author: fernando
 */

#include "log_processing.h"
#include <fstream>
#include <list>

#ifdef LOGS
#include "log_helper.h"
#endif

double mysecond() {
	struct timeval tp;
	struct timezone tzp;
	gettimeofday(&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

void start_count_app(char *test, char *app) {
#ifdef LOGS
	char test_info[500];
	snprintf(test_info, 500, "gold_file: %s", test);

	start_log_file(app, test_info);
#endif
}

void finish_count_app() {
#ifdef LOGS
	end_log_file();
#endif
}

void saveLayer(network net, int iterator, int n) {

}

void compareLayer(layer l, int i) {

}

const char** get_image_filenames(char *img_list_path, int *image_list_size) {
	std::list < std::string > data;
	std::string line;
	std::ifstream img_list_file(img_list_path);
	if (img_list_file.is_open()) {
		while (getline(img_list_file, line)) {
			data.push_back(line);
		}
		img_list_file.close();
	}
	const char** array = new const char*[data.size()];

	for (std::list<std::string>::const_iterator it = data.begin(); it != data.end();
			++it) {
		array[*image_list_size] = it->c_str();
		(*image_list_size)++;
	}

	// use the array

	return array;
}

/**
 * it was adapted from max_index in utils.c line 536
 * auxiliary funtion for save_gold
 */

inline int get_index(float *a, int n) {
	if (n <= 0)
		return -1;
	int i, max_i = 0;
	float max = a[0];
	for (i = 1; i < n; ++i) {
		if (a[i] > max) {
			max = a[i];
			max_i = i;
		}
	}
	return max_i;
}

/**
 * it was addapted from draw_detections in image.c line 174
 * it saves a gold file for radiation tests
 */
void save_gold(FILE *fp, image im, int num, float thresh, box *boxes,
		float **probs, int classes) {
	int i;

	for (i = 0; i < num; ++i) {
		int class_ = get_index(probs[i], classes);
		float prob = probs[i][class_];
		if (prob > thresh) {
			int width = im.h * .012;
//			printf("%d: %.0f%%\n", class_, prob * 100);
//			int offset = class_ * 123457 % classes;
//			float red = get_color(2, offset, classes);
//			float green = get_color(1, offset, classes);
//			float blue = get_color(0, offset, classes);
//			float rgb[3];

//			rgb[0] = red;
//			rgb[1] = green;
//			rgb[2] = blue;
			box b = boxes[i];

			float left = (b.x - b.w / 2.) * float(im.w);
			float right = (b.x + b.w / 2.) * float(im.w);
			float top = (b.y - b.h / 2.) * float(im.h);
			float bot = (b.y + b.h / 2.) * float(im.h);

			if (left < 0)
				left = 0;
			if (right > im.w - 1)
				right = im.w - 1;
			if (top < 0)
				top = 0;
			if (bot > im.h - 1)
				bot = im.h - 1;

			//will save to fp file in this order
			//class number, left, top, right, bottom, prob (confidence)
			fprintf(fp, "%d %f %f %f %f %f\n", class_, left, top, right, bot,
					prob);

		}
	}
}
