/*
 * log_processing.cpp
 *
 *  Created on: 30/04/2017
 *      Author: fernando
 */

#include "log_processing.h"
#include <fstream>
#include <list>
#include <vector>
#include <string>
#include "helpful.h"

#include <iostream>
#ifdef LOGS
#include "log_helper.h"
#endif

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

void save_layer(network net, int img_iterator, int test_iteration,
		char *log_filename) {

}

rectangle init_rectangle(int class_, float left, float top, float right,
		float bottom, float prob) {
	rectangle temp;
	temp.left = left;
	temp.top = top;
	temp.right = right;
	temp.bottom = bottom;
	temp.prob = prob;
	temp.class_ = class_;
	return temp;
}

char** get_image_filenames(char *img_list_path, int *image_list_size) {
	std::list < std::string > data;
	std::string line;
	std::ifstream img_list_file(img_list_path);
	if (img_list_file.is_open()) {
		while (getline(img_list_file, line)) {
			data.push_back(line);
		}
		img_list_file.close();
	}
	char** array = (char**) malloc(sizeof(char*) * data.size());

	for (std::list<std::string>::const_iterator it = data.begin();
			it != data.end(); ++it) {
		char *temp = (char*) malloc(it->size() * sizeof(char));
		strcpy(temp, it->c_str());
		array[*image_list_size] = temp;
		(*image_list_size)++;
	}

	// use the array

	return array;
}

/**
 * it was adapted from max_index in utils.c line 536image load_image(char *filename, int w, int h, int c);
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
void save_gold(FILE *fp, int w, int h, int num, float thresh, box *boxes,
		float **probs, int classes) {
	for (int i = 0; i < num; ++i) {
		int class_ = get_index(probs[i], classes);
		float prob = probs[i][class_];
		if (prob > thresh) {
//			int width = h * .012;
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

			float left = (b.x - b.w / 2.) * float(w);
			float right = (b.x + b.w / 2.) * float(w);
			float top = (b.y - b.h / 2.) * float(h);
			float bot = (b.y + b.h / 2.) * float(h);

			if (left < 0)
				left = 0;
			if (right > w - 1)
				right = w - 1;
			if (top < 0)
				top = 0;
			if (bot > h - 1)
				bot = h - 1;

			//will save to fp file in this order
			//class number, left, top, right, bottom, prob (confidence)
			fprintf(fp, "%d;%f;%f;%f;%f;%f;\n", class_, left, top, right, bot,
					prob);

		}
	}
}

detection load_gold(Args *arg) {
	detection gold;
	std::list < std::string > data;
	std::string line;
	std::ifstream img_list_file(arg->gold_inout);
	if (img_list_file.is_open()) {
		while (getline(img_list_file, line)) {
			data.push_back(line);
		}
	} else {
		std::cout << "ERROR ON OPENING GOLD FILE\n";
		exit(-1);
	}

//	reading only header
	std::string header = data.front();

	data.pop_front();

	std::vector < std::string > split_ret = split(header, ';');
//	0       1           2              3              4            5            6
//	thresh; hier_tresh; img_list_size; img_list_path; config_file; config_data; model;weights
	arg->thresh = atof(split_ret[0].c_str());
	arg->hier_thresh = atof(split_ret[1].c_str());
	gold.img_list_size = atoi(split_ret[2].c_str());
	arg->img_list_path = const_cast<char*>(split_ret[3].c_str());
	arg->config_file = const_cast<char*>(split_ret[4].c_str());
	arg->cfg_data = const_cast<char*>(split_ret[5].c_str());
	arg->model = const_cast<char*>(split_ret[6].c_str());
	arg->weights = (char*) calloc(split_ret[7].size(), sizeof(char));

	strcpy(arg->weights, split_ret[7].c_str());

	//fill gold content
	gold.image_names = (char**) calloc(gold.img_list_size, sizeof(char*));
	//the first size of gold rectangles is the img_list_size
	gold.detection_result = (rectangle**) calloc(gold.img_list_size,
			sizeof(rectangle*));

	//for rect list size
	gold.rect_list_size = (int*) calloc(gold.img_list_size, sizeof(int));

	int img_iterator = 0;
	std::list<rectangle> rectangle_aux;

	for (std::list<std::string>::const_iterator it = data.begin();
			it != data.end(); ++it) {
		std::string temp((*it).c_str());
		std::vector < string > rect_line = split(temp, ';');

		//if it is less than 2 if is a image path line
		if (rect_line.size() < 2) {
			if (rectangle_aux.size() != 0) {
				gold.detection_result[img_iterator] = (rectangle*) calloc(
						rectangle_aux.size(), sizeof(rectangle));

				//here I record the number of rectangles for each image
				gold.rect_list_size[img_iterator] = rectangle_aux.size();

				int i = 0;
				while (rectangle_aux.size()) {
					rectangle final_it = rectangle_aux.front();

					gold.detection_result[img_iterator][i] = final_it;
					i++;
					rectangle_aux.pop_front();
				}
			}
			if (it != data.begin())
				img_iterator++;

			gold.image_names[img_iterator] = (char*) calloc(rect_line[0].size(),
					sizeof(char));
			strcpy(gold.image_names[img_iterator], rect_line[0].c_str());
		} else {
			//class number, left, top, right, bottom, prob (confidence)
			//(int class_, float left, float top, float right, float bottom, float prob);
			rectangle rect = init_rectangle(atoi(rect_line[0].c_str()), // class
			atof(rect_line[1].c_str()), //left
			atof(rect_line[2].c_str()), //top
			atof(rect_line[3].c_str()), //right
			atof(rect_line[4].c_str()), //bottom
			atof(rect_line[5].c_str()) //prob
					);
			rectangle_aux.push_back(rect);

		}

	}

	//the last image rectangles
	if (rectangle_aux.size() != 0) {
		gold.detection_result[img_iterator] = (rectangle*) calloc(
				rectangle_aux.size(), sizeof(rectangle));

		//here I record the number of rectangles for each image
		gold.rect_list_size[img_iterator] = rectangle_aux.size();

		int i = 0;
		while (rectangle_aux.size()) {
			rectangle final_it = rectangle_aux.front();

			gold.detection_result[img_iterator][i] = final_it;
			i++;
			rectangle_aux.pop_front();
		}
	}

	return gold;
}

void delete_detection_var(detection *det, Args *arg) {
	if (det->image_names) {
		for (int i = 0; i < det->img_list_size; i++)
			free(det->image_names[i]);

		free(det->image_names);
	}

	if (det->detection_result) {
		for (int i = 0; i < det->img_list_size; i++)
			free(det->detection_result[i]);
		free(det->detection_result);
	}

	if (arg->weights)
		free(arg->weights);
}

void clear_boxes_and_probs(box *boxes, float **probs, int n, int m) {
	memset(boxes, 0, sizeof(box) * n);
	for (int i = 0; i < n; i++) {
		memset(probs[i], 0, sizeof(float) * m);
	}
}

void print_rectangle(rectangle ret) {
	std::cout << ret.class_ << ";" << ret.left << ";" << ret.top << ";"
			<< ret.right << ";" << ret.bottom << ";" << ret.prob << ";"
			<< std::endl;
}

void print_detection(detection det) {
	for (int i = 0; i < det.img_list_size; i++) {
		std::cout << det.image_names[i] << std::endl;
		for (int j = 0; j < det.rect_list_size[i]; j++) {
			print_rectangle(det.detection_result[i][j]);

		}
	}
}

inline void compare_rectangle(int img_pos, rectangle g, rectangle f, char *error_detail) {
	error_detail[0] = '#';

	float g_array[5] = { g.left, g.bottom, g.right, g.top, g.prob };
	float f_array[5] = { f.left, f.bottom, f.right, f.top, f.prob };
	int diff_class = abs(f.class_ - g.class_);

	bool diff = false;
	for (int i = 0; i < 5; i++) {

		if (abs(g_array[i] - f_array[i]) > THRESHOLD_ERROR)
			diff = true;
	}

	if (diff || diff_class > THRESHOLD_ERROR)
		sprintf(error_detail, "image_list_position: [%d]"
				" r_lef: %1.16e e_lef: %1.16e"
				" r_bot: %1.16e e_bot: %1.16e"
				" r_rig: %1.16e e_rig: %1.16e"
				" r_top: %1.16e e_top: %1.16e"
				" r_prb: %1.16e e_prb: %1.16e"
				" r_cls: %d e_cls: %d", img_pos, f.left, g.left, f.bottom,
				g.bottom, f.right, g.right, f.top, g.top, f.prob, g.prob,
				f.class_, g.class_);

}

void compare(rectangle *gold_rect, int classes, int num, float **found_probs,
		box *found_boxes, int img_iteration, network net, int test_iteration,
		int save_layer, float thresh, int w, int h) {
	int errors = 0;

	for (int i = 0; i < num; ++i) {
		int class_ = get_index(found_probs[i], classes);
		float prob = found_probs[i][class_];
		if (prob > thresh) {

			box b = found_boxes[i];
			rectangle r_found;
			rectangle r_gold = gold_rect[i];

			r_found.left = (b.x - b.w / 2.) * float(w);
			r_found.right = (b.x + b.w / 2.) * float(w);
			r_found.top = (b.y - b.h / 2.) * float(h);
			r_found.bottom = (b.y + b.h / 2.) * float(h);

			if (r_found.left < 0)
				r_found.left = 0;
			if (r_found.right > w - 1)
				r_found.right = w - 1;
			if (r_found.top < 0)
				r_found.top = 0;
			if (r_found.bottom > h - 1)
				r_found.bottom = h - 1;

			printf("r found\n");
			print_rectangle(r_found);
			printf("r gold\n");
			print_rectangle(r_gold);

			r_found.prob = prob;
			r_found.class_ = class_;
			char error_detail[500];
			compare_rectangle(img_iteration, r_gold, r_found, error_detail);

			printf("%s\n", error_detail);

			if (error_detail[0] != '#')
				errors++;

#ifdef LOGS
			log_error_detail(error_detail);
#endif
		}

	}
#ifdef LOGS
	log_error_count(errors);

	//save layers here
	if(errors && save_layer) {
		save_layer(net, img_iteration, test_iteration, get_log_file_name());
	}
#endif

}
