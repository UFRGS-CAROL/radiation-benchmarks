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
 * it was adapted from draw_detections in image.c line 174
 * it saves a gold file for radiation tests
 */
void save_gold(FILE *fp, char *img, int num, int classes, float **probs,
		box *boxes) {
	fprintf(fp, "%s\n", img);
	for (int i = 0; i < num; ++i) {
		int class_ = get_index(probs[i], classes);
		float prob = probs[i][class_];
		box b = boxes[i];
		fprintf(fp, "%f;%f;%f;%f;%f;%d;\n", prob, b.x, b.y, b.w, b.h, class_);

	}

}

prob_array load_prob_array(int num, int classes, std::ifstream &ifp) {
	prob_array ret;
	ret.boxes = (box*) calloc(num, sizeof(box));
	ret.probs = (float**) calloc(num, sizeof(float*));
	for (int i = 0; i < num; i++) {
		ret.probs[i] = (float*) calloc(classes, sizeof(float));
	}

	std::string line;
	std::vector < std::string > splited;
	for (int i = 0; i < num; ++i) {
		getline(ifp, line);
		splited = split(line, ';');

		box b;
		b.x = atof(splited[1].c_str());
		b.y = atof(splited[2].c_str());
		b.w = atof(splited[3].c_str());
		b.h = atof(splited[4].c_str());
		int class_ = atof(splited[5].c_str());


		ret.probs[i][class_] = atof(splited[0].c_str());

		ret.boxes[i] = b;
	}
	return ret;
}

detection load_gold(Args *arg) {
	detection gold;
	std::string line;
	std::ifstream img_list_file(arg->gold_inout);
	if (img_list_file.is_open()) {
		getline(img_list_file, line);
	} else {
		std::cout << "ERROR ON OPENING GOLD FILE\n";
		exit(-1);
	}

	std::vector < std::string > split_ret = split(line, ';');
//	0       1           2              3              4            5            6
//	thresh; hier_tresh; img_list_size; img_list_path; config_file; config_data; model;weights;total;classes;
	arg->thresh = atof(split_ret[0].c_str());
	arg->hier_thresh = atof(split_ret[1].c_str());
	gold.plist_size = atoi(split_ret[2].c_str());
	arg->img_list_path = const_cast<char*>(split_ret[3].c_str());
	arg->config_file = const_cast<char*>(split_ret[4].c_str());
	arg->cfg_data = const_cast<char*>(split_ret[5].c_str());
	arg->model = const_cast<char*>(split_ret[6].c_str());
	arg->weights = (char*) calloc(split_ret[7].size(), sizeof(char));

	strcpy(arg->weights, split_ret[7].c_str());
	gold.total = atoi(split_ret[8].c_str());
	gold.classes = atoi(split_ret[9].c_str());

	//allocate detector
	gold.img_names = (char**) calloc(gold.plist_size, sizeof(char*));
	gold.pb_gold = (prob_array*) calloc(gold.plist_size, sizeof(prob_array));

	for (int i = 0; i < gold.plist_size && getline(img_list_file, line); i++) {
		gold.img_names[i] = const_cast<char*>(line.c_str());
		gold.pb_gold[i] = load_prob_array(gold.total, gold.classes, img_list_file);
	}

	return gold;
}

void delete_detection_var(detection *det, Args *arg) {

}

void clear_boxes_and_probs(box *boxes, float **probs, int n, int m) {
	memset(boxes, 0, sizeof(box) * n);
	for (int i = 0; i < n; i++) {
		memset(probs[i], 0, sizeof(float) * m);
	}
}

void print_detection(detection det) {

}

void compare() {

#ifdef LOGS
	log_error_detail(error_detail);
#endif
#ifdef LOGS
	log_error_count(errors);

	//save layers here
	if(errors && save_layer) {
		save_layer(net, img_iteration, test_iteration, get_log_file_name());
	}
#endif

}
