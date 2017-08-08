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

const char *ABFT_TYPES[] = {"none", "gemm", "smart_pooling", "l1", "l2", "trained_weights"};

void start_count_app(char *test, int save_layer, int abft, int iterations,
		char *app) {
#ifdef LOGS
	char save_layer_char[10];
	char iterations_char[50];
	sprintf(save_layer_char, "%d", save_layer);
	sprintf(iterations_char, "%d", iterations);

	std::string test_info = std::string("gold_file: ") + std::string(test) +
	" save_layer: " + save_layer_char + " abft_type: " +
	ABFT_TYPES[abft] + " iterations: " + iterations_char;

	start_log_file(app, const_cast<char*>(test_info.c_str()));
#endif
}

void finish_count_app() {
#ifdef LOGS
	end_log_file();
#endif
}

void start_iteration_app() {
#ifdef LOGS
	start_iteration();
#endif
}

void end_iteration_app() {
#ifdef LOGS
	end_iteration();
#endif
}

/**
 * support function only to check if two layers have
 * the same value
 */
inline bool compare_layer(float *l1, float *l2, int n) {
	for (int i = 0; i < n; i++) {
		float diff = fabs(l1[i] - l2[i]);
		if (diff > LAYER_THRESHOLD_ERROR) {
//			printf("passou  onde nao devia %f\n\n", diff);
			return true;
		}
	}
	return false;
}

void alloc_gold_layers_arrays(detection *det, network *net) {
	det->net = net;
	int layers_size = det->net->n;
	layer *layers = det->net->layers;
	det->gold_layers = (float**) calloc(layers_size, sizeof(float*));

	if (det->gold_layers != NULL) {

		for (int i = 0; i < layers_size; i++) {
			layer l = layers[i];

			det->gold_layers[i] = (float*) calloc(l.outputs, sizeof(float));
		}
	}
#ifdef GPU
	det->found_layers = (float**) calloc(layers_size, sizeof(float*));
	if(det->found_layers != NULL) {
		for (int i = 0; i < layers_size; i++) {
			layer l = layers[i];

			det->found_layers[i] = (float*) calloc(l.outputs, sizeof(float));

		}
	}
#endif

}

void delete_gold_layers_arrays(detection det) {
	float **gold_layers = det.gold_layers;
	float **found_layers = det.found_layers;
	int layers_size = det.net->n;

	for (int i = 0; i < layers_size; i++) {
		if (gold_layers[i])
			free(gold_layers[i]);
#ifdef GPU
		if (found_layers[i])
		free(found_layers[i]);
#endif
	}
	free(gold_layers);
#ifdef GPU
	free(found_layers);
#endif

}

inline char* get_small_log_file(char *log_file) {
	std::string temp(log_file);
	std::vector < std::string > ret_array = split(temp, '/');
	std::string str_ret = ret_array[ret_array.size() - 1];
	char *ret = (char*) calloc(str_ret.size(), sizeof(char));
	strcpy(ret, str_ret.c_str());
	return ret;
}

FILE* open_layer_file(char *output_filename, const char *mode) {
	FILE* fp = NULL;
	if ((fp = fopen(output_filename, mode)) == NULL) {
		printf("ERROR ON OPENING %s file\n", output_filename);
		exit(-1);
	}
	return fp;
}

void save_layer(detection *det, int img_iterator, int test_iteration,
		char *log_filename, int generate) {
	int layers_size = det->net->n;

	FILE *output_file, *gold_file;
	char *small_log_file = get_small_log_file(log_filename);
	for (int i = 0; i < layers_size; i++) {
		char output_filename[500];
		sprintf(output_filename, "%s%s_layer_%d_img_%d_test_it_%d.layer",
		LAYER_GOLD, small_log_file, i, img_iterator, test_iteration);

		layer l = det->net->layers[i];
		float *output_layer;

#ifdef GPU
		cudaMemcpy (det->found_layers[i], l.output_gpu, l.outputs*sizeof(float), cudaMemcpyDeviceToHost);
		output_layer = det->found_layers[i];
#else
		output_layer = l.output;
#endif
		//if generate is set no need to compare
		if (!generate) {
			//open gold
			char gold_filename[500];
			sprintf(gold_filename, "%sgold_layer_darknet_v2_%d_img_%d_test_it_0.layer",
			LAYER_GOLD, i, img_iterator);

			gold_file = open_layer_file(gold_filename, "r");
			if (l.outputs
					!= fread(det->gold_layers[i], sizeof(float), l.outputs,
							gold_file)) {
				printf("ERROR ON READ size %s\n", gold_filename);
				fclose(gold_file);
				exit(-1);
			}
			fclose(gold_file);

			if (compare_layer(det->gold_layers[i], output_layer, l.outputs)) {

				output_file = open_layer_file(output_filename, "w");
				fwrite(output_layer, sizeof(float), l.outputs, output_file);
				fclose(output_file);
			}

		} else {
			output_file = open_layer_file(output_filename, "w");
			fwrite(output_layer, sizeof(float), l.outputs, output_file);
			fclose(output_file);
		}

	}

	free(small_log_file);

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
//	fprintf(fp, "%s\n", img);
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
	arg->img_list_path = (char*) calloc(split_ret[3].size(), sizeof(char)); //const_cast<char*>(split_ret[3].c_str());
	arg->config_file = (char*) calloc(split_ret[4].size(), sizeof(char)); //const_cast<char*>(split_ret[4].c_str());
	arg->cfg_data = (char*) calloc(split_ret[5].size(), sizeof(char)); //const_cast<char*>(split_ret[5].c_str());
	arg->model = (char*) calloc(split_ret[6].size(), sizeof(char)); //const_cast<char*>(split_ret[6].c_str());
	arg->weights = (char*) calloc(split_ret[7].size(), sizeof(char));

	strcpy(arg->img_list_path, split_ret[3].c_str());
	strcpy(arg->config_file, split_ret[4].c_str());
	strcpy(arg->cfg_data, split_ret[5].c_str());
	strcpy(arg->model, split_ret[6].c_str());
	strcpy(arg->weights, split_ret[7].c_str());
	gold.total = atoi(split_ret[8].c_str());
	gold.classes = atoi(split_ret[9].c_str());

//allocate detector
	gold.img_names = (char**) calloc(gold.plist_size, sizeof(char*));
	gold.pb_gold = (prob_array*) calloc(gold.plist_size, sizeof(prob_array));

	for (int i = 0; i < gold.plist_size && getline(img_list_file, line); i++) {
		line.erase(line.size() - 1);
		gold.img_names[i] = (char*) calloc(line.size(), sizeof(char));
		std::vector < string > line_splited = split(line, ';');
		strcpy(gold.img_names[i], line_splited[0].c_str());

		gold.pb_gold[i] = load_prob_array(gold.total, gold.classes,
				img_list_file);
	}

	return gold;
}

void delete_detection_var(detection *det, Args *arg) {
	free(arg->weights);
	for (int i = 0; det->plist_size > i; i++) {
		if (det->img_names[i])
			free(det->img_names[i]);
		if (det->pb_gold[i].boxes)
			free(det->pb_gold[i].boxes);

		for (int j = 0; j < det->total; j++) {
			if (det->pb_gold[i].probs[j])
				free(det->pb_gold[i].probs[j]);
		}
		if (det->pb_gold[i].probs)
			free(det->pb_gold[i].probs);

	}

	if (det->img_names)
		free(det->img_names);

	if (arg->save_layers) {
		delete_gold_layers_arrays(*det);
	}
}

void clear_boxes_and_probs(box *boxes, float **probs, int n, int m) {
	memset(boxes, 0, sizeof(box) * n);
	for (int i = 0; i < n; i++) {
		memset(probs[i], 0, sizeof(float) * m);
	}
}

inline void print_box(box b) {
	std::cout << b.x << " " << b.y << " " << b.w << " " << b.h << "\n";
}

void print_detection(detection det) {
	for (int i = 0; i < det.plist_size; i++) {
		std::cout << det.img_names[i] << "\n";
		prob_array p = det.pb_gold[i];
		for (int j = 0; j < det.total; j++) {
			print_box(p.boxes[j]);
//			for (int k = 0; k < det.classes; k++) {
//				std::cout << p.probs[j][k] << " ";
//			}
			std::cout << "\n";
		}
	}

}

inline bool error_check(char *error_detail, float f_pb, float g_pb, box f_b,
		box g_b, char* img, int class_, int pb_i) {
	float diff_float[3] = { (float) fabs(f_b.x - g_b.x), (float) fabs(
			f_b.y - g_b.y), (float) fabs(f_pb - g_pb) };
	int diff_int[3] = { abs(f_b.h - g_b.h), abs(f_b.w - g_b.w), 0 };
	bool diff = false;
	for (int i = 0; i < 3; i++) {
		if (diff_float[i] > THRESHOLD_ERROR)
			diff = true;

		if (diff_int[i] > 0)
			diff = true;
	}

	if (diff)
		sprintf(error_detail, "img: [%s]"
				" prob[%d][%d] r:%1.16e e:%1.16e"
				" x_r: %1.16e x_e: %1.16e"
				" y_r: %1.16e y_e: %1.16e"
				" w_r: %1.16e w_e: %1.16e"
				" h_r: %1.16e h_e: %1.16e", img, pb_i, class_, f_pb, g_pb,
				f_b.x, g_b.x, f_b.y, g_b.y, f_b.w, g_b.w, f_b.h, g_b.h);

	return diff;
}

void compare(detection *det, float **f_probs, box *f_boxes, int num,
		int classes, int img, int save_layers, int test_iteration) {

//	network *net = det->net;
	prob_array gold = det->pb_gold[img];
	float **g_probs = gold.probs;
	box *g_boxes = gold.boxes;
	char* img_string = det->img_names[img];

	int error_count = 0;
	for (int i = 0; i < num; ++i) {
		int class_ = get_index(g_probs[i], classes);
		float g_prob = g_probs[i][class_];
		float f_prob = f_probs[i][class_];
		box g_b = g_boxes[i];
		box f_b = f_boxes[i];

		char error_detail[1000];
		if (error_check(error_detail, f_prob, g_prob, f_b, g_b, img_string,
				class_, i)) {
			error_count++;

#ifdef LOGS
			log_error_detail(error_detail);
#else
			printf("%s\n", error_detail);
#endif

		}

	}

#ifdef LOGS
	log_error_count(error_count);

	printf("%d errors found at %s detection\n", error_count, img_string);
//save layers here
	if(error_count && save_layers) {
		save_layer(det, img, test_iteration, get_log_file_name(), 0);
	}
#endif

}
