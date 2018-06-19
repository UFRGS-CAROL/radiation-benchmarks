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
#include <cmath>
#include <iostream>
#ifdef LOGS
#include "log_helper.h"

#endif

void start_count_app(char *test, int save_layer, int abft, int iterations,
		char *app, unsigned char use_tensor_core_mode) {
#ifdef LOGS
	char save_layer_char[10];
	char iterations_char[50];
	sprintf(save_layer_char, "%d", save_layer);
	sprintf(iterations_char, "%d", iterations);


	std::string test_info = std::string("gold_file: ") + std::string(test) +
	" save_layer: " + save_layer_char + " abft_type: " +
	ABFT_TYPES[abft] + " iterations: " + iterations_char + " tensor_core_mode: " + std::to_string(int(use_tensor_core_mode));

	set_iter_interval_print(10);

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

void update_timestamp_app() {
#ifdef LOGS
	update_timestamp();
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
#ifdef GPU
	float **found_layers = det.found_layers;
#endif
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

inline std::string get_small_log_file(char *log_file) {
	std::string temp(log_file);
	std::vector<std::string> ret_array = split(temp, '/');
	std::string str_ret = ret_array[ret_array.size() - 1];
	return str_ret;
}

FILE* open_layer_file(char *output_filename, const char *mode) {
	FILE* fp = NULL;
	if ((fp = fopen(output_filename, mode)) == NULL) {
		printf("ERROR ON OPENING %s file\n", output_filename);
		exit(-1);
	}
	return fp;
}

inline void calc_real_coordinates(box b, image im, int *x_max, int *y_max,
		int *x_min, int *y_min) {

	int left = ((float) b.x - (float) b.w / 2.) * (float) im.w;
	int right = ((float) b.x + (float) b.w / 2.) * (float) im.w;
	int top = ((float) b.y + (float) b.h / 2.) * (float) im.h;
	int bot = ((float) b.y - (float) b.h / 2.) * (float) im.h;

	if (left < 0)
		left = 0;
	if (right > im.w - 1)
		right = im.w - 1;
	if (top < 0)
		top = 0;
	if (bot > im.h - 1)
		bot = im.h - 1;
	*x_max = right;
	*x_min = left;
	*y_max = top;
	*y_min = bot;

}

std::pair<float, float> online_precision_recall(std::vector<box> gold,
		std::vector<box> found, float threshold, image im) {
	float true_positive = 0;

	for (unsigned i = 0; i < gold.size(); i++) {
		//float x, y, w, h;
		int x_max_gold, y_max_gold, x_min_gold, y_min_gold;
		calc_real_coordinates(gold[i], im, &x_max_gold, &y_max_gold,
				&x_min_gold, &y_min_gold);

		for (unsigned z = 0; z < found.size(); z++) {
			int x_max_found, y_max_found, x_min_found, y_min_found;

			calc_real_coordinates(found[z], im, &x_max_found, &y_max_found,
					&x_min_found, &y_min_found);

			long intersection = 0;

			for (int x = x_min_found; x <= x_max_found; x++) {
				for (int y = y_min_found; y <= y_max_found; y++) {
					if ((x >= x_min_gold) && (x <= x_max_gold)
							&& (y >= y_min_gold) && (y <= y_max_gold)) {
						intersection++;
					}
				}
			}
			long total = (x_max_gold - x_min_gold) * (y_max_gold - y_min_gold)
					+ (x_max_found - x_min_found) * (y_max_found - y_min_found)
					- intersection;
			if (total) {
				if (((float) intersection / (float) total) >= threshold) {
					true_positive++;
					break;
				}
			}
		}

	}
	float false_negative = gold.size() - true_positive;
	float recall; // = true_positive / (true_positive + false_negative);

	if (true_positive + false_negative) {
		recall = true_positive / (true_positive + false_negative);
	} else {
		recall = 0;
	}

	float out_positive = 0;
	for (unsigned z = 0; z < found.size(); z++) {
		int x_max_found, y_max_found, x_min_found, y_min_found;

		calc_real_coordinates(found[z], im, &x_max_found, &y_max_found,
				&x_min_found, &y_min_found);

		for (unsigned i = 0; i < gold.size(); i++) {
			int x_max_gold, y_max_gold, x_min_gold, y_min_gold;
			calc_real_coordinates(gold[i], im, &x_max_gold, &y_max_gold,
					&x_min_gold, &y_min_gold);

			long intersection = 0;

			for (int x = x_min_gold; x <= x_max_gold; x++) {
				for (int y = y_min_gold; y <= y_max_gold; y++) {
					if ((x >= x_min_found) && (x <= x_max_found)
							&& (y >= y_min_found) && (y <= y_max_found)) {
						intersection++;
					}
				}
			}
			long total = (x_max_gold - x_min_gold) * (y_max_gold - y_min_gold)
					+ (x_max_found - x_min_found) * (y_max_found - y_min_found)
					- intersection;
			if (total) {
				if (((float) intersection / (float) total) >= threshold) {
					out_positive++;
					break;
				}
			}
		}
	}
	float false_positive = found.size() - out_positive;
	float precision; // = true_positive / (true_positive + false_positive);
	if (true_positive + false_positive) {
		precision = true_positive / (true_positive + false_positive);
	} else {
		precision = 0;
	}

	return std::pair<float, float>(precision, recall);
}

void save_layer(detection *det, int img_iterator, int test_iteration,
		char *log_filename, int generate, char *img_list_filename) {
	int layers_size = det->net->n;

	FILE *output_file, *gold_file;
	std::string small_log_file = get_small_log_file(log_filename);
	std::string img_list_filename_string(img_list_filename);
	std::vector<std::string> temp_splited = split(img_list_filename_string,
			'/');
	img_list_filename_string = temp_splited[temp_splited.size() - 1];

	for (int i = 0; i < layers_size; i++) {
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
			std::string gold_filename = std::string(LAYER_GOLD)
					+ img_list_filename_string + "_"
					+ std::string(det->network_name) + "_gold_layer_"
					+ std::to_string(i) + "_img_" + std::to_string(img_iterator)
					+ "_test_it_0.layer";

			gold_file = open_layer_file(
					const_cast<char*>(gold_filename.c_str()), "r");
			if (l.outputs
					!= fread(det->gold_layers[i], sizeof(float), l.outputs,
							gold_file)) {
				printf("ERROR ON READ size %s\n", gold_filename.c_str());
				fclose(gold_file);
				exit(-1);
			}
			fclose(gold_file);

			if (compare_layer(det->gold_layers[i], output_layer, l.outputs)) {
				std::string output_filename = std::string(LAYER_GOLD)
						+ std::string(small_log_file) + "_"
						+ std::string(det->network_name) + "_layer_"
						+ std::to_string(i) + "_img_"
						+ std::to_string(img_iterator) + "_test_it_"
						+ std::to_string(test_iteration) + ".layer";

				output_file = open_layer_file(
						const_cast<char*>(output_filename.c_str()), "w");
				fwrite(output_layer, sizeof(float), l.outputs, output_file);
				fclose(output_file);
			}

		} else {
			//open gold
			std::string gold_filename = std::string(LAYER_GOLD)
					+ img_list_filename_string + "_"
					+ std::string(det->network_name) + "_gold_layer_"
					+ std::to_string(i) + "_img_" + std::to_string(img_iterator)
					+ "_test_it_0.layer";
			output_file = open_layer_file(
					const_cast<char*>(gold_filename.c_str()), "w");
			fwrite(output_layer, sizeof(float), l.outputs, output_file);
			fclose(output_file);
		}

	}

//	free(small_log_file);

}

char** get_image_filenames(char *img_list_path, int *image_list_size) {
	std::list<std::string> data;
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
//			printf("in time max %f\n", max);

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
//	for (int i = 0; i < num; ++i) {
//		int class_ = get_index(probs[i], classes);
//		float prob = probs[i][class_];
//		box b = boxes[i];
//		fprintf(fp, "%f;%f;%f;%f;%f;%d;\n", prob, b.x, b.y, b.w, b.h, class_);
//
//	}
	std::vector<std::string> to_print;
	for (int i = 0; i < num; i++) {
		box b = boxes[i];
		int class_ = get_index(probs[i], classes);
//		for (int class_ = 0; class_ < classes; class_++) {
		float prob = probs[i][class_];
//			if (prob){
		fprintf(fp, "%f;%f;%f;%f;%f;%d;\n", prob, b.x, b.y, b.w, b.h, class_);
//				std::string str_to_print = std::to_string(prob) + ";" +
//						std::to_string(b.x) + ";" + std::to_string(b.y) + ";" +
//						std::to_string(b.w) + ";" + std::to_string(b.h) + ";" +
//						std::to_string(class_) + ";";
//				to_print.push_back(str_to_print);
//			}
//		}
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
	std::vector<std::string> splited;
	for (int i = 0; i < num; ++i) {

		getline(ifp, line);
		splited = split(line, ';');

		box b;
		b.x = atof(splited[1].c_str());
		b.y = atof(splited[2].c_str());
		b.w = atof(splited[3].c_str());
		b.h = atof(splited[4].c_str());
		int class_ = atoi(splited[5].c_str());

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

	std::vector<std::string> split_ret = split(line, ';');
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
		std::vector<string> line_splited = split(line, ';');
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
	for (int i = 0; i < n; i++) {
		boxes[i].x = 0;
		boxes[i].y = 0;
		boxes[i].w = 0;
		boxes[i].h = 0;
	}
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

inline std::string error_check(float f_pb, float g_pb, box f_b, box g_b,
		char* img, int class_g, int class_f, int pb_i) {

	std::vector<float> diff_array = { std::fabs(f_b.x - g_b.x),  //x axis
	std::fabs(f_b.y - g_b.y), //y axis
	std::fabs(f_pb - g_pb), //probabilities
	std::fabs(f_b.h - g_b.h), // height
	std::fabs(f_b.w - g_b.w), (float) std::abs(class_g - class_f) }; // width and class

//	if (class_g != class_f)
//		std::cout << " val " << class_g << " " << class_f << "\n";
	bool diff = false;
	for (auto diff_element : diff_array) {
		if (diff_element > THRESHOLD_ERROR)
			diff = true;
	}

	if (diff) {
		char error_detail[1000];

		sprintf(error_detail, "img: [%s]"
				" prob_r[%d][%d]: %1.16e"
				" prob_e[%d][%d]: %1.16e"
				" x_r: %1.16e x_e: %1.16e"
				" y_r: %1.16e y_e: %1.16e"
				" w_r: %1.16e w_e: %1.16e"
				" h_r: %1.16e h_e: %1.16e", img, pb_i, class_f, f_pb, pb_i,
				class_g, g_pb, f_b.x, g_b.x, f_b.y, g_b.y, f_b.w, g_b.w, f_b.h,
				g_b.h);
		return std::string(error_detail);
	}

	return "";
}

int compare(detection *det, float **f_probs, box *f_boxes, int num, int classes,
		int img, int save_layers, int test_iteration, char *img_list_path,
		error_return max_pool_errors, image im, int stream_mr) {
//	printf("passou no compare\n");

//	network *net = det->net;
	prob_array gold = det->pb_gold[img];
	float **g_probs = gold.probs;
	box *g_boxes = gold.boxes;
	char* img_string = det->img_names[img];
//	printf("sssss passou no compare\n");

	// Check PR if critical
	std::list<box> found_boxes;
	std::list<box> gold_boxes;

	int error_count = 0;
	for (int i = 0; i < num; i++) {
//		printf("the error is here %p\n", g_boxes);
		box g_b = g_boxes[i];
//		printf("the error is here %p\n", f_boxes);		
		box f_b = f_boxes[i];
//		printf("before get index\n");
		int class_g = get_index(g_probs[i], classes);
		int class_f = get_index(f_probs[i], classes);

		float g_prob = g_probs[i][class_g];
		float f_prob = f_probs[i][class_f];

//		printf("Gold box dddd\n");
//		print_box(g_b);

//		printf("Found box\n");
//		print_box(g_b);
//		printf("Gold prob %f found prob %f\n", g_prob, f_prob);


		if (f_prob >= CONSIDERING_DETECTION) {
			found_boxes.push_back(f_b);
		}
		if (g_prob >= CONSIDERING_DETECTION) {
			gold_boxes.push_back(g_b);
		}

		std::string error_detail = error_check(f_prob, g_prob, f_b, g_b,
				img_string, class_g, class_f, i);


		if (error_detail != "") {
			error_count++;
			error_detail += " stream_modular_redundancy " + std::to_string(stream_mr);
#ifdef LOGS
			log_error_detail(const_cast<char*>(error_detail.c_str()));
#else
			printf("%s\n", error_detail);
#endif

		}

	}

	//set to false for log_info detail
	//even if no SDC is detected it will write if smart pooling worked
	//log smart pooling operation
	bool found = false;
	std::string abft_error_info = "";
	for (size_t i = 0; i < max_pool_errors.err_detected_size; i++) {
		abft_error_info += "error_detected[" + std::to_string(i) + "]: "
				+ std::to_string(max_pool_errors.error_detected[i]) + " ";
		if (max_pool_errors.error_detected[i] != 0) {
			found = true;
		}
	}

#ifdef LOGS
	log_error_count(error_count);
	if(found) {
		log_info_detail(const_cast<char*>(abft_error_info.c_str()));
	}

	//printf("%d errors found at %s detection\n", error_count, img_string);

	//save layers here
	std::vector<box> gold_boxes_vector(std::begin(gold_boxes), std::end(gold_boxes));
	std::vector<box> found_boxes_vector(std::begin(found_boxes), std::end(found_boxes));

	if (error_count != 0) {
		std::pair<float, float> pr = online_precision_recall(gold_boxes_vector, found_boxes_vector,
				PR_THRESHOLD, im);

		if(save_layers && ((pr.first != 1.0) || (pr.second != 1.0))) {
			save_layer(det, img, test_iteration, get_log_file_name(), 0, img_list_path);
		}
		printf("ONLINE PR %f %f\n", pr.first, pr.second);
	}
#endif
	return error_count;

}

