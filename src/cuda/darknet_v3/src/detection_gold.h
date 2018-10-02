/*
 * DetectionGold.h
 *
 *  Created on: 27/09/2018
 *      Author: fernando
 */

#ifndef DETECTIONGOLD_H_
#define DETECTIONGOLD_H_

#include "network.h" //save layer
#include "layer.h" //save layer
#include "box.h" //boxes
#include "log_processing.h"

#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>

#define THRESHOLD_ERROR 1e-10
#define STORE_PRECISION 12

#define LAYER_GOLD "/var/radiation-benchmarks/data/"

#define PR_THRESHOLD 0.5
#define CONSIDERING_DETECTION 0.2

#define MAX_ERROR_COUNT 500 * 2

/**
 * AUXILIARY CLASSES
 */

struct Detection {
	std::vector<real_t> prob;
	box bbox;

	int nboxes;
	real_t objectness;
	int sort_class;
	int classes;

	Detection() :
			prob(std::vector<real_t>()), bbox(box()), nboxes(0), objectness(0), sort_class(
					0), classes(0) {
	}

	Detection(int classes, int nboxes, int sort_class, real_t objectness,
			std::vector<real_t> prob, box bb) :
			prob(prob), bbox(bb), nboxes(nboxes), objectness(objectness), sort_class(
					sort_class), classes(classes) {
	}

	Detection(const Detection& a) :
			prob(a.prob), bbox(a.bbox), nboxes(a.nboxes), objectness(
					a.objectness), sort_class(a.sort_class), classes(a.classes) {
	}

	Detection& operator=(const Detection& other) // copy assignment
			{
		if (this != &other) { // self-assignment check expected
			this->prob = std::vector < real_t > (other.prob);
			this->bbox = other.bbox;
			this->nboxes = other.nboxes;
			this->objectness = other.objectness;
			this->sort_class = other.sort_class;
			this->classes = other.classes;
		}
		return *this;
	}

};

struct GoldHash {
	std::unordered_map<std::string, std::vector<Detection> > data;

	std::vector<Detection> operator[](std::string img) {
		return this->data[img];
	}

	void put(std::string img, std::vector<Detection> a) {
		std::pair<std::string, std::vector<Detection> > tmp(img, a);
		this->data.insert(tmp);
	}
};

struct DetectionGold {
	std::string network_name;
	std::string gold_inout;
	bool generate;
	real_t thresh, hier_thresh;
	int plist_size;
	std::string img_list_path, config_file, cfg_data, model, weights;
	std::vector<std::string> gold_img_names;
	int iterations, tensor_core_mode, stream_mr;
	int total_errors;

	//gold atribute
	GoldHash gold_hash_var;

	Log* app_log;

	DetectionGold(int argc, char **argv, real_t thresh, real_t hier_thresh,
			char *img_list_path, char *config_file, char *config_data,
			char *model, char *weights);

	virtual ~DetectionGold();

	int run(detection* dets, int nboxes, int img_index, int classes);

	void start_iteration();
	void end_iteration();

	void load_gold_hash(std::ifstream& gold_file);

	void write_gold_header();

	void gen(detection* dets, int nboxes, int img_index,
			std::ofstream& gold_file, int classes);
	int cmp(detection* dets, int nboxes, int img_index, int classes);

	//log functions
//	void start_log(std::string gold, int save_layer, int abft, int iterations,
//			std::string app, unsigned char use_tensor_core_mode);
//
//	void end_iteration_app();
//
//	void start_iteration_app();
//
//	void update_timestamp_app();
//
//	void log_error_info(std::string error_detail);
//
//	void update_error_count(long error_count);
};

#endif /* DETECTIONGOLD_H_ */

