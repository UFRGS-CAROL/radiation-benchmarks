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

#if REAL_TYPE == HALF
#define THRESHOLD_ERROR 1e-2
#define STORE_PRECISION 4

#elif REAL_TYPE == FLOAT
#define THRESHOLD_ERROR 1e-3
#define STORE_PRECISION 7

#elif REAL_TYPE == DOUBLE
#define THRESHOLD_ERROR 1e-5
#define STORE_PRECISION 9

#endif

#define THRESHOLD_ERROR_INTEGER 1
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
	int iterations, stream_mr;
	unsigned char tensor_core_mode;
	int total_errors;
	bool normalized_coordinates;

	//gold atribute
	GoldHash gold_hash_var;

	Log* app_log;

	DetectionGold(int argc, char **argv, real_t thresh, real_t hier_thresh,
			char *img_list_path, char *config_file, char *config_data,
			char *model, char *weights);

	virtual ~DetectionGold();

	int run(detection** dets, int* nboxes, int img_index, int classes, int img_w,
			int img_h);

	void start_iteration();
	void end_iteration();

	void load_gold_hash(std::ifstream& gold_file);

	void write_gold_header();

	void gen(detection* dets, int nboxes, int img_index,
			std::ofstream& gold_file, int classes);
	int cmp(detection* dets, int nboxes, int img_index, int classes, int img_w,
			int img_h, int inet);

	std::ostringstream generate_gold_line(int bb, detection det, const box& b,
			detection* dets);

	Detection load_gold_line(std::ifstream& gold_file, int nboxes);

	int compare_line(real_t g_objectness, real_t f_objectness,
			int g_sort_class, int f_sort_class, const box& g_box, const box& f_box,
			const std::string& img, int nb, int classes, const real_t* g_probs,
			const real_t* f_probs, int img_w, int img_h, int inet);
};

#endif /* DETECTIONGOLD_H_ */

