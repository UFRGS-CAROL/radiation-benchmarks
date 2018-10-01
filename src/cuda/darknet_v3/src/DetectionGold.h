/*
 * DetectionGold.h
 *
 *  Created on: 27/09/2018
 *      Author: fernando
 */

#ifndef DETECTIONGOLD_H_
#define DETECTIONGOLD_H_

#include "Log.h"
#include <vector>
#include "network.h" //save layer
#include "layer.h" //save layer
#include "box.h" //boxes

//	box bbox;
//	int classes;
//	real_t *prob;
//	real_t *mask;
//	real_t objectness;
//	int sort_class;
// int l_coord
#include <tuple>
#include <unordered_map>

#define THRESHOLD_ERROR 0.05
#define LAYER_THRESHOLD_ERROR 0.0000001

#define LAYER_GOLD "/var/radiation-benchmarks/data/"

#define PR_THRESHOLD 0.5
#define CONSIDERING_DETECTION 0.2

#define MAX_ERROR_COUNT 500 * 2

struct Detection {
//	std::vector<real_t> mask;
	std::vector<real_t> prob;
	box bbox;

	int nboxes;
	real_t objectness;
	int sort_class;

	Detection() :
			nboxes(0), objectness(0), sort_class(0), prob(
					std::vector<real_t>()), bbox(box()) {
	}

	Detection(int nboxes, int sort_class, real_t objectness,
			std::vector<real_t> prob, box bb) :
			nboxes(nboxes), objectness(objectness), sort_class(sort_class), prob(
					prob), bbox(bb) {
	}

	Detection(const Detection& a) :
			nboxes(a.nboxes), objectness(a.objectness), sort_class(
					a.sort_class), prob(a.prob), bbox(a.bbox) {
	}

	Detection& operator=(const Detection& other) // copy assignment
			{
		if (this != &other) { // self-assignment check expected
			this->prob = std::vector < real_t > (other.prob);
			this->bbox = other.bbox;
			this->nboxes = other.nboxes;
			this->objectness = other.objectness;
			this->sort_class = other.sort_class;
		}
		return *this;
	}

	friend std::ostream& operator<<(std::ostream& os, const Detection& b) {
		os << b.nboxes << " " << b.objectness << " " << b.sort_class << " "
				<< b.bbox.x << " " << b.bbox.y;
		return os;
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

class DetectionGold {
public:
	std::string network_name;
	std::string gold_inout;
	bool generate;
	real_t thresh, hier_thresh;
	int plist_size;
	std::string img_list_path, config_file, cfg_data, model, weights;
	std::vector<std::string> gold_img_names;
	int iterations, tensor_core_mode, stream_mr;

	// For logging functions
	Log *app_logging;

	//gold atribute
	GoldHash gold_hash_var;

	DetectionGold(int argc, char **argv, real_t thresh, real_t hier_thresh,
			char *img_list_path, char *config_file, char *config_data,
			char *model, char *weights);

	virtual ~DetectionGold();

	void run(detection* dets, int nboxes, int img_index, int classes);

	void start_iteration();
	void end_iteration();

private:
	void load_gold_hash(std::ifstream& gold_file);

	void write_gold_header();

	void gen(detection* dets, int nboxes, int img_index,
			std::ofstream& gold_file, int classes);
	void cmp(detection* dets, int nboxes, int img_index, int classes);
};

#endif /* DETECTIONGOLD_H_ */

