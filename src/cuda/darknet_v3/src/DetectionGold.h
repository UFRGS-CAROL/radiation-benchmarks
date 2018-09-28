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

typedef std::vector<std::tuple<box, int, std::vector<real_t>, std::vector<real_t>, real_t,
		int, int> >gold_tuple_array;

typedef std::unordered_map<std::string, gold_tuple_array> gold_hash;

#define THRESHOLD_ERROR 0.05
#define LAYER_THRESHOLD_ERROR 0.0000001

#define LAYER_GOLD "/var/radiation-benchmarks/data/"

#define PR_THRESHOLD 0.5
#define CONSIDERING_DETECTION 0.2

#define MAX_ERROR_COUNT 500 * 2

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
	gold_hash gold_hash_var;


	DetectionGold(int argc, char **argv, real_t thresh, real_t hier_thresh,
			char *img_list_path, char *config_file, char *config_data,
			char *model, char *weights);

	virtual ~DetectionGold();

	void compare_or_generate(detection *dets, int nboxes, int img_index,
			network& net);



private:
	/**
	 * it was adapted from draw_detections in network.cu
	 * it saves a gold file for radiation tests
	 */
	void save_gold_img_i(detection *dets, int nboxes, int classes,
			std::ofstream& gold_file, int l_coords);

	void load_gold_hash(std::ifstream& gold_file);

	std::string print_box(box b);

	void write_gold_header();
	void generate_method(int img_index, int nboxes, network& net,
			detection* dets);
	void compare_method(int nboxes, detection* dets, std::string img, int l_coord);
};

#endif /* DETECTIONGOLD_H_ */

