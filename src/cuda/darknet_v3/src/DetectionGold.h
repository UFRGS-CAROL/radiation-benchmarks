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

#define THRESHOLD_ERROR 0.05
#define LAYER_THRESHOLD_ERROR 0.0000001

#define LAYER_GOLD "/var/radiation-benchmarks/data/"

#define PR_THRESHOLD 0.5
#define CONSIDERING_DETECTION 0.2

#define MAX_ERROR_COUNT 500 * 2

struct ProbArray {
	std::vector<box> boxes;
	std::vector<std::vector<real_t> > probs;

	void read_prob_array(int num, int classes, std::ifstream &ifp) ;
};

class DetectionGold {
public:
	std::string network_name;
	std::string gold_inout;
	bool generate;
	real_t thresh, hier_thresh;
	int plist_size, total, classes;
	std::string img_list_path, config_file, cfg_data, model, weights;
	std::vector<std::string> gold_img_names;
	std::vector<ProbArray> pb_gold;
	int iterations, tensor_core_mode, stream_mr;

	// For logging functions
	Log *app_logging;

	DetectionGold(int argc, char **argv, real_t thresh,
			real_t hier_thresh, int img_list_size, char *img_list_path,
			char *config_file, char *config_data, char *model, char *weights,
			int total, int classes);
	void write_gold_header();
	/**
	 * it was adapted from draw_detections in image.c line 174
	 * it saves a gold file for radiation tests
	 */
	void save_gold_img_i(char *img, int num, int classes, real_t **probs,
			box *boxes);

	virtual ~DetectionGold();

	void clear_boxes_and_probs(box *boxes, real_t **probs, int n, int m);

	std::string error_check(real_t f_pb, real_t g_pb, box f_b, box g_b,
			std::string img, int class_g, int class_f, int pb_i);

	bool compare(real_t **f_probs, box *f_boxes, int num, int classes, int img,
			int save_layers, int test_iteration);

	std::string print_box(box b);

	friend std::ostream& operator<<(std::ostream& ret, DetectionGold &det);
	/**
	 * it was adapted from max_index in utils.c line 536image load_image(char *filename, int w, int h, int c);
	 * auxiliary funtion for save_gold
	 */
	int get_index(real_t *a, int n);

};

#endif /* DETECTIONGOLD_H_ */

