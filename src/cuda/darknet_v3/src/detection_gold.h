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

#include <string>
#include <utility>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <iostream>

#include <memory>

#ifdef BUILDPROFILER
#include "include/Profiler.h"
#include "include/NVMLWrapper.h"

#ifdef FORJETSON
#include "include/JTX2Inst.h"
#define OBJTYPE JTX2Inst
#else
#include "include/NVMLWrapper.h"
#define OBJTYPE NVMLWrapper
#endif // FORJETSON

#endif

#if REAL_TYPE == HALF
#define THRESHOLD_ERROR 1e-2
#define STORE_PRECISION 4

#elif REAL_TYPE == FLOAT
#define THRESHOLD_ERROR 1e-3
#define STORE_PRECISION 8

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

	friend std::ostream &operator<<(std::ostream& os, const Detection& det) {
		os << "BB -- x: " << det.bbox.x;
		os << " y:" << det.bbox.y;
		os << " h:" << det.bbox.h;
		os << " w:" << det.bbox.w;
		os << std::endl;
		os << " nboxes:" << det.nboxes;
		os << " objectness:" << det.objectness;
		os << " classes:" << det.classes << std::endl;
		os << " sort class:" << det.sort_class;
		os << " number of probs:" << det.prob.size();
		return os;
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
	int iterations; //, stream_mr;
	unsigned char tensor_core_mode;
	int total_errors;
	bool normalized_coordinates;
	bool compare_layers;
	//gold attribute
	std::unordered_map<std::string, std::vector<Detection> > gold_hash_var;

//	Log* app_log;
#ifdef LOGS
#ifdef BUILDPROFILER
	std::shared_ptr<rad::Profiler> profiler_thread;
#endif
#endif

	DetectionGold(int argc, char **argv, real_t thresh, real_t hier_thresh,
			char *img_list_path, char *config_file, char *config_data,
			char *model, char *weights);

	virtual ~DetectionGold();

	int run(detection* dets, int nboxes, int img_index, int classes, int img_w, int img_h);

	void load_gold_hash(std::ifstream& gold_file);

	void write_gold_header() const;

	void gen(detection* dets, int nboxes, int img_index,
			std::ofstream& gold_file, int classes);
	int cmp(detection* dets, int nboxes, int img_index, int classes, int img_w,
			int img_h);

	int compare_detection(const Detection& g_det, const detection& f_det, const std::string& img,
			int nb, int classes, int img_w, int img_h) const;

private:
	static std::string generate_gold_line(int bb, detection det, const box& b,
			detection* dets);

	static void load_gold_line(std::ifstream& gold_file, Detection& det,
			int nboxes);

};

#endif /* DETECTIONGOLD_H_ */

