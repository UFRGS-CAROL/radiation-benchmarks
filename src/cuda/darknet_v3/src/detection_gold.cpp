/*
 * DetectionGold.cu
 *
 *  Created on: 28/09/2018
 *      Author: fernando
 */

#include "detection_gold.h"

#include <iterator>
#include "helpful.h"
#include <sstream>
#include <ctime>
#include <iostream>
#include <cmath>

/**
 * Detection Gold class
 */

void DetectionGold::write_gold_header() {
	//	0       1           2              3              4            5            6        7
	//	thresh; hier_tresh; img_list_size; img_list_path; config_file; config_data; model;weights;
	std::string gold_header = std::to_string(this->thresh) + ";";
	gold_header += std::to_string(this->hier_thresh) + ";";
	gold_header += std::to_string(this->plist_size) + ";";
	gold_header += this->img_list_path + ";";
	gold_header += this->config_file + ";";
	gold_header += this->cfg_data + ";";
	gold_header += this->model + ";";
	gold_header += this->weights + ";";

	std::ofstream gold(this->gold_inout, std::ofstream::trunc);
	if (gold.is_open()) {
		gold << gold_header << std::endl;
		gold.close();
	} else {
		std::cerr << "ERROR ON OPENING GOLD OUTPUT FILE\n";
		exit(-1);
	}
}

DetectionGold::DetectionGold(int argc, char **argv, real_t thresh,
		real_t hier_thresh, char *img_list_path, char *config_file,
		char *config_data, char *model, char *weights) {
	char *def = nullptr;
	this->gold_inout = std::string(find_char_arg(argc, argv, "-gold", def));
	this->generate = find_int_arg(argc, argv, "-generate", 0);
	this->network_name = "darknet_v3_";
#if REAL_TYPE == HALF
	this->network_name += "half";
#elif REAL_TYPE == FLOAT
	this->network_name += "single";
#elif REAL_TYPE == DOUBLE
	this->network_name += "double";
#endif

	this->iterations = find_int_arg(argc, argv, "-iterations", 1);
	this->tensor_core_mode = find_int_arg(argc, argv, "-tensor_cores", 0);
	this->stream_mr = find_int_arg(argc, argv, "-smx_redundancy", 0);
	this->thresh = thresh;
	this->hier_thresh = hier_thresh;
	this->total_errors = 0;

	if (!this->generate) {

		//		Log(std::string gold, int save_layer, int abft, int iterations,
		//				std::string app, unsigned char use_tensor_core_mode)
		this->app_log->start_log(this->gold_inout, 0, 0, this->iterations,
				this->network_name, this->tensor_core_mode);

		//	detection gold;
		std::string line;
		std::ifstream gold_file(this->gold_inout, std::ifstream::in);
		if (gold_file.is_open()) {
			getline(gold_file, line);
		} else {
			std::cerr << "ERROR ON OPENING GOLD FILE\n";
			exit(-1);
		}

		std::vector < std::string > split_ret = split(line, ';');
		//	0       1           2              3              4            5            6      7
		//	thresh; hier_tresh; img_list_size; img_list_path; config_file; config_data; model;weights;
		this->thresh = std::stof(split_ret[0]);
		this->hier_thresh = std::stof(split_ret[1]);
		this->plist_size = std::stoi(split_ret[2]);
		this->img_list_path = split_ret[3];
		this->config_file = split_ret[4];
		this->cfg_data = split_ret[5];
		this->model = split_ret[6];
		this->weights = split_ret[7];

		//allocate detector
		this->load_gold_hash(gold_file);

		gold_file.close();

	} else {
		this->img_list_path = std::string(img_list_path);

		//reading the img list path content
		std::ifstream tmp_img_file(this->img_list_path);
		std::copy(std::istream_iterator < std::string > (tmp_img_file),
				std::istream_iterator<std::string>(),
				std::back_inserter(this->gold_img_names));

		this->plist_size = this->gold_img_names.size();
		this->config_file = std::string(config_file);
		this->cfg_data = std::string(config_data);
		this->model = std::string(model);
		this->weights = std::string(weights);

		this->write_gold_header();
		this->iterations = 1;

	}
}

bool operator!=(const box& a, const box& b) {
	real_t x_diff = std::abs((a.x - b.x));
	real_t y_diff = std::abs((a.y - b.y));
	real_t w_diff = std::abs((a.w - b.w));
	real_t h_diff = std::abs((a.h - b.h));

	return (x_diff > THRESHOLD_ERROR || y_diff > THRESHOLD_ERROR
			|| w_diff > THRESHOLD_ERROR || h_diff > THRESHOLD_ERROR);
}

int DetectionGold::cmp(detection* found_dets, int nboxes, int img_index,
		int classes) {
	std::string img = this->gold_img_names[img_index];
	std::vector<Detection> gold_dets = this->gold_hash_var[img];

	int error_count = 0;


	for (int nb = 0; nb < nboxes; nb++) {
		Detection g_det = gold_dets[nb];
		detection f_det = found_dets[nb];

		box g_box = g_det.bbox;
		box f_box = f_det.bbox;

		real_t g_objectness = g_det.objectness;
		real_t f_objectness = f_det.objectness;

		int g_sort_class = g_det.sort_class;
		int f_sort_class = f_det.sort_class;

		real_t objs_diff = std::abs(g_objectness - f_objectness);
		int sortc_diff = std::abs(g_sort_class - f_sort_class);

		if ((objs_diff > THRESHOLD_ERROR) || (sortc_diff > THRESHOLD_ERROR)
				|| (g_box != f_box)) {
			std::ostringstream error_info("");
			error_info.precision(STORE_PRECISION);

			error_info << "img: " << img << " detection: " << nb << " x_e: "
					<< g_box.x << " x_r: " << f_box.x << " y_e: " << g_box.y
					<< " y_r: " << f_box.y << " h_e: " << g_box.h << " h_r: "
					<< f_box.h << " w_e: " << g_box.w << " w_r: " << f_box.w
					<< " objectness_e: " << g_objectness << " objectness_r: "
					<< f_objectness << " sort_class_e: " << g_sort_class
					<< " sort_class_r: " << f_sort_class;

			this->app_log->log_error_info(error_info.str());
			error_count++;
		}

		for (int cl = 0; cl < classes; ++cl) {
			real_t g_prob = g_det.prob[cl];
			real_t f_prob = f_det.prob[cl];
			real_t prob_diff = std::abs(g_prob - f_prob);

			if ((g_prob >= this->thresh || f_prob >= this->thresh)
					&& prob_diff > THRESHOLD_ERROR) {
				std::ostringstream error_info("");
				error_info.precision(STORE_PRECISION);

				error_info << "img: " << img << " detection: " << nb
						<< " class: " << cl << " prob_e: " << g_prob
						<< " prob_r: " << f_prob;
				this->app_log->log_error_info(error_info.str());
				error_count++;
			}
		}
	}
	this->total_errors+= error_count;
	if(this->total_errors > MAX_ERROR_COUNT){
		this->app_log->update_error_count(this->total_errors);
		exit(-1);
	}
	this->app_log->update_error_count(error_count);
	return error_count;
}

int DetectionGold::run(detection *dets, int nboxes, int img_index,
		int classes) {
	int ret = 0;
	// To generate function
	//std::string img, detection* dets, int nboxes, int classes, int l_coord
	if (this->generate) {

		std::ofstream gold_file(this->gold_inout, std::ofstream::app);
		if (!gold_file.is_open()) {
			std::cerr << "ERROR ON OPENING GOLD FILE\n";
			exit(-1);
		}
		this->gen(dets, nboxes, img_index, gold_file, classes);
		gold_file.close();
	} else {
		// To compare function
		//detection is allways nboxes size
		ret = this->cmp(dets, nboxes, img_index, classes);
	}
	return ret;
}

void DetectionGold::gen(detection *dets, int nboxes, int img_index,
		std::ofstream& gold_file, int classes) {
	//first write the image string name
	std::string img = this->gold_img_names[img_index];

	gold_file << img << ";" << nboxes << ";" << std::endl;

	for (int bb = 0; bb < nboxes; bb++) {
		detection det = dets[bb];
		box b = det.bbox;

		std::ostringstream box_str("");
		box_str.precision(STORE_PRECISION);

		box_str << dets[bb].objectness << ";" << dets[bb].sort_class << ";"
				<< b.x << ";" << b.y << ";" << b.w << ";" << b.h << ";"
				<< det.classes << ";" << std::endl;

		for (int cl = 0; cl < det.classes; cl++) {
			real_t prob = dets[bb].prob[cl];
			box_str << prob << ";";
		}

		gold_file << box_str.str() << std::endl;
	}

}

void DetectionGold::load_gold_hash(std::ifstream& gold_file) {
//allocate detector
	this->gold_img_names = std::vector < std::string > (this->plist_size);
	std::string line;

	for (int i = 0; i < this->plist_size; i++) {

		//	gold_file << img << ";" << nboxes << ";" << std::endl;
		getline(gold_file, line);
		std::vector < std::string > splited_line = split(line, ';');

		// Set each img_name path
		this->gold_img_names[i] = splited_line[0];

		// Probarray creation

		int nboxes = std::stoi(splited_line[1]);

		std::vector<Detection> detections(nboxes, Detection());

		for (int bb = 0; bb < nboxes; ++bb) {

			// Getting bb box
			box b;
			getline(gold_file, line);

			splited_line = split(line, ';');
			real_t objectness = std::stof(splited_line[0]);
			int sort_class = std::stoi(splited_line[1]);
			b.x = std::stof(splited_line[2]);
			b.y = std::stof(splited_line[3]);
			b.w = std::stof(splited_line[4]);
			b.h = std::stof(splited_line[5]);
			int classes = std::stoi(splited_line[6]);

			// Getting the probabilities
			std::vector < real_t > probs(classes, 0.0);

			if (getline(gold_file, line)) {
				splited_line = split(line, ';');

				for (auto cl = 0; cl < classes; cl++) {
					probs[cl] = std::stof(splited_line[cl]);
				}
			}
			detections[bb] = Detection(classes, nboxes, sort_class, objectness,
					probs, b);
		}

		this->gold_hash_var.put(this->gold_img_names[i], detections);
	}


}

DetectionGold::~DetectionGold() {
	if (!this->generate) {
#ifdef LOGS
		end_log_file();
#endif
		delete this->app_log;
	}

}
//
//void DetectionGold::start_iteration() {
//	if (!this->generate) {
//#ifdef LOGS
//		start_iteration();
//#endif
//	}
//}
//
//void DetectionGold::end_iteration() {
//	if (!this->generate) {
//#ifdef LOGS
//		end_iteration();
//#endif
//	}
//
//	this->current_iteration++;
//}

//void DetectionGold::start_log(std::string gold, int save_layer, int abft,
//		int iterations, std::string app, unsigned char use_tensor_core_mode) {
//#ifdef LOGS
//	std::string test_info = std::string("gold_file: ") + gold;
//
//	test_info += " save_layer: " + std::to_string(save_layer) + " abft_type: ";
//
//	test_info += std::string(ABFT_TYPES[abft]) + " iterations: "
//	+ std::to_string(iterations);
//
//	test_info += " tensor_core_mode: "
//	+ std::to_string(int(use_tensor_core_mode));
//
//	set_iter_interval_print(10);
//
//	start_log_file(const_cast<char*>(app.c_str()),
//			const_cast<char*>(test_info.c_str()));
//#endif
//}

//void DetectionGold::update_timestamp_app() {
//#ifdef LOGS
//	update_timestamp();
//#endif
//}

//void DetectionGold::log_error_info(std::string error_detail) {
//#ifdef LOGS
//	log_error_detail(const_cast<char*>(error_detail.c_str()));
//#endif
//}

//void DetectionGold::update_error_count(long error_count) {
//#ifdef LOGS
//	if(error_count)
//	log_error_count(error_count);
//#endif
//}
