/*
 * DetectionGold.cu
 *
 *  Created on: 28/09/2018
 *      Author: fernando
 */

#include "DetectionGold.h"
#include <iterator>
#include "helpful.h"
#include <sstream>
#include <ctime>

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
		std::cout << "ERROR ON OPENING GOLD OUTPUT FILE\n";
		exit(-1);
	}
}

DetectionGold::DetectionGold(int argc, char **argv, real_t thresh,
		real_t hier_thresh, char *img_list_path, char *config_file,
		char *config_data, char *model, char *weights) {
	char *def;
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

	std::cout << this->generate << " " << this->iterations << " "
			<< this->gold_inout << "\n";

	if (!this->generate) {

		//		Log(std::string gold, int save_layer, int abft, int iterations,
		//				std::string app, unsigned char use_tensor_core_mode)
		this->app_logging = new Log(this->gold_inout, 0, 0, this->iterations,
				this->network_name, this->tensor_core_mode);

		//	detection gold;
		std::string line;
		std::ifstream gold_file(this->gold_inout, std::ifstream::in);
		if (gold_file.is_open()) {
			getline(gold_file, line);
		} else {
			std::cout << "ERROR ON OPENING GOLD FILE\n";
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
		std::cout << "Size of vector outside "
				<< this->gold_hash_var[this->gold_img_names[0]].size() << "\n";
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
	}

	//check if iterations is bigger than img_list_size
	if (this->iterations < this->plist_size) {
		this->iterations = this->plist_size;
	}

}

bool operator!=(const box& a, const box& b) {
	return (a.h != b.h || a.w != b.w || a.x != b.x || a.y != a.y);
}

void DetectionGold::cmp(detection* found_dets, int nboxes, int img_index,
		int classes) {
	std::string img = this->gold_img_names[img_index];

	std::vector<Detection> gold_dets = this->gold_hash_var[img];

	int min_nboxes = gold_dets.size();
	int error_count = 0;

	std::cout << "size of gold_dets " << gold_dets.size() << "\n";

	if (min_nboxes != nboxes) {
		std::string error_detail = "img: " + img + " nboxes_e: "
				+ std::to_string(min_nboxes) + " nboxes_r: "
				+ std::to_string(nboxes);
		this->app_logging->log_error_info(error_detail);
		min_nboxes = std::min(nboxes, min_nboxes);
		error_count++;
	}

	for (int nb = 0; nb < min_nboxes; nb++) {
		Detection g_det = gold_dets[nb];
		detection f_det = found_dets[nb];

		box g_box = g_det.bbox;
		box f_box = f_det.bbox;

		real_t g_objectness = g_det.objectness;
		real_t f_objectness = f_det.objectness;

		int g_sort_class = g_det.sort_class;
		int f_sort_class = f_det.sort_class;

		if ((g_box != f_box) || (g_objectness != f_objectness)
				|| (g_sort_class != f_sort_class)) {
			std::ostringstream error_info("");
			error_info.precision(6);

			error_info << "img: " << img << " detection: " << nb << " x_e: "
					<< g_box.x << " x_r: " << f_box.x << " y_e: " << g_box.y
					<< " y_r: " << f_box.y << " h_e: " << g_box.h << " h_r: "
					<< f_box.h << " w_e: " << g_box.w << " w_r: " << f_box.w
					<< " objectness_e: " << g_objectness << " objectness_r: "
					<< f_objectness << " sort_class_e: " << g_sort_class
					<< " sort_class_r: " << f_sort_class;

			this->app_logging->log_error_info(error_info.str());
			error_count++;
		}

		for (int cl = 0; cl < classes; ++cl) {
			real_t g_prob = g_det.prob[cl];
			real_t f_prob = f_det.prob[cl];

			if (g_prob != f_prob) {
				std::ostringstream error_info("");
				error_info.precision(6);

				error_info << "img: " << img << " detection: " << nb
						<< " class: " << cl << " prob_e: " << g_prob
						<< " prob_r: " << f_prob;
				this->app_logging->log_error_info(error_info.str());
				error_count++;
			}
		}
	}

	this->app_logging->update_error_count(error_count);
}

void DetectionGold::run(detection *dets, int nboxes, int img_index,
		int classes) {
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
		std::time_t start = std::time(nullptr);

		this->cmp(dets, nboxes, img_index, classes);

		std::cout << "Seconds to compare: "
				<< std::difftime(std::time(nullptr), start) << " s.\n";

	}
}

void DetectionGold::gen(detection *dets, int nboxes, int img_index,
		std::ofstream& gold_file, int classes) {
	//first write the image string name
	std::string img = this->gold_img_names[img_index];

	gold_file << img << ";" << nboxes << ";" << classes << ";" << std::endl;

	for (int i = 0; i < nboxes; ++i) {

		box b = dets[i].bbox;

		gold_file << dets[i].objectness << ";" << dets[i].sort_class << ";"
				<< b.x << ";" << b.y << ";" << b.w << ";" << b.h << ";"
				<< std::endl;

		for (int cl = 0; cl < classes; ++cl) {
			real_t prob = dets[i].prob[cl];
			if (prob != 0)
				gold_file << prob << ";" << cl << ";" << std::endl;
		}
		//just to end the box info
		gold_file << "--;" << std::endl;
	}

}

void DetectionGold::load_gold_hash(std::ifstream& gold_file) {
//allocate detector
	this->gold_img_names = std::vector < std::string > (this->plist_size);
	std::string line;

	for (int i = 0; i < this->plist_size && getline(gold_file, line); i++) {
		//	gold_file << img << ";" << nboxes << ";" << classes << ";" << std::endl;
		std::vector < std::string > splited_line = split(line, ';');
		// Set each img_name path
		this->gold_img_names[i] = splited_line[0];

		// Probarray creation
		int nboxes = std::stoi(splited_line[1]);
		int classes = std::stoi(splited_line[2]);
		//
		std::vector<Detection> detections(nboxes);

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

			// Getting the probabilities
			std::vector < real_t > probs(classes, 0.0);

			while (getline(gold_file, line)) {
				splited_line = split(line, ';');

				if (splited_line[0] == "--")
					break;
				real_t prob = std::stof(splited_line[0]);
				int class_ = std::stoi(splited_line[1]);

				probs[class_] = prob;

			}

			detections[bb] = Detection(nboxes, sort_class, objectness, probs,
					b);

		}

		this->gold_hash_var.put(this->gold_img_names[i], detections);
	}

}

DetectionGold::~DetectionGold() {
	if (this->app_logging) {
		delete this->app_logging;
	}
}

void DetectionGold::start_iteration() {
	if (!this->generate)
		this->app_logging->start_iteration_app();
}

void DetectionGold::end_iteration() {
	if (!this->generate)
		this->app_logging->end_iteration_app();
}
