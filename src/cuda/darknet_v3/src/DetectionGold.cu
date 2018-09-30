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

DetectionGold::DetectionGold(int argc, char **argv, real_t thresh,
		real_t hier_thresh, char *img_list_path, char *config_file,
		char *config_data, char *model, char *weights) {
	char *def;
	this->gold_inout = std::string(find_char_arg(argc, argv, "-gold", def));
	this->generate = find_int_arg(argc, argv, "-generate", 0);
	this->network_name = "darknet_v3";
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
		std::ifstream gold_file(this->gold_inout);
		if (gold_file.is_open()) {
			getline(gold_file, line);
		} else {
			std::cout << "ERROR ON OPENING GOLD FILE\n";
			exit(-1);
		}

		std::vector < std::string > split_ret = split(line, ';');
		//	0       1           2              3              4            5            6      7
		//	thresh; hier_tresh; img_list_size; img_list_path; config_file; config_data; model;weights;coord;classes
		this->thresh = std::stof(split_ret[0]);
		this->hier_thresh = std::stof(split_ret[1]);
		this->plist_size = std::stoi(split_ret[2]);
		this->img_list_path = split_ret[3];
		this->config_file = split_ret[4];
		this->cfg_data = split_ret[5];
		this->model = split_ret[6];
		this->weights = split_ret[7];
		this->coord = std::stoi(split_ret[8]);
		this->classes = std::stoi(split_ret[9]);

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
	}

	//check if iterations is bigger than img_list_size
	if (this->iterations < this->plist_size) {
		this->iterations = this->plist_size;
	}

}

void DetectionGold::write_gold_header() {
	//	0       1           2              3              4            5            6        7      8     9
	//	thresh; hier_tresh; img_list_size; img_list_path; config_file; config_data; model;weights;classes;
	std::string gold_header = std::to_string(this->thresh) + ";";
	gold_header += std::to_string(this->hier_thresh) + ";";
	gold_header += std::to_string(this->plist_size) + ";";
	gold_header += this->img_list_path + ";";
	gold_header += this->config_file + ";";
	gold_header += this->cfg_data + ";";
	gold_header += this->model + ";";
	gold_header += this->weights + ";";
	gold_header += this->coord + ";";
	gold_header += this->classes + ";\n";

	std::ofstream gold(this->gold_inout);
	if (gold.is_open()) {
		gold << gold_header;
		gold.close();
	} else {
		std::cout << "ERROR ON OPENING GOLD OUTPUT FILE\n";
		exit(-1);
	}
}

bool operator!=(const box& a, const box& b) {
	return (a.h != b.h || a.w != b.w || a.x != b.x || a.y != a.y);
}

void DetectionGold::cmp(detection* dets, int nboxes, int img_index,
		int l_coord) {
	std::ostringstream error_info("");
	std::string img = this->gold_img_names[img_index];

}

void DetectionGold::run(detection *dets, int nboxes, int img_index,
		int l_coord) {
	// To generate function
	//std::string img, detection* dets, int nboxes, int classes, int l_coord
	if (this->generate) {

		std::ofstream gold_file(this->gold_inout, std::ofstream::app);
		if (!gold_file.is_open()) {
			std::cerr << "ERROR ON OPENING GOLD FILE\n";
			exit(-1);
		}
		this->gen(dets, nboxes, img_index, l_coord, gold_file);
		gold_file.close();
	} else {
		// To compare function
		//detection is allways nboxes size
		this->cmp(dets, nboxes, img_index, l_coord);
	}
}

void DetectionGold::gen(detection *dets, int nboxes, int img_index, int l_coord,
		std::ofstream& gold_file) {
	//first write the image string name
	std::string img = this->gold_img_names[img_index];

	gold_file << img << ";" << nboxes << ";\n";
	for (int i = 0; i < nboxes; ++i) {

		for (int j = 0; j < l_coord; j++) {
			gold_file << dets[i].mask[j] << ";";
		}
		gold_file << "\n";

		box b = dets[i].bbox;

		gold_file << dets[i].objectness << ";" << dets[i].sort_class << ";"
				<< b.x << ";" << b.y << ";" << b.w << ";" << b.h << ";\n";

		for (int j = 0; j < this->classes; ++j) {
			gold_file << dets[i].prob[j] << ";";
		}
		gold_file << "\n";
	}

}

void DetectionGold::load_gold_hash(std::ifstream& gold_file) {
//allocate detector
	this->gold_img_names = std::vector < std::string > (this->plist_size);
	std::string line;

	for (int i = 0; i < this->plist_size && getline(gold_file, line); i++) {
		//	gold_file << img << ";" << nboxes << ";\n";
		std::vector < std::string > splited_line = split(line, ';');
		// Set each img_name path
		this->gold_img_names[i] = splited_line[0];
		// Probarray creation
		int nboxes = std::stoi(splited_line[1]);

		//
		std::vector<Detection> detections(nboxes);

		for (int bb = 0; bb < nboxes && getline(gold_file, line); ++bb) {
			Detection i_element;
			i_element.nboxes = nboxes;

			// Getting mask
			getline(gold_file, line);
			splited_line = split(line, ';');
			for (auto mask : splited_line)
				i_element.mask.push_back(real_t(std::stof(mask)));

			// Getting bb box
			box b;
			i_element.objectness = std::stof(splited_line[0]);
			i_element.sort_class = std::stoi(splited_line[1]);
			b.x = std::stof(splited_line[2]);
			b.y = std::stof(splited_line[3]);
			b.w = std::stof(splited_line[4]);
			b.h = std::stof(splited_line[5]);

			i_element.boxes.push_back(b);

			// Getting the probabilities
			getline(gold_file, line);
			splited_line = split(line, ';');
			for (auto prob : splited_line)
				i_element.prob.push_back(std::stof(prob));

			detections[bb] = i_element;
		}

		this->gold_hash_var[this->gold_img_names[i]] = detections;

	}

}

DetectionGold::~DetectionGold() {
	if (this->app_logging) {
		delete this->app_logging;
	}
}

std::string DetectionGold::print_box(box b) {
	return std::to_string(b.x) + " " + std::to_string(b.y) + " "
			+ std::to_string(b.w) + " " + std::to_string(b.h) + "\n";
}

