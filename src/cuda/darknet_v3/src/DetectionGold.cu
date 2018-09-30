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
		//	thresh; hier_tresh; img_list_size; img_list_path; config_file; config_data; model;weights;
		this->thresh = atof(split_ret[0].c_str());
		this->hier_thresh = atof(split_ret[1].c_str());
		this->plist_size = atoi(split_ret[2].c_str());
		this->img_list_path = split_ret[3].size();
		this->config_file = split_ret[4].size();
		this->cfg_data = split_ret[5].size();
		this->model = split_ret[6].size();//const_cast<char*>(split_ret[6].c_str());
		this->weights = split_ret[7].size();

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
	gold_header += this->weights + ";\n";

	std::ofstream gold(this->gold_inout);
	if (gold.is_open()) {
		gold << gold_header;
		gold.close();
	} else {
		std::cout << "ERROR ON OPENING GOLD OUTPUT FILE\n";
		exit(-1);
	}
}

void DetectionGold::generate_method(int img_index, int nboxes, network& net,
		detection* dets) {
	layer l = net.layers[net.n - 1];
	std::ofstream gold_file(this->gold_inout);
	if (gold_file.is_open()) {
		//first write the image string name
		gold_file << this->gold_img_names[img_index] << ";\n";
		this->save_gold_img_i(dets, nboxes, l.classes, gold_file, l.coords);
		gold_file.close();
	} else {
		std::cout << "ERROR ON OPENING GOLD OUTPUT FILE\n";
		exit(-1);
	}
}

bool operator!=(const box& a, const box& b) {
	return (a.h != b.h || a.w != b.w || a.x != b.x || a.y != a.y);
}

void DetectionGold::compare_method(int nboxes, detection* dets, std::string img) {
	std::ostringstream error_info("");

	//----------------------------------------------------------------
	//Comparing nboxes
	int g_nboxes = this->gold_hash_var[img].size();
	int min_nboxes = g_nboxes;
	if (g_nboxes != nboxes) {
		error_info = std::ostringstream("");
		error_info << "img: " << img << " nboxes_e: " << g_nboxes
				<< " nboxes_r: " << nboxes;
		this->app_logging->log_error_info(
				const_cast<char*>(error_info.str().c_str()));
		min_nboxes = std::min(g_nboxes, nboxes);
	}

	gold_tuple_array gold_tuple_var = this->gold_hash_var[img];

	//detection is always nboxes size
	for (int i = 0; i < min_nboxes; i++) {

	}
}

void DetectionGold::compare_or_generate(detection *dets, int nboxes, int img_index) {

	// To generate function
	if (this->generate) {
		generate_method(img_index, nboxes, net, dets);
		// To compare function
	} else {
		int l_coord = net.layers[net.n - 1].coords;
		//detection is allways nboxes size
		compare_method(nboxes, dets, this->gold_img_names[img_index], l_coord);
	}
}

/**
 * it was adapted from draw_detections in image.c line 174
 * it saves a gold file for radiation tests
 */
void DetectionGold::save_gold_img_i(std::string img, detection *dets, int num,
		real_t thresh, int classes, std::ofstream& gold_file) {
	gold_file << img << ";" << thresh << ";" << classes << ";" << num << ";\n";
	for (i = 0; i < num; ++i) {
		for (j = 0; j < classes; ++j) {
			real_t prob = dets[i].prob[j];
			box b = dets[i].bbox;
			real_t mask = dets[i].mask;

			gold_file << prob << ";" << b.x << ";" << b.y << ";" << b.w, ";"
					<< b.h
			";" << class_ << ";\n";

		}
	}
}

void DetectionGold::load_gold_hash(std::ifstream& gold_file) {
//allocate detector
	this->gold_img_names = std::vector < std::string > (this->plist_size);
//	gold_file << std::to_string(classes) << ";" << std::to_string(nboxes)
//			<< std::to_string(l_coords) << ";\n";
//	for (int i = 0; i < nboxes; i++) {
//		detection det = dets[i];
//
//		gold_file << this->print_box(det.bbox) << std::to_string(det.objectness)
//				<< std::to_string(det.sort_class) << ";\n";
//
//		if (l_coords > 4) {
//			for (int j = 0; j < l_coords; j++)
//				gold_file << std::to_string(det.mask[j]);
//		}
//
//		for (int j = 0; j < classes; j++) {
//			//	real_t *prob;
//			gold_file << std::to_string(det.prob[j]) << ";\n";
//		}
//	}
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

