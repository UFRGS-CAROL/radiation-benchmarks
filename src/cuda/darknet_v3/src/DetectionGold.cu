/*
 * DetectionGold.cu
 *
 *  Created on: 28/09/2018
 *      Author: fernando
 */

#include "DetectionGold.h"
#include <iterator>
#include "helpful.h"

#include <tuple>
#include <unordered_map>


//	box bbox;
//	int classes;
//	real_t *prob;
//	real_t *mask;
//	real_t objectness;
//	int sort_class;
// int nboxes

typedef std::tuple<box, int, std::vector<real_t>, std::vector<real_t>, real_t,
		int, int> gold_tuple;

typedef std::unordered_map<std::string, gold_tuple> gold_hash;

//thresh; hier_tresh; img_list_size; img_list_path; config_file; config_data; model;weights;total;classes;
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
		std::ifstream img_list_file(this->gold_inout);
		if (img_list_file.is_open()) {
			getline(img_list_file, line);
		} else {
			std::cout << "ERROR ON OPENING GOLD FILE\n";
			exit(-1);
		}

		std::vector < std::string > split_ret = split(line, ';');
		//	0       1           2              3              4            5            6
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
		this->gold_img_names = std::vector < std::string > (this->plist_size);

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

void DetectionGold::compare_or_generate(detection *dets, int nboxes,
		int img_index, network &net) {

	// To generate function
	if (this->generate) {
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

		// To compare function
	} else {

	}
}

/**
 * it was adapted from draw_detections in image.c line 174
 * it saves a gold file for radiation tests
 */
void DetectionGold::save_gold_img_i(detection *dets, int nboxes, int classes,
		std::ofstream& gold_file, int l_coords) {

//	detection *dets = (detection*) calloc(nboxes, sizeof(detection));
//		for (i = 0; i < nboxes; ++i) {
//			dets[i].prob = (real_t*) calloc(l.classes, sizeof(real_t));
//			if (l.coords > 4) {
//				dets[i].mask = (real_t*) calloc(l.coords - 4, sizeof(real_t));
//			}
//		}
	//Store first classes, nboxes, l_coords
	gold_file << std::to_string(classes) << ";" << std::to_string(nboxes) << std::to_string(l_coords)
			<< ";\n";
	for (int i = 0; i < nboxes; i++) {
		detection det = dets[i];
		//	box bbox;
		//	real_t objectness;
		//	int sort_class;
		gold_file << this->print_box(det.bbox) << std::to_string(det.objectness)
				<< std::to_string(det.sort_class) << ";\n";

		if (l_coords > 4) {
			for (int j = 0; j < l_coords; j++)
				gold_file << std::to_string(det.mask[j]);
		}

		for (int j = 0; j < classes; j++) {
			//	real_t *prob;
			gold_file << std::to_string(det.prob[j]) << ";\n";
		}
	}
}

DetectionGold::~DetectionGold() {
	if (this->app_logging) {
		delete this->app_logging;
	}
}

std::string DetectionGold::error_check(real_t f_pb, real_t g_pb, box f_b,
		box g_b, std::string img, int class_g, int class_f, int pb_i) {

	std::vector<real_t> diff_array = {std::fabs(f_b.x - g_b.x),  //x axis
		std::fabs(f_b.y - g_b.y),//y axis
		std::fabs(f_pb - g_pb),//probabilities
		std::fabs(f_b.h - g_b.h),// height
		std::fabs(f_b.w - g_b.w), (real_t) std::abs(class_g - class_f)}; // width and class

	//	if (class_g != class_f)
	//		std::cout << " val " << class_g << " " << class_f << "\n";
	bool diff = false;
	for (auto diff_element : diff_array) {
		if (diff_element > THRESHOLD_ERROR) {
			diff = true;
			break;
		}
	}

	if (diff) {
		char error_detail[1000];

		sprintf(error_detail, "img: [%s]"
				" prob_r[%d][%d]: %1.16e"
				" prob_e[%d][%d]: %1.16e"
				" x_r: %1.16e x_e: %1.16e"
				" y_r: %1.16e y_e: %1.16e"
				" w_r: %1.16e w_e: %1.16e"
				" h_r: %1.16e h_e: %1.16e", img, pb_i, class_f, f_pb, pb_i,
				class_g, g_pb, f_b.x, g_b.x, f_b.y, g_b.y, f_b.w, g_b.w,
				f_b.h, g_b.h);
		return std::string(error_detail);
	}

	return "";
}

std::string DetectionGold::print_box(box b) {
	return std::to_string(b.x) + " " + std::to_string(b.y) + " "
			+ std::to_string(b.w) + " " + std::to_string(b.h) + "\n";
}

