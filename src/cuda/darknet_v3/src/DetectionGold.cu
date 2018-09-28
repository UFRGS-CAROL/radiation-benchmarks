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

void DetectionGold::compare_method(int nboxes, detection* dets, std::string img,
		int f_l_coord) {
	//----------------------------------------------------------------
	//Comparing nboxes
	int g_nboxes = this->gold_hash_var[img].size();
	int min_nboxes = g_nboxes;
	if (g_nboxes != nboxes) {
		error_info = "";
		error_info << "img: " << img << " detection_bbox: " << i
				<< " nboxes_e: " << g_nboxes << " nboxes_r: " << nboxes;
		this->app_logging->log_error_info(error_info.c_str());
		min_nboxes = std::min(g_nboxes, nboxes);
	}

	gold_tuple_array gold_tuple_var = this->gold_hash_var[img];

	//detection is always nboxes size
	for (int i = 0; i < min_nboxes; i++) {
		//found detections
		detection det = dets[i];
		box f_bbox = det.bbox;
		int f_classes = det.classes;
		real_t* f_prob = det.prob;
		real_t* f_mask = det.mask;
		real_t f_objectness = det.objectness;
		int f_sort_class = det.objectness;

		//gold detections
		box g_bbox = std::get < 0 > (gold_tuple_var[i]);
		int g_classes = std::get < 1 > (gold_tuple_var[i]);
		std::vector < real_t > g_prob = std::get < 2 > (gold_tuple_var[i]);
		std::vector < real_t > g_mask = std::get < 3 > (gold_tuple_var[i]);
		real_t g_objectness = std::get < 4 > (gold_tuple_var[i]);
		int g_sort_class = std::get < 5 > (gold_tuple_var[i]);
		int g_l_coord = std::get < 6 > (gold_tuple_var[i]);

		std::ostringstream error_info = "";
		//----------------------------------------------------------------
		//Comparing objectness
		if (g_objectness != f_objectness) {
			error_info = "";
			error_info << "img: " << img << " detection_bbox: " << i
					<< " objectness_e: " << g_objectness << " objectness_r: "
					<< f_objectness;
			this->app_logging->log_error_info(error_info.c_str());

		}
		//----------------------------------------------------------------
		//Comparing sort_class
		if (g_sort_class != f_sort_class) {
			error_info = "";
			error_info << "img: " << img << " detection_bbox: " << i
					<< " sort_class_e: " << g_sort_class << " sort_class_r: "
					<< f_sort_class;
			this->app_logging->log_error_info(error_info.c_str());

		}

		//----------------------------------------------------------------
		//Comparing bbox
		if (g_bbox != f_bbox) {
			error_info = "";
			error_info << "img: " << img << " detection_bbox: " << i << " x_e: "
					<< g_bbox.x << " x_r: " << f_bbox.x << "y_e: " << g_bbox.y
					<< " y_r: " << f_bbox.y << "h_e: " << g_bbox.h << " h_r: "
					<< f_bbox.h << "w_e: " << g_bbox.w << " w_r: " << f_bbox.w;
			this->app_logging->log_error_info(error_info.c_str());
		}

		//----------------------------------------------------------------
		//Comparing prob array
		// prob is always classes size
		int min_classes = g_classes;
		if (f_classes != g_classes) {
			error_info = "";
			error_info << "img: " << img
					<< " different number of classes for detection " << i
					<< " diff " << std::abs(f_classes - g_classes);
			this->app_logging->log_error_info(error_info.c_str());
			min_classes = std::min(g_classes, f_classes);
		}

		for (int cl = 0; cl < min_classes; cl++) {
			real_t g = g_prob[cl];
			real_t f = f_prob[cl];
			if (g != f) {
				error_info = "";
				error_info << "img: " << img << " detection_bbox: " << i
						<< " detection_class: " << cl << " prob_e: " << g
						<< " prob_r: " << f;
				this->app_logging->log_error_info(error_info.c_str());
			}
		}

		//----------------------------------------------------------------
		//Comparing mask
		int min_l_coord = g_l_coord;
		if (g_l_coord != f_l_coord) {
			error_info = "";
			error_info << "img: " << img
					<< " different number of l_coord for detection " << i
					<< " diff " << std::abs(g_l_coord - f_l_coord);
			this->app_logging->log_error_info(error_info.c_str());
			min_l_coord = std::min(g_classes, f_classes);
		}

		for (int lc = 0; lc < min_l_coord; lc++) {
			real_t g = g_mask[cl];
			real_t f = f_mask[cl];
			if (g != f) {
				error_info = "";
				error_info << "img: " << img << " detection_bbox: " << i
						<< " detection_mask: " << cl << " mask_e: " << g
						<< " mask_r: " << f;
				this->app_logging->log_error_info(error_info.c_str());
			}
		}

	}
}

void DetectionGold::compare_or_generate(detection *dets, int nboxes,
		int img_index, network &net) {

	// To generate function
	if (this->generate) {
		generate_method(img_index, nboxes, net, dets);
		// To compare function
	} else {
		int l_coord = net.layers[net.n - 1];
		//detection is allways nboxes size
		compare_method(nboxes, dets, this->gold_img_names[img_index], l_coord);
	}
}

/**
 * it was adapted from draw_detections in image.c line 174
 * it saves a gold file for radiation tests
 */
void DetectionGold::save_gold_img_i(detection *dets, int nboxes, int classes,
		std::ofstream& gold_file, int l_coords) {
	//Store first classes, nboxes, l_coords
	gold_file << std::to_string(classes) << ";" << std::to_string(nboxes)
			<< std::to_string(l_coords) << ";\n";
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

