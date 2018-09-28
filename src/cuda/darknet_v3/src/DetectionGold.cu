/*
 * DetectionGold.cu
 *
 *  Created on: 28/09/2018
 *      Author: fernando
 */

#include "DetectionGold.h"
#include <iterator>
#include "helpful.h"


void ProbArray::read_prob_array(int num, int classes, std::ifstream &ifp) {
	this->boxes = std::vector < box > (num);

	for (int i = 0; i < num; i++) {
		this->probs[i] = std::vector < real_t > (classes);
	}

	std::string line;
	std::vector < std::string > splited;
	for (int i = 0; i < num; ++i) {

		getline(ifp, line);
		splited = split(line, ';');

		box b;
		b.x = atof(splited[1].c_str());
		b.y = atof(splited[2].c_str());
		b.w = atof(splited[3].c_str());
		b.h = atof(splited[4].c_str());
		int class_ = atoi(splited[5].c_str());

		this->probs[i][class_] = atof(splited[0].c_str());

		this->boxes[i] = b;
	}
}

//thresh; hier_tresh; img_list_size; img_list_path; config_file; config_data; model;weights;total;classes;
DetectionGold::DetectionGold(int argc, char **argv, real_t thresh,
		real_t hier_thresh, char *img_list_path, char *config_file,
		char *config_data, char *model, char *weights, int classes) {
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

	if (this->generate != true) {

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
		//	thresh; hier_tresh; img_list_size; img_list_path; config_file; config_data; model;weights;total;classes;
		this->thresh = atof(split_ret[0].c_str());
		this->hier_thresh = atof(split_ret[1].c_str());
		this->plist_size = atoi(split_ret[2].c_str());
		this->img_list_path = split_ret[3].size();
		this->config_file = split_ret[4].size();
		this->cfg_data = split_ret[5].size();
		this->model = split_ret[6].size();//const_cast<char*>(split_ret[6].c_str());
		this->weights = split_ret[7].size();

		this->classes = atoi(split_ret[8].c_str());

		//check if iterations is bigger than img_list_size
		if (this->iterations < this->plist_size){
			this->iterations = this->plist_size;
		}

		//allocate detector
		this->gold_img_names = std::vector < std::string > (this->plist_size);
		this->pb_gold = std::vector < ProbArray > (this->plist_size);

		for (int i = 0; i < this->plist_size && getline(img_list_file, line);
				i++) {
			line.erase(line.size() - 1);

			std::vector < std::string > line_splited = split(line, ';');
			this->gold_img_names[i] = line_splited[0];

			this->pb_gold[i].read_prob_array(this->nboxes, this->classes,
					img_list_file);
		}
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
		this->classes = classes;
		this->write_gold_header();
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

	gold_header += std::to_string(this->classes) + ";\n";
	std::ofstream gold(this->gold_inout);
	if (gold.is_open()) {
		gold << gold_header;
		gold.close();
	} else {
		std::cout << "ERROR ON OPENING GOLD OUTPUT FILE\n";
		exit(-1);
	}
}

/**
 * it was adapted from draw_detections in image.c line 174
 * it saves a gold file for radiation tests
 */
void DetectionGold::save_gold_img_i(char *img, int num, int classes,
		real_t **probs, box *boxes) {
	std::ofstream output(this->gold_inout);

	if (output.is_open()) {

		std::vector < std::string > to_print;
		for (int i = 0; i < num; i++) {
			box b = boxes[i];
			int class_ = get_index(probs[i], classes);
			real_t prob = probs[i][class_];
			std::string str_to_print = std::to_string(prob) + ";"
					+ std::to_string(b.x) + ";" + std::to_string(b.y) + ";"
					+ std::to_string(b.w) + ";" + std::to_string(b.h) + ";"
					+ std::to_string(class_) + ";\n";
			output << str_to_print;
		}
		output.close();
	}

}

DetectionGold::~DetectionGold() {
	if (this->app_logging) {
		delete this->app_logging;
	}
}

void DetectionGold::clear_boxes_and_probs(box *boxes, real_t **probs, int n,
		int m) {
	for (int i = 0; i < n; i++) {
		boxes[i].x = 0;
		boxes[i].y = 0;
		boxes[i].w = 0;
		boxes[i].h = 0;
	}
	for (int i = 0; i < n; i++) {
		memset(probs[i], 0, sizeof(real_t) * m);
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

bool DetectionGold::compare(real_t **f_probs, box *f_boxes, int num,
		int classes, int img, int save_layers, int test_iteration) {

	ProbArray gold = this->pb_gold[img];
	std::string img_string = this->gold_img_names[img];

	// Check PR if critical
	std::vector<box> found_boxes;
	std::vector<box> gold_boxes;

	int error_count = 0;
	for (int i = 0; i < num; i++) {
		box g_b = gold.boxes[i];
		box f_b = f_boxes[i];

		int class_g = get_index(gold.probs[i].data(), classes);
		int class_f = get_index(f_probs[i], classes);

		real_t g_prob = gold.probs[i][class_g];
		real_t f_prob = f_probs[i][class_f];

		if (f_prob >= CONSIDERING_DETECTION) {
			found_boxes.push_back(f_b);
		}
		if (g_prob >= CONSIDERING_DETECTION) {
			gold_boxes.push_back(g_b);
		}

		std::string error_detail = error_check(f_prob, g_prob, f_b, g_b,
				img_string, class_g, class_f, i);

		if (error_detail != "") {
			error_count++;
			error_detail += " stream_modular_redundancy "
					+ std::to_string(this->stream_mr);
#ifdef LOGS
			log_error_detail(const_cast<char*>(error_detail.c_str()));
#else
			std << error_detail << std::endl;
#endif

		}

	}

#ifdef LOGS
	log_error_count(error_count);
	//printf("%d errors found at %s detection\n", error_count, img_string);
#endif
	return error_count;

}

std::string DetectionGold::print_box(box b) {
	return std::to_string(b.x) + " " + std::to_string(b.y) + " "
			+ std::to_string(b.w) + " " + std::to_string(b.h) + "\n";
}

std::ostream& operator<<(std::ostream& ret, DetectionGold &det) {
	for (int i = 0; i < det.plist_size; i++) {
		ret << det.gold_img_names[i] << "\n";
		ProbArray p = det.pb_gold[i];
		for (int j = 0; j < det.nboxes; j++) {
			ret << det.print_box(p.boxes[j]) << "\n";
		}
	}
	return ret;
}

/**
 * it was adapted from max_index in utils.c line 536image load_image(char *filename, int w, int h, int c);
 * auxiliary funtion for save_gold
 */
int DetectionGold::get_index(real_t *a, int n) {
	if (n <= 0)
		return -1;
	int i, max_i = 0;
	real_t max = a[0];
	for (i = 1; i < n; ++i) {
		if (a[i] > max) {
			max = a[i];
			max_i = i;
			//			printf("in time max %f\n", max);

		}
	}
	return max_i;
}
