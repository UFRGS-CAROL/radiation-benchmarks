/*
 * DetectionGold.cu
 *
 *  Created on: 28/09/2018
 *      Author: fernando
 */

#include <iterator>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "parse_gemm_layer.h"
#include "detection_gold.h"
#include "log_processing.h"

#define DEBUG_LINE(msg) printf("FILE: %s LINE: %d -- MSG: %s\n", __FILE__, __LINE__, msg);

//extern std::vector<std::string> split(const std::string&, char);
#include "helpful.h"

/**
 * Detection Gold class
 */

void DetectionGold::write_gold_header() const {
    //  0       1           2              3              4            5            6        7         8
    //  thresh; hier_tresh; img_list_size; img_list_path; config_file; config_data; model;weights;tensor_core;
    std::string gold_header = std::to_string(this->thresh) + ";";
    gold_header += std::to_string(this->hier_thresh) + ";";
    gold_header += std::to_string(this->plist_size) + ";";
    gold_header += this->img_list_path + ";";
    gold_header += this->config_file + ";";
    gold_header += this->cfg_data + ";";
    gold_header += this->model + ";";
    gold_header += this->weights + ";";
    gold_header += std::to_string(int(this->tensor_core_mode)) + ";";

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

    this->normalized_coordinates = find_int_arg(argc, argv, "-norm_coord", 0);

    this->iterations = find_int_arg(argc, argv, "-iterations", 1);
    this->tensor_core_mode = find_int_arg(argc, argv, "-tensor_cores", 0);
//	this->stream_mr = find_int_arg(argc, argv, "-smx_redundancy", 1);
//  this->compare_layers = find_arg(argc, argv, "-cmp_layer");
    this->compare_layers = false;
//    this->stream_mr = 1;

    this->thresh = thresh;
    this->hier_thresh = hier_thresh;
    this->total_errors = 0;

    std::cout << "Radiation test specific info\n";
    std::cout << "Norm. Coord.: " << this->normalized_coordinates << std::endl;
    std::cout << "Tensor cores: " << int(this->tensor_core_mode) << std::endl;
//	std::cout << "SMX redundancy: " << this->stream_mr << std::endl;
    std::cout << "Radiation test mode: " << this->generate << std::endl;
    std::cout << "Gold path: " << this->gold_inout << std::endl;

    Log::start_log(this->gold_inout, 0, 0, this->iterations, this->network_name,
                   this->tensor_core_mode); //, this->stream_mr);
    if (!this->generate) {
        //	current_op = SIMULATE_SCHEDULER_FAULT;
        //  detection gold;
        std::string line;
        std::ifstream gold_file(this->gold_inout, std::ifstream::in);
        if (gold_file.is_open()) {
            getline(gold_file, line);
        } else {
            std::cerr << "ERROR ON OPENING GOLD FILE\n";
            exit(-1);
        }

        std::vector <std::string> split_ret = split(line, ';');
        //  0       1           2              3              4            5            6      7         8
        //  thresh; hier_tresh; img_list_size; img_list_path; config_file; config_data; model;weights;tensor_core;
        this->thresh = std::stof(split_ret[0]);
        this->hier_thresh = std::stof(split_ret[1]);
        this->plist_size = std::stoi(split_ret[2]);
        this->img_list_path = split_ret[3];
        this->config_file = split_ret[4];
        this->cfg_data = split_ret[5];
        this->model = split_ret[6];
        this->weights = split_ret[7];
        this->tensor_core_mode = std::stoi(split_ret[8]);

        //allocate detector
        this->load_gold_hash(gold_file);

        gold_file.close();

#ifdef LOGS
#ifdef BUILDPROFILER
        auto str = this->app_log->get_log_path();
        this->profiler_thread = std::make_shared<rad::OBJTYPE>(0, str);

        //START PROFILER THREAD
        profiler_thread->start_profile();
#endif
#endif
    } else {
        this->img_list_path = std::string(img_list_path);

        //reading the img list path content
        std::ifstream tmp_img_file(this->img_list_path);
        std::copy(std::istream_iterator<std::string>(tmp_img_file),
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
    //set_layer_processing_parameters(current_op);
}

bool operator!=(const box &a, const box &b) {
    real_t x_diff = std::fabs((a.x - b.x));
    real_t y_diff = std::fabs((a.y - b.y));
    real_t w_diff = std::fabs((a.w - b.w));
    real_t h_diff = std::fabs((a.h - b.h));

    return (x_diff > THRESHOLD_ERROR || y_diff > THRESHOLD_ERROR
            || w_diff > THRESHOLD_ERROR || h_diff > THRESHOLD_ERROR);
}

bool operator!=(const std::tuple <real_t, real_t, real_t, real_t> f,
                std::tuple <real_t, real_t, real_t, real_t> g) {

    real_t x1_diff = std::fabs(std::get<0>(f) - std::get<0>(g));
    real_t x2_diff = std::fabs(std::get<1>(f) - std::get<1>(g));
    real_t x3_diff = std::fabs(std::get<2>(f) - std::get<2>(g));
    real_t x4_diff = std::fabs(std::get<3>(f) - std::get<3>(g));

    return (x1_diff > THRESHOLD_ERROR_INTEGER
            || x2_diff > THRESHOLD_ERROR_INTEGER
            || x3_diff > THRESHOLD_ERROR_INTEGER
            || x4_diff > THRESHOLD_ERROR_INTEGER);
}

int DetectionGold::compare_detection(const Detection &g_det, const detection &f_det, const std::string &img, int nb,
                                     int classes, int img_w, int img_h) const {

    box g_box = g_det.bbox;
    box f_box = f_det.bbox;

    real_t g_objectness = g_det.objectness;
    real_t f_objectness = f_det.objectness;

    int g_sort_class = g_det.sort_class;
    int f_sort_class = f_det.sort_class;

    const real_t *g_probs = g_det.prob.data();
    const real_t *f_probs = f_det.prob;

    real_t objs_diff = std::fabs(g_objectness - f_objectness);
    int sortc_diff = std::fabs(g_sort_class - f_sort_class);

    int error_count = 0;

    if ((objs_diff > THRESHOLD_ERROR) || (sortc_diff > THRESHOLD_ERROR)
        || (g_box != f_box)) {
        std::stringstream error_info("");
        error_info << std::setprecision(STORE_PRECISION);

        error_info << " img: " << img << " detection: " << nb << " x_e: " << g_box.x << " x_r: " << f_box.x << " y_e: "
                   << g_box.y << " y_r: " << f_box.y << " h_e: " << g_box.h
                   << " h_r: " << f_box.h << " w_e: " << g_box.w << " w_r: "
                   << f_box.w << " objectness_e: " << g_objectness
                   << " objectness_r: " << f_objectness << " sort_class_e: "
                   << g_sort_class << " sort_class_r: " << f_sort_class
                   << " img_w: " << img_w << " img_h: " << img_h;

        auto err_str = error_info.str();
        Log::log_error_info(err_str);
        std::cout << err_str << "\n";
        error_count++;
    }

    for (int cl = 0; cl < classes; ++cl) {
        real_t f_prob = f_probs[cl];
        real_t g_prob = g_probs[cl];

        real_t prob_diff = std::fabs(g_prob - f_prob);
        if ((g_prob >= this->thresh || f_prob >= this->thresh)
            && prob_diff > THRESHOLD_ERROR) {
            std::stringstream error_info("");
            error_info << std::setprecision(STORE_PRECISION);
            error_info << " img: " << img << " detection: "
                       << nb << " class: " << cl << " prob_e: " << g_prob
                       << " prob_r: " << f_prob;

            auto err_str = error_info.str();
            Log::log_error_info(err_str);
            std::cout << err_str << "\n";

            error_count++;
        }
    }

//	DEBUG_LINE("AFTER CHECKING");

    return error_count;
}

int DetectionGold::cmp(detection *found_dets, int nboxes, int img_index, int classes, int img_w, int img_h) {
    std::string img = this->gold_img_names[img_index];
    int error_count = 0;
    auto gold_boxes = this->gold_hash_var[img].size();

    if (nboxes >= 0) {
        //Check if detection is more boxes than the gold has
        int boxes_diff = nboxes - gold_boxes;

        //IT MUST BE A COPY
        std::vector <Detection> gold_dets = this->gold_hash_var[img];

        if (boxes_diff > 0) {    //it has more detected boxes than expected
            for (int i = 0; i < boxes_diff; i++) {
                gold_dets.push_back(gold_dets.back());
            }
        } else if (boxes_diff < 0) { //when there are less boxes than expected
            std::string err_str = "Smaller number of boxes detected:" + std::to_string(nboxes) + " while gold is:"
                                  + std::to_string(gold_boxes);

            Log::log_error_info(err_str);
            error_count += std::abs(boxes_diff);
        }

        for (int nb = 0; nb < nboxes; nb++) {
            const Detection &g_det = gold_dets[nb];
            const detection &f_det = found_dets[nb];

            //Only basic types are passed to this functions
            error_count += this->compare_detection(g_det, f_det, img, nb, classes, img_w, img_h);
        }

    } else {
        std::string err_str = "Number of boxes detected:" + std::to_string(nboxes) + " while gold is:"
                              + std::to_string(gold_boxes);

        Log::log_error_info(err_str);
        error_count = gold_boxes;

    }
    this->total_errors += error_count;

    Log::update_error_count(error_count);

    return error_count;
}

int DetectionGold::run(detection *dets, int nboxes, int img_index, int classes, int img_w, int img_h) {
    int ret = 0;
    // To generate function
    //std::string img, detection* dets, int nboxes, int classes, int l_coord
    if (this->generate) {

        std::ofstream gold_file(this->gold_inout, std::ofstream::app);
        if (!gold_file.is_open()) {
            std::cerr << "ERROR ON OPENING GOLD FILE\n";
            exit(-1);
        }

        //assuming at least one execution is completed
        this->gen(dets, nboxes, img_index, gold_file, classes);
        gold_file.close();
    } else {
        // To compare function
        //detection is always nboxes size
//		for (int inet = 0; inet < this->stream_mr; inet++) {
        ret += this->cmp(dets, nboxes, img_index, classes, img_w, img_h);
//		}

        //reset_counters();
    }
    return ret;
}

void DetectionGold::gen(detection *dets, int nboxes, int img_index, std::ofstream &gold_file, int classes) {
    //first write the image string name
    std::string img = this->gold_img_names[img_index];

    gold_file << img << ";" << nboxes << ";" << std::endl;

    for (int bb = 0; bb < nboxes; bb++) {
        detection det = dets[bb];
        box b = det.bbox;

        std::string box_str = DetectionGold::generate_gold_line(bb, det, b, dets);
        gold_file << box_str << std::endl;
    }

}

std::string DetectionGold::generate_gold_line(int bb, detection det, const box &b, detection *dets) {
    std::ostringstream box_str("");
    box_str.precision(STORE_PRECISION);

    box_str << dets[bb].objectness << ";";
    box_str << det.sort_class << ";";
    box_str << b.x << ";";
    box_str << b.y << ";";
    box_str << b.w << ";";
    box_str << b.h << ";";
    box_str << det.classes << ";" << std::endl;

    for (int cl = 0; cl < det.classes; cl++) {
        box_str << det.prob[cl] << ";";
    }
    return box_str.str();
}

void DetectionGold::load_gold_line(std::ifstream &gold_file, Detection &det,
                                   int nboxes) {
    std::string line;
    getline(gold_file, line);
    det.nboxes = nboxes;

    auto splited_line = split(line, ';');
    det.objectness = std::stof(splited_line[0]);
    det.sort_class = std::stoi(splited_line[1]);
    det.bbox.x = std::stof(splited_line[2]);
    det.bbox.y = std::stof(splited_line[3]);
    det.bbox.w = std::stof(splited_line[4]);
    det.bbox.h = std::stof(splited_line[5]);
    det.classes = std::stoi(splited_line[6]);

    // Getting the probabilities
    det.prob.resize(det.classes);

    if (getline(gold_file, line)) {
        splited_line = split(line, ';');

        for (auto cl = 0; cl < det.classes; cl++) {
            det.prob[cl] = std::stof(splited_line[cl]);
        }
    }
}

void DetectionGold::load_gold_hash(std::ifstream &gold_file) {
//allocate detector
    this->gold_img_names = std::vector<std::string>(this->plist_size);
    std::string line;

    for (int i = 0; i < this->plist_size; i++) {
        getline(gold_file, line);
        std::vector <std::string> splited_line = split(line, ';');

        // Set each img_name path
        this->gold_img_names[i] = splited_line[0];

        // Prob array creation

        int nboxes = std::stoi(splited_line[1]);

        auto &detections = this->gold_hash_var[this->gold_img_names[i]];
        detections.resize(nboxes);

        for (int bb = 0; bb < nboxes; ++bb) {
            this->load_gold_line(gold_file, detections[bb], nboxes);
        }
    }

}

DetectionGold::~DetectionGold() {
    if (!this->generate) {
#ifdef LOGS
#ifdef BUILDPROFILER
        profiler_thread->end_profile();
#endif
#endif
    }
    Log::end_log();
}
