/*
 * log_processing.h
 *
 *  Created on: 30/04/2017
 *      Author: fernando
 */

#ifndef LOG_PROCESSING_H_
#define LOG_PROCESSING_H_

#include <sys/time.h> //cont time
#include "network.h" //save layer
#include "layer.h" //save layer
#include "box.h" //boxes


#include "abft.h"
#include "args.h" //load gold

#include <stdio.h> //FILE

#define THRESHOLD_ERROR 0.05
#define LAYER_THRESHOLD_ERROR 0.0000001

#define LAYER_GOLD "/var/radiation-benchmarks/data/"


static const char *ABFT_TYPES[] = { "none", "gemm", "smart_pooling", "l1", "l2",
		"trained_weights" };


typedef struct prob_array_ {
	box *boxes;
	float **probs;

} prob_array;

//to store all gold content
typedef struct detection_ {
	prob_array *pb_gold;
	int plist_size;
	int classes;
	int total;
	char **img_names;

	//layers vars
	float **found_layers;
	float **gold_layers;
	network *net;
	int layers_size;

	char *network_name;
} detection;

#ifdef __cplusplus
extern "C" {
#endif

//#ifdef LOGS
//char * get_log_file_name();
//int start_log_file(char *benchmark_name, char *test_info);
//int end_log_file();
//int start_iteration();
//int end_iteration();
//int log_error_count(unsigned long int kernel_errors);
//int log_error_detail(char *string);
//int log_info_detail(char *string);
//#endif

/**
 * functions to start log file
 */
void start_count_app(char *test, int save_layer, int abft, int iterations,
		char *app);

void finish_count_app();

void start_iteration_app();
void end_iteration_app();

/**
 * compare and save layers
 */
void save_layer(detection *det, int img_iterator, int test_iteration,
		char *log_filename, int generate, char *img_list_filename);

void alloc_gold_layers_arrays(detection *det, network *net);

/**
 * get_image_filenames are used by generate
 */
char** get_image_filenames(char*, int*);

void save_gold(FILE *fp, char *img, int num, int classes, float **probs,
		box *boxes);

/**
 * radiation test functions
 */

void delete_detection_var(detection*, Args*);

detection load_gold(Args*);

void compare(detection *det, float **f_probs, box *f_boxes, int num,
		int classes, int img, int save_layer, int test_iteration,
		char *img_list_path, error_return max_pool_errors);

void clear_boxes_and_probs(box*, float**, int, int);

void print_detection(detection);

#ifdef __cplusplus
} //end extern "C"
#endif //end IF __cplusplus

#endif /* LOG_PROCESSING_H_ */
