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

#include "args.h" //load gold

#include <stdio.h> //FILE

#define THRESHOLD_ERROR 0.005

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
} detection;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * functions to start log file
 */
void start_count_app(char*, char*);

void finish_count_app();

/**
 * compare and save layers
 */
void saveLayer(network, int, int);
void compareLayer(layer, int);

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

int compare_detections();

void clear_boxes_and_probs(box*, float**, int, int);

void print_detection(detection);

void compare(prob_array gold, float **f_probs, box *f_boxes, int num,
		int classes, int img, int save_layer, network net, int test_iteration);

#ifdef __cplusplus
} //end extern "C"
#endif //end IF __cplusplus

#endif /* LOG_PROCESSING_H_ */
