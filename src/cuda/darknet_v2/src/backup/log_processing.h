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

typedef struct rect {
	float left;
	float top;
	float right;
	float bottom;
	float prob;
	int class_;
} rectangle;

typedef struct detection_ {
	char **image_names;
	rectangle **detection_result;
	int img_list_size;
	int *rect_list_size;
} detection;

#ifdef __cplusplus
extern "C" {
#endif

rectangle init_rectangle(int, float, float, float, float, float);
void print_rectangle(rectangle);

void start_count_app(char*, char*);

void finish_count_app();

void saveLayer(network, int, int);
void compareLayer(layer, int);

char** get_image_filenames(char*, int*);

void save_gold(FILE*, int, int, int, float, box*, float**, int);

void delete_detection_var(detection*, Args*);

detection load_gold(Args*);

int compare_detections(int, int, int, float, box*, float**, int);

void clear_boxes_and_probs(box*, float**, int, int);

void print_detection(detection);

/**
 * magic function
 * (rectangle *gold_rect, float **found_probs, box *found_boxes,
 int img_iteration, network net, int test_iteration, bool save_layer)
 */
void compare(rectangle*, int, int, float**, box*, int, network, int, int, float, int, int);

#ifdef __cplusplus
} //end extern "C"
#endif //end IF __cplusplus

#endif /* LOG_PROCESSING_H_ */
