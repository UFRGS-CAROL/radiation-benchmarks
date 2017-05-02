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

#include <stdio.h> //FILE


#define THRESHOLD_ERROR 0.005

#ifdef __cplusplus
extern "C" {
#endif

double mysecond();

void start_count_app(char *test, char *app);

void finish_count_app();

void saveLayer(network net, int iterator, int n);
void compareLayer(layer l, int i);

const char** get_image_filenames(char *img_list_path, int *image_list_size);

void save_gold(FILE *fp, image im, int num, float thresh, box *boxes,
		float **probs, int classes);

#ifdef __cplusplus
} //end extern "C"
#endif

#endif /* LOG_PROCESSING_H_ */
