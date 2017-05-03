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


typedef struct rect{
	float left;
	float top;
	float right;
	float bottom;
	float prob;
	int class_;
}rectangle;


typedef struct detection_{
	char **image_names;
	rectangle **detection_result;
	int img_list_size;
	int rect_list_size;
}detection;


#ifdef __cplusplus
extern "C" {
#endif

inline rectangle init_rectangle(int class_, float left, float top, float right, float bottom, float prob);


void start_count_app(char *test, char *app);

void finish_count_app();

void saveLayer(network net, int iterator, int n);
void compareLayer(layer l, int i);

char** get_image_filenames(char *img_list_path, int *image_list_size);

void save_gold(FILE *fp, int w, int h, int num, float thresh, box *boxes,
		float **probs, int classes);

void delete_detection_var(detection *det);

detection load_gold(Args *arg);

int compare_detections(int w, int h, int num, float thresh, box *boxes,
		float **probs, int classes);


void clear_boxes_and_probs(box *boxes, float **probs, int n);



#ifdef __cplusplus
} //end extern "C"
#endif //end IF __cplusplus

#endif /* LOG_PROCESSING_H_ */
