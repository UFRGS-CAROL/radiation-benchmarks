/*
 * detection_gold_w.h
 *
 *  Created on: 02/10/2018
 *      Author: fernando
 */

#ifndef DETECTION_GOLD_W_H_
#define DETECTION_GOLD_W_H_

#include "darknet.h"


#ifdef __cplusplus
extern "C" {
#endif

struct detection_gold;
typedef struct detection_gold detection_gold_t;

detection_gold_t* create_detection_gold(int argc, char **argv, real_t thresh,
		real_t hier_thresh, char *img_list_path, char *config_file,
		char *config_data, char *model, char *weights);

void destroy_detection_gold(detection_gold_t *m);

int run(detection_gold_t *m, detection* dets, int nboxes, int img_index, int classes, int img_w, int img_h);

void start_iteration_wrapper(detection_gold_t *m);
void end_iteration_wrapper(detection_gold_t *m);

int get_iterations(detection_gold_t *m);

int get_img_num(detection_gold_t *m);

unsigned char get_use_tensor_cores(detection_gold_t *m);

//int get_smx_redundancy(detection_gold_t *m);

#ifdef __cplusplus
}
#endif

#endif /* DETECTION_GOLD_W_H_ */
