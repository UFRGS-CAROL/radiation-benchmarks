/*
 * logs_processing.h
 *
 *  Created on: 07/09/2017
 *      Author: fernando
 */

#ifndef LOGS_PROCESSING_H_
#define LOGS_PROCESSING_H_
#include "CoMDTypes.h"

#ifdef __cplusplus
extern "C" {
#endif




typedef struct{
	char gold_path[2048];
	int iterations;
	SimFlat **gold_data;
	int steps;
}Gold;


void start_count_app(char *gold_file, int iterations);
void finish_count_app();
void start_iteration_app();
void end_iteration_app();

void save_gold(Gold *g);
void compare_and_log(SimFlat *gold, SimFlat *found);

void load_gold(Gold *g);
void init_gold(Gold *g, char *gold_path, int steps);
void destroy_gold(Gold *g);


#ifdef __cplusplus
} //end extern "C"
#endif //end IF __cplusplus

#endif /* LOGS_PROCESSING_H_ */
