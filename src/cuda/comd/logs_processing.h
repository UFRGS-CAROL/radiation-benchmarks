/*
 * logs_processing.h
 *
 *  Created on: 07/09/2017
 *      Author: fernando
 */

#ifndef LOGS_PROCESSING_H_
#define LOGS_PROCESSING_H_

#ifdef __cplusplus
extern "C" {
#endif


void start_count_app(char *gold_file, int iterations);
void finish_count_app();
void start_iteration_app();
void end_iteration_app();


#ifdef __cplusplus
} //end extern "C"
#endif //end IF __cplusplus

#endif /* LOGS_PROCESSING_H_ */
