/*
 * log_processing.h
 *
 *  Created on: 22/09/2016
 *      Author: fernando
 */

#ifndef LOG_PROCESSING_H_
#define LOG_PROCESSING_H_

#include <sys/time.h>
#include "yolo.h"
#include "network.h"
#include "layer.h"
#include "box.h" //boxes
#include <stdio.h> //FILE*
#include <math.h> //fabs
#include <stdlib.h> //calloc
#include <string.h>

#define THRESHOLD_ERROR 0.005



typedef struct prob_arry {
	box *boxes;
	float **probs;
	long classes;
	long total_size;
} ProbArray;

//to store all gold filenames
typedef struct gold_pointers {
//	box *boxes_gold;
//	ProbArray pb;
	ProbArray *pb_gold;
	long plist_size;
	FILE* gold;
	int has_file;
} GoldPointers;

//allocate all memory
GoldPointers new_gold_pointers(int classes, int total_size,
		const int plist_size, char *file_path, char *open_mode);

//clean memory
void free_gold_pointers(GoldPointers *gp);

/**
 * The output will be stored in this order
 long plist_size;
 long classes;
 long total_size;
 for(<plist_size times>){
 -----pb_gold.boxes
 -----pb_gold.probs
 }
 */
void gold_pointers_serialize(GoldPointers gp);
//don't mess up with the memory in C, shit gonna happens
//void free_gold_pointers(GoldPointers *gp);

/**
 *  the input must be read in this order
 long plist_size;
 long classes;
 long total_size;
 for(<plist_size times>){
 -----pb_gold.boxes
 -----pb_gold.probs
 }
 */
void read_yolo_gold(GoldPointers *gp);

/**
 * if some error happens the error_count will be != 0
 */
unsigned long comparable_and_log(GoldPointers gold, GoldPointers current);

void clear_vectors(GoldPointers *gp);

int prob_array_comparable_and_log(ProbArray gold, ProbArray pb, long plist_iteration);

void saveLayer(network net, int iterator, int n);
void compareLayer(layer l, int i);

inline double mysecond();

#endif /* LOG_PROCESSING_H_ */
