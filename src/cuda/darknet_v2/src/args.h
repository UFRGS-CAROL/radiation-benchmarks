/*
 * args.h
 *
 *  Created on: 22/09/2016
 *      Author: fernando
 */

#ifndef ARGS_H_
#define ARGS_H_
#include <getopt.h> //opt functions
#include <unistd.h> //acess F_OK
#include <stdio.h> //printf
#include <stdlib.h> //atol
#include "abft.h"

//#define MAX_ABFT_TYPES 8

typedef enum {
	 NONE,
	 GEMM,
	 SMART_POOLING,
	 L1_HARDENING,
	 L2_HARDENING,
	 TRAINED_WEIGHTS,
	 SMART_DMR,
	 SMART_TMR
} abft_type ;


/**
 * -c - config_file = configuration file
 * -w - weights = neural network weights
 * -i - input_data_path = path to all input data *.jpg files
 * -n - iterations = how many radiation iterations
 * -g - generate   = generates a gold
 */
typedef struct arguments {
	char *config_file;
	char *weights;
	long iterations;
	char *gold_inout;
	int generate_flag;
	//input images ...
	char *img_list_path;
	int save_layers;

	//test detector
	char *cfg_data;
	char *model;

	float thresh;
	float hier_thresh;

	abft_type abft;

	//use tensor cores
	unsigned char use_tensor_cores;
} Args;

void args_init_and_setnull(Args *arg);
/**
 * return 1 if everything is ok, and 0 if not
 */
int check_args(Args *arg);
/**
 * print the passed arg
 */
void print_args(const Args arg);

/**
 * @parse_arguments
 * parameter arguments to_parse
 * return 0 ok, -1 wrong
 */
int parse_arguments(Args *to_parse, int argc, char **argv);

void usage(char **argv, char *model, char *message);

#endif /* ARGS_H_ */
