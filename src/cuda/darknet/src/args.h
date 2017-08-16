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

#define MAX_ABFT_TYPES 5

/**
 * -e - execution_type = <yolo/classifier/imagenet...>
 * -m - execution_model = <test/train/valid>
 * -c - config_file = configuration file
 * -w - weights = neural network weights
 * -i - input_data_path = path to all input data *.jpg files
 * -n - iterations = how many radiation iterations
 * -g - generate   = generates a gold
 */
typedef struct arguments {
	char *execution_type;
	char *execution_model;
	char *config_file;
	char *weights;
//  char *input_data_path;
	long int iterations;
	int generate_flag;
	//if yolo test
	char *test_filename;
	int cam_index;
	float thresh;
	int frame_skip;

	//input images ...
	char *img_list_path;
	char *base_result_out;

	int gpu_index;
	char *gold_inout;
	int save_layers;
	int abft;

	char *cfg_data;
	char *model;
	float hier_thresh;

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
