/*
 * args.c
 *
 *  Created on: 25/09/2016
 *      Author: fernando
 */

#include "args.h"

/**
 * Init all args to default
 */
void args_init_and_setnull(Args *arg) {
	arg->config_file = NULL;
	arg->weights = NULL;
	arg->gold_inout = NULL;
	arg->img_list_path = NULL;
	arg->iterations = 1;
	arg->save_layers = 0;
	arg->generate_flag = 0;

	//test detector
	arg->cfg_data = "cfg/coco.data";
	arg->model = "detect";
	arg->thresh = 0.24;
	arg->hier_thresh = 0.5;
	arg->abft = 0;
}

/**
 * return 1 if it is a generation execution
 * 2 if it is a test
 * -1 if something is wrong
 *
 */
int check_args(Args *arg) {
	if (arg->iterations < 0) {
		printf("Use a valid value for iterations\n");
		return -1;
	}

	if (arg->generate_flag == 0 && arg->gold_inout == NULL) {
		printf("If generate is not set, gold input must be passed\n");
		return -1;
	} else if (arg->gold_inout != NULL && arg->generate_flag == 0) {
		return 2;
	}

	//check config_file
	if (access(arg->config_file, F_OK) == -1) {
		printf("Config file does not exist\n");
		return -1;
	}
	//check weights
	if (access(arg->weights, F_OK) == -1) {
		printf("Weights does not exist\n");
		return -1;
	}

	if (arg->generate_flag == 1 && arg->gold_inout == NULL) {
		printf("Generate gold path not passed\n");
		return -1;
	}

	//check img_list_path
	if (access(arg->img_list_path, F_OK) == -1) {
		printf("img_list_path does not exist\n");
		return -1;
	}

	//make sure if it is generate is only one iteration
	arg->iterations = ((arg->generate_flag) ? 1 : arg->iterations);
	return 1;
}
/**
 * print the passed arg
 */
void print_args(const Args arg) {
	printf("config file = %s\n"
			"weights = %s\n"
			"iterations = %ld\n"
			"gold_input/output = %s\n"
			"gold_flag = %d\n"
			"img_list_path = %s\n"
			"save_layer = %d\n"
			"model = %s\n"
			"cfg_data = %s\n"
			"threshold = %f\n"
			"hier_thresh = %f\n"
			"abft = %d\n", arg.config_file, arg.weights, arg.iterations,
			arg.gold_inout, arg.generate_flag, arg.img_list_path,
			arg.save_layers, arg.model, arg.cfg_data, arg.thresh,
			arg.hier_thresh, arg.abft);
}

/**
 * @parse_arguments
 * parameter arguments to_parse
 * return 0 ok, -1 wrong
 */
int parse_arguments(Args *to_parse, int argc, char **argv) {
	static struct option long_options[] = { 	//options
			{ "config_file", required_argument, NULL, 'c' }, //<yolo, imagenet..>.cfg
					{ "weights", required_argument, NULL, 'w' }, //<yolo, imagenet..>weights
					{ "iterations", required_argument, NULL, 'n' }, //log data iterations
					{ "generate", required_argument, NULL, 'g' }, //generate gold
					{ "img_list_path", required_argument, NULL, 'l' }, //data path list input
					{ "gold_inout", required_argument, NULL, 'd' },  //gold path
					{ "save_layers", required_argument, NULL, 's' }, //save layers
					{ "abft", required_argument, NULL, 'a'},
					{ NULL, 0, NULL, 0 } };

	// loop over all of the options
	char ch;
	int option_index = 0;
	to_parse->generate_flag = 0;
	int max_args = 12;
	while ((ch = getopt_long(argc, argv, "c:w:n:g:l:d:s:a:", long_options,
			&option_index)) != -1 && --max_args) {

		// check to see if a single character or long option came through
		switch (ch) {

		case 'c': {
			to_parse->config_file = optarg;
			break;
		}
		case 'w': {
			to_parse->weights = optarg;
			break;
		}

		case 'n': {
			to_parse->iterations = atol(optarg);
			break;

		}
		case 'g': {
			to_parse->gold_inout = optarg;
			to_parse->generate_flag = 1;
			break;
		}
		case 'l': {
			to_parse->img_list_path = optarg;
			break;
		}

		case 'd': {
			to_parse->gold_inout = optarg;
			break;
		}

		case 's': {
			to_parse->save_layers = atoi(optarg);
			break;
		}

		case 'a': {
			printf("abft %s\n", optarg);
			to_parse->abft = atoi(optarg);
			break;
		}

		}

	}
	print_args(*to_parse);
	return check_args(to_parse);

}

void usage(char **argv, char *model, char *message) {
	printf("Some argument is missing, to use %s option\n", model);
	printf("usage: %s %s ", argv[0], message);
	printf(
			"\n-c --config_file = configuration file\n"
					"-w --weights = neural network weights\n"
					"-n --iterations = how many radiation iterations\n"
					"-g --generate   = generates a gold\n"
					"-l --img_list_path = list for all dataset image\n"
					"-d --gold_inout = if not writing a gold a gold is being reading\n"
					"-s --save_layers = this must set to 1 if you want to save all wrong computed layers\n");
}
