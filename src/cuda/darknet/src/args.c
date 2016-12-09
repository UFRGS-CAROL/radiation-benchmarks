/*
 * args.c
 *
 *  Created on: 25/09/2016
 *      Author: fernando
 */

#include "args.h"

void args_init_and_setnull(Args *arg) {
	arg->config_file = NULL;
	arg->execution_model = NULL;
	arg->config_file = NULL;
	arg->weights = NULL;
//  arg->input_data_path = NULL;
	arg->gold_output = NULL;
	arg->base_result_out = NULL;
	arg->cam_index = -1;
	arg->frame_skip = -1;
	arg->gold_output = 0;
	arg->img_list_path = NULL;
	arg->iterations = 1;
	arg->gold_input = NULL;
	arg->save_layers = 0;
	arg->abft = 0;

}
/**
 * return 1 if everything is ok, and 0 if not
 */
int check_args(Args *arg) {
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

	if (arg->iterations < 0) {
		printf("Use a valid value for iterations\n");
		return -1;
	}

//  if (arg->input_data_path == NULL) {
//      printf("No input path set\n");
//      return -1;
//  }
	if (arg->generate_flag == 1 && arg->gold_output == NULL) {
		printf("Generate gold path not passed\n");
		return -1;
	}

	if (arg->generate_flag == 0 && arg->gold_input == NULL) {
		printf("If generate is not set, gold input must be passed\n");
		return -1;
	}

	if (arg->img_list_path == NULL) {
		printf("Img list not passed\n");
		return -1;
	}
	if (arg->base_result_out == NULL) {
		printf("Base output not passed\n");
		return -1;
	}

	if (arg->gpu_index > 5 && arg->gpu_index < -2) {
		printf("gpu_index not passed\n");
		return -1;
	}

	if (arg->gpu_index > 5 && arg->gpu_index < -2) {
		printf("gpu_index not passed\n");
		return -1;
	}

	//make sure if it is generate is only one iteration
	arg->iterations = ((arg->generate_flag) ? 1 : arg->iterations);
	return 0;
}
/**
 * print the passed arg
 */
void print_args(const Args arg) {
	printf(
			"execution type = %s\n"
					"execution model = %s\n"
					"config file = %s\n"
					"weights = %s\n"
//          "input_data_path = %s\n"
					"iterations = %ld\n"
					"gold_input/output = %s\n"
					"gold_flag = %d\n"
					"img_list_path = %s\n"
					"base_result_out = %s\n"
					"gpu_index = %d\n"
					"save_layer = %d\n"
					"abft = %d\n", arg.execution_type,
			arg.execution_model,
			arg.config_file, arg.weights, arg.iterations,
			((arg.generate_flag == 0) ? arg.gold_input : arg.gold_output),
			arg.generate_flag, arg.img_list_path, arg.base_result_out,
			arg.gpu_index, arg.save_layers, arg.abft);
}

/**
 * @parse_arguments
 * parameter arguments to_parse
 * return 0 ok, -1 wrong
 */
int parse_arguments(Args *to_parse, int argc, char **argv) {
	static struct option long_options[] = { { "execution_type",
			required_argument, NULL, 'e' }, //yolo/cifar/imagenet...
			{ "execution_model", required_argument, NULL, 'm' }, //test/valid...
			{ "config_file", required_argument, NULL, 'c' }, //<yolo, imagenet..>.cfg
			{ "weights", required_argument, NULL, 'w' }, //<yolo, imagenet..>weights
//          { "input_data_path",    required_argument, NULL, 'i' },
			{ "iterations", required_argument, NULL, 'n' }, //log data iterations
			{ "generate", required_argument, NULL, 'g' }, //generate gold
			{ "img_list_path", required_argument, NULL, 'l' }, //data path list input
			{ "base_result_out", required_argument, NULL, 'b' }, //result output
			{ "gpu_index", required_argument, NULL, 'x' }, //gpu index
			{ "gold_input", required_argument, NULL, 'd' },
			{ "save_layers",required_argument, NULL, 's' },
			{ "abft",       required_argument, NULL, 'a' },
					{ NULL, 0, NULL, 0 } };

	// loop over all of the options
	char ch;
	int ok = -1;
	int option_index = 0;
	to_parse->generate_flag = 0;
	int max_args = 12;
	while ((ch = getopt_long(argc, argv, "e:m:c:w:i:n:g:l:b:x:d:s:a:", long_options,
			&option_index)) != -1 && --max_args) {
		// check to see if a single character or long option came through
		switch (ch) {

		case 'e': {
			to_parse->execution_type = optarg; // or copy it if you want to
			break;
		}
		case 'm': {
			to_parse->execution_model = optarg; // or copy it if you want to
			break;
		}
		case 'c': {
			to_parse->config_file = optarg;
			break;
		}
		case 'w': {
			to_parse->weights = optarg;
			break;
		}
//      case 'i': {
//          to_parse->input_data_path = optarg;
//          break;
//      }
		case 'n': {
			to_parse->iterations = atol(optarg);
			break;

		}
		case 'g': {
			to_parse->gold_output = optarg;
			to_parse->generate_flag = 1;
			break;
		}
		case 'l': {
			to_parse->img_list_path = optarg;
			break;
		}
		case 'b': {
			to_parse->base_result_out = optarg;
			break;
		}

		case 'x': {
			to_parse->gpu_index = atoi(optarg);
			break;
		}

		case 'd': {
			to_parse->gold_input = optarg;
			break;
		}

		case 's': {
			to_parse->save_layers = atoi(optarg);
			break;
		}

		case 'a': {
			to_parse->abft = atoi(optarg);
			break;
		}
		}

		ok = 0;
	}
	print_args(*to_parse);
	return (ok || check_args(to_parse));

}

void usage(char **argv, char *model, char *message) {
	printf("Some argument is missing, to use %s option\n", model);
	printf("usage: %s %s ", argv[0], message);
	printf("\n-e --execution_type = <yolo/classifier/imagenet...>\n"
			"-m --execution_model = <test/train/valid>\n"
			"-c --config_file = configuration file\n"
			"-w --weights = neural network weights\n"
			//"-i --input_data_path = path to all input data *.jpg files\n"
			"-n --iterations = how many radiation iterations\n"
			"-g --generate   = generates a gold\n"
			"-l --img_list_path = list for all dataset image\n"
			"-b --base_result_out = output of base\n"
			"-x --gpu_index = GPU index\n"
			"-d --gold_input = if not writing a gold a gold is being reading\n"
			"-s --save_layers = this must set to 1 if you want to save all wrong computed layers\n"
			"-a --abft = this must be set to 1 or 2 to use abft, 1 for dumb abft and 2 for smart one\n");
}
