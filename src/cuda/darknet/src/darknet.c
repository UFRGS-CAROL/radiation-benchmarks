#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "parser.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "connected_layer.h"
#include <unistd.h>
#include <getopt.h>
#include <limits.h>

#include "yolo.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

void change_rate(char *filename, float scale, float add) {
	// Ready for some weird shit??
	FILE *fp = fopen(filename, "r+b");
	if (!fp)
		file_error(filename);
	float rate = 0;
	fread(&rate, sizeof(float), 1, fp);
	printf("Scaling learning rate from %f to %f\n", rate, rate * scale + add);
	rate = rate * scale + add;
	fseek(fp, 0, SEEK_SET);
	fwrite(&rate, sizeof(float), 1, fp);
	fclose(fp);
}

void average(int argc, char *argv[]) {
	char *cfgfile = argv[2];
	char *outfile = argv[3];
	gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	network sum = parse_network_cfg(cfgfile);

	char *weightfile = argv[4];
	load_weights(&sum, weightfile);

	int i, j;
	int n = argc - 5;
	for (i = 0; i < n; ++i) {
		weightfile = argv[i + 5];
		load_weights(&net, weightfile);
		for (j = 0; j < net.n; ++j) {
			layer l = net.layers[j];
			layer out = sum.layers[j];
			if (l.type == CONVOLUTIONAL) {
				int num = l.n * l.c * l.size * l.size;
				axpy_cpu(l.n, 1, l.biases, 1, out.biases, 1);
				axpy_cpu(num, 1, l.filters, 1, out.filters, 1);
			}
			if (l.type == CONNECTED) {
				axpy_cpu(l.outputs, 1, l.biases, 1, out.biases, 1);
				axpy_cpu(l.outputs * l.inputs, 1, l.weights, 1, out.weights, 1);
			}
		}
	}
	n = n + 1;
	for (j = 0; j < net.n; ++j) {
		layer l = sum.layers[j];
		if (l.type == CONVOLUTIONAL) {
			int num = l.n * l.c * l.size * l.size;
			scal_cpu(l.n, 1. / n, l.biases, 1);
			scal_cpu(num, 1. / n, l.filters, 1);
		}
		if (l.type == CONNECTED) {
			scal_cpu(l.outputs, 1. / n, l.biases, 1);
			scal_cpu(l.outputs * l.inputs, 1. / n, l.weights, 1);
		}
	}
	save_weights(sum, outfile);
}

void operations(char *cfgfile) {
	gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	int i;
	long ops = 0;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		if (l.type == CONVOLUTIONAL) {
			ops += 2 * l.n * l.size * l.size * l.c * l.out_h * l.out_w;
		} else if (l.type == CONNECTED) {
			ops += 2 * l.inputs * l.outputs;
		}
	}
	printf("Floating Point Operations: %ld\n", ops);
}

void partial(char *cfgfile, char *weightfile, char *outfile, int max) {
	gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights_upto(&net, weightfile, max);
	}
	*net.seen = 0;
	save_weights_upto(net, outfile, max);
}

void stacked(char *cfgfile, char *weightfile, char *outfile) {
	gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	net.seen = 0;
	save_weights_double(net, outfile);
}

#include "convolutional_layer.h"
void rescale_net(char *cfgfile, char *weightfile, char *outfile) {
	gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	int i;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		if (l.type == CONVOLUTIONAL) {
			rescale_filters(l, 2, -.5);
			break;
		}
	}
	save_weights(net, outfile);
}

void rgbgr_net(char *cfgfile, char *weightfile, char *outfile) {
	gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	int i;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		if (l.type == CONVOLUTIONAL) {
			rgbgr_filters(l);
			break;
		}
	}
	save_weights(net, outfile);
}

void normalize_net(char *cfgfile, char *weightfile, char *outfile) {
	gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	int i, j;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		if (l.type == CONVOLUTIONAL) {
			net.layers[i].batch_normalize = 1;
			net.layers[i].scales = calloc(l.n, sizeof(float));
			for (j = 0; j < l.n; ++j) {
				net.layers[i].scales[i] = 1;
			}
			net.layers[i].rolling_mean = calloc(l.n, sizeof(float));
			net.layers[i].rolling_variance = calloc(l.n, sizeof(float));
		}
	}
	save_weights(net, outfile);
}

void denormalize_net(char *cfgfile, char *weightfile, char *outfile) {
	gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	int i;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		if (l.type == CONVOLUTIONAL && l.batch_normalize) {
			denormalize_convolutional_layer(l);
			net.layers[i].batch_normalize = 0;
		}
		if (l.type == CONNECTED && l.batch_normalize) {
			denormalize_connected_layer(l);
			net.layers[i].batch_normalize = 0;
		}
		if (l.type == GRU && l.batch_normalize) {
			denormalize_connected_layer(*l.input_z_layer);
			denormalize_connected_layer(*l.input_r_layer);
			denormalize_connected_layer(*l.input_h_layer);
			denormalize_connected_layer(*l.state_z_layer);
			denormalize_connected_layer(*l.state_r_layer);
			denormalize_connected_layer(*l.state_h_layer);
			l.input_z_layer->batch_normalize = 0;
			l.input_r_layer->batch_normalize = 0;
			l.input_h_layer->batch_normalize = 0;
			l.state_z_layer->batch_normalize = 0;
			l.state_r_layer->batch_normalize = 0;
			l.state_h_layer->batch_normalize = 0;
			net.layers[i].batch_normalize = 0;
		}
	}
	save_weights(net, outfile);
}

void visualize(char *cfgfile, char *weightfile) {
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	visualize_network(net);
#ifdef OPENCV
	cvWaitKey(0);
#endif
}

void args_init_and_setnull(Args *arg) {
	arg->config_file = NULL;
	arg->execution_model = NULL;
	arg->config_file = NULL;
	arg->weights = NULL;
//	arg->input_data_path = NULL;
	arg->generate = NULL;
	arg->base_result_out = NULL;
	arg->cam_index = -1;
	arg->frame_skip = -1;
	arg->generate = 0;
	arg->img_list_path = NULL;
	arg->iterations = 1;
}
/**
 * return 1 if everything is ok, and 0 if not
 */
int check_args(const Args arg) {
	//check config_file
	if (access(arg.config_file, F_OK) == -1) {
		printf("Config file does not exist\n");
		return -1;
	}
	//check weights
	if (access(arg.weights, F_OK) == -1) {
		printf("Weights does not exist\n");
		return -1;
	}

	if (arg.iterations < 0 || arg.iterations > INT_MAX) {
		printf("Use a valid value for iterations\n");
		return -1;
	}

//	if (arg.input_data_path == NULL) {
//		printf("No input path set\n");
//		return -1;
//	}
	if (arg.generate_flag == 1 && arg.generate == NULL) {
		printf("Generate gold path not passed\n");
		return -1;
	}

	if (arg.img_list_path == NULL) {
		printf("Img list not passed\n");
		return -1;
	}
	if (arg.base_result_out == NULL) {
		printf("Base output not passed\n");
		return -1;
	}

	if (arg.gpu_index > 5 && arg.gpu_index < -2) {
		printf("gpu_index not passed\n");
		return -1;
	}

	if (arg.gpu_index > 5 && arg.gpu_index < -2) {
		printf("gpu_index not passed\n");
		return -1;
	}
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
//			"input_data_path = %s\n"
					"iterations = %ld\n"
					"generate = %s\n"
					"img_list_path = %s\n"
					"base_result_out = %s\n"
					"gpu_index = %d\n", arg.execution_type, arg.execution_model,
			arg.config_file, arg.weights, arg.iterations,
			((arg.generate_flag == 0) ? "not generating gold" : arg.generate),
			arg.img_list_path, arg.base_result_out, arg.gpu_index);
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
//			{ "input_data_path", 	required_argument, NULL, 'i' },
			{ "iterations", required_argument, NULL, 'n' }, //log data iterations
			{ "generate", required_argument, NULL, 'g' }, //generate gold
			{ "img_list_path", required_argument, NULL, 'l' }, //data path list input
			{ "base_result_out", required_argument, NULL, 'b' }, //result output
			{ "gpu_index", required_argument, NULL, 'x' }, //gpu index
			{ NULL, 0, NULL, 0 } };

	// loop over all of the options
	char ch;
	int ok = -1;
	int option_index = 0;
	to_parse->generate_flag = 0;
	while ((ch = getopt_long(argc, argv, "e:m:c:w:i:n:g:l:b:x:", long_options,
			&option_index)) != -1) {
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
//		case 'i': {
//			to_parse->input_data_path = optarg;
//			break;
//		}
		case 'n': {
			to_parse->iterations = atol(optarg);
			break;

		}
		case 'g': {
			to_parse->generate = optarg;
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

		}
		ok = 0;
	}
	return (ok || check_args(*to_parse));

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
			"-x --gpu_index = GPU index\n");
}

int main(int argc, char **argv) {
	//test_resize("data/bad.jpg");
	//test_box();
	//test_convolutional_layer();
	//I added the new parameters usage
	if (argc < 2) {
		//fprintf(stderr, "usage: %s <function>\n", argv[0]);
		usage(argv, "<yolo/valid/classifer>", "<function>");
		return 0;
	}

//	gpu_index = find_int_arg(argc, argv, "-i", 0);
//	if (find_arg(argc, argv, "-nogpu")) {
//		gpu_index = -1;
//	}

	//try to parse
	Args to_parse;
	args_init_and_setnull(&to_parse);
	if (parse_arguments(&to_parse, argc, argv) == 0) {
		//I'll do firsrt for yolo, next I dont know
		print_args(to_parse);
#ifndef GPU
		to_parse.gpu_index = -1;
#else
		if(to_parse.gpu_index >= 0) {
			cudaError_t status = cudaSetDevice(to_parse.gpu_index);
			check_error(status);
		}

#ifdef LOGS
		char test_info[90];
		snprintf(test_info, 90, "execution_type:%s execution_model:%s img_list_path:%s weights:%s config_file:%s iterations:%d", to_parse.execution_type
				, to_parse.execution_model, to_parse.img_list_path, to_parse.weights, to_parse.config_file, to_parse.iterations);
		if (!(to_parse.generate_flag)) start_log_file("cudaDarknet", test_info);
#endif

#endif



		if (strcmp(to_parse.execution_type, "yolo") == 0) {
			run_yolo(to_parse);
		}

		/*
		 if (0 == strcmp(argv[1], "imagenet")) {
		 run_imagenet(argc, argv);
		 } else if (0 == strcmp(argv[1], "average")) {
		 average(argc, argv);
		 } else if (0 == strcmp(argv[1], "yolo")) {
		 run_yolo(argc, argv);
		 } else if (0 == strcmp(argv[1], "cifar")) {
		 run_cifar(argc, argv);
		 } else if (0 == strcmp(argv[1], "go")) {
		 run_go(argc, argv);
		 } else if (0 == strcmp(argv[1], "rnn")) {
		 run_char_rnn(argc, argv);
		 } else if (0 == strcmp(argv[1], "vid")) {
		 run_vid_rnn(argc, argv);
		 } else if (0 == strcmp(argv[1], "coco")) {
		 run_coco(argc, argv);
		 } else if (0 == strcmp(argv[1], "classifier")) {
		 run_classifier(argc, argv);
		 } else if (0 == strcmp(argv[1], "art")) {
		 run_art(argc, argv);
		 } else if (0 == strcmp(argv[1], "tag")) {
		 run_tag(argc, argv);
		 } else if (0 == strcmp(argv[1], "compare")) {
		 run_compare(argc, argv);
		 } else if (0 == strcmp(argv[1], "dice")) {
		 run_dice(argc, argv);
		 } else if (0 == strcmp(argv[1], "writing")) {
		 run_writing(argc, argv);
		 } else if (0 == strcmp(argv[1], "3d")) {
		 composite_3d(argv[2], argv[3], argv[4]);
		 } else if (0 == strcmp(argv[1], "test")) {
		 test_resize(argv[2]);
		 } else if (0 == strcmp(argv[1], "captcha")) {
		 run_captcha(argc, argv);
		 } else if (0 == strcmp(argv[1], "nightmare")) {
		 run_nightmare(argc, argv);
		 } else if (0 == strcmp(argv[1], "change")) {
		 change_rate(argv[2], atof(argv[3]), (argc > 4) ? atof(argv[4]) : 0);
		 } else if (0 == strcmp(argv[1], "rgbgr")) {
		 rgbgr_net(argv[2], argv[3], argv[4]);
		 } else if (0 == strcmp(argv[1], "denormalize")) {
		 denormalize_net(argv[2], argv[3], argv[4]);
		 } else if (0 == strcmp(argv[1], "normalize")) {
		 normalize_net(argv[2], argv[3], argv[4]);
		 } else if (0 == strcmp(argv[1], "rescale")) {
		 rescale_net(argv[2], argv[3], argv[4]);
		 } else if (0 == strcmp(argv[1], "ops")) {
		 operations(argv[2]);
		 } else if (0 == strcmp(argv[1], "partial")) {
		 partial(argv[2], argv[3], argv[4], atoi(argv[5]));
		 } else if (0 == strcmp(argv[1], "average")) {
		 average(argc, argv);
		 } else if (0 == strcmp(argv[1], "stacked")) {
		 stacked(argv[2], argv[3], argv[4]);
		 } else if (0 == strcmp(argv[1], "visualize")) {
		 visualize(argv[2], (argc > 3) ? argv[3] : 0);
		 } else if (0 == strcmp(argv[1], "imtest")) {
		 test_resize(argv[2]);
		 } else {
		 fprintf(stderr, "Not an option: %s\n", argv[1]);
		 }
		 */
	} else {
		usage(argv, "<yolo/valid/classifer>", "<function>");
	}
#ifdef GPU && LOGS
	if (!(to_parse.generate_flag)) end_log_file();
#endif
	args_init_and_setnull(&to_parse);
	return 0;
}

