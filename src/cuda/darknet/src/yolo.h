/*
 * yolo.h
 *
 *  Created on: Sep 21, 2016
 *      Author: carol
 */

#ifndef YOLO_H_
#define YOLO_H_


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
//	char *input_data_path;
	long int iterations;
	char *generate;
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
} Args;

extern void run_imagenet(int argc, char **argv);
//extern void run_yolo(int argc, char **argv);
extern void run_yolo(const Args args);
extern void run_coco(int argc, char **argv);
extern void run_writing(int argc, char **argv);
extern void run_captcha(int argc, char **argv);
extern void run_nightmare(int argc, char **argv);
extern void run_dice(int argc, char **argv);
extern void run_compare(int argc, char **argv);
extern void run_classifier(int argc, char **argv);
extern void run_char_rnn(int argc, char **argv);
extern void run_vid_rnn(int argc, char **argv);
extern void run_tag(int argc, char **argv);
extern void run_cifar(int argc, char **argv);
extern void run_go(int argc, char **argv);
extern void run_art(int argc, char **argv);

#endif /* YOLO_H_ */
