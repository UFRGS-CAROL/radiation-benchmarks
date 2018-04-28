/*
 * yolo.h
 *
 *  Created on: 25/09/2016
 *      Author: fernando
 */

#include "args.h"

#ifndef YOLO_H_
#define YOLO_H_



void run_voxel(int argc, char **argv);
void run_yolo_rad(Args args);
void run_yolo(int argc, char **argv);
void run_detector(int argc, char **argv);
void run_coco(int argc, char **argv);
void run_writing(int argc, char **argv);
void run_captcha(int argc, char **argv);
void run_nightmare(int argc, char **argv);
void run_dice(int argc, char **argv);
void run_compare(int argc, char **argv);
void run_classifier(int argc, char **argv);
void run_char_rnn(int argc, char **argv);
void run_vid_rnn(int argc, char **argv);
void run_tag(int argc, char **argv);
void run_cifar(int argc, char **argv);
void run_go(int argc, char **argv);
void run_art(int argc, char **argv);
void run_super(int argc, char **argv);

float* layer_output[32];

#endif /* YOLO_H_ */
