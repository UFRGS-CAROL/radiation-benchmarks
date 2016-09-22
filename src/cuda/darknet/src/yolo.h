/*
 * yolo.h
 *
 *  Created on: Sep 21, 2016
 *      Author: carol
 */

#ifndef YOLO_H_
#define YOLO_H_

#include "args.h"

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
