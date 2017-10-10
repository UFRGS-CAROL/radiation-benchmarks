/*
 * image.h
 *
 *  Created on: Oct 10, 2017
 *      Author: carol
 */

#ifndef IMAGE_H_
#define IMAGE_H_


typedef struct{
	fp *data;
	unsigned width;
	unsigned height;
	char magic_number[2];
	unsigned max_gray_value;

}PGMImage;


#endif /* IMAGE_H_ */
