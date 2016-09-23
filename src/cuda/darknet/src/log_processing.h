/*
 * log_processing.h
 *
 *  Created on: 22/09/2016
 *      Author: fernando
 */

#ifndef LOG_PROCESSING_H_
#define LOG_PROCESSING_H_

//to store all gold filenames

/**
 * The output will be stored in this order
  * boxes
 * probs
 */
void write_yolo_gold(FILE *fps, box *boxes, int boxes_size, float ***probs, int plist_size, int total_size, int classes) {
	//write all boxes
	fwrite(boxes, sizeof(box),  boxes_size, fps);

	int i, j;
	//write probs
	for(i = 0; i < plist_size; i++){
		for(j = 0; j < total_size; j++){
			fwrite(probs[i][j], sizeof(float), classes, fps);
		}
	}
}

void read_yolo_gold(FILE *fps, box *boxes, int boxes_size, float ***probs, int plist_size, int total_size, int classes){
	//read all boxes
	fread(boxes, sizeof(box), boxes_size, fps);

	int i, j;
	for(i = 0; i < plist_size; i++){
		for(j = 0; j < total_size; j++){
			fread(probs[i][j], sizeof(float), classes, fps);
		}
	}
}

#endif /* LOG_PROCESSING_H_ */
