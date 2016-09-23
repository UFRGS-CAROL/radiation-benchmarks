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
 * total classes w h
 * id
 * boxes
 * probs
 */
void write_yolo_gold(FILE *fps, char *id, box *boxes, float **probs, int total,
		int classes, int w, int h) {

	fprintf(fps, "%d %d %d %d\n", total, classes, w, h);
	int i, j;
	//saving id
	fprintf(fps, "%s\n", id);
	//saving all boxes
	for (i = 0; i < total; i++) {
		fprintf(fps, "%f %f %f %f\n", boxes[i].x, boxes[i].y, boxes[i].w,
				boxes[i].h); //    float x, y, w, h;
	}

	//saving all probs
	for (i = 0; i < total; i++) {
		for (j = 0; j < classes; j++) {
			fprintf(fps, "%f", probs[i][j]);
			if ((j + 1) != classes)
				fprintf(fps, " ");
		}
		fprintf(fps, "\n");
	}

}

void read_yolo_gold(FILE *fps, char **id, box *boxes, float **probs, int *total,
		int *classes, int *w, int *h) {
	//reading parameters
	fscanf(fps, "%d %d %d %d", total, classes, w, h);
	int it = 0;
	while (!feof(fps)) {
		printf("%d\n", it++);
		//reading id
		fscanf(fps, "%s", id);
		int i, j;
		//reading boxes
		for (i = 0; i < *total; i++) {
			fscanf(fps, "%f %f %f %f\n", &boxes[i].x, &boxes[i].y, &boxes[i].w,
					&boxes[i].h);
		}

		//reading probs
		for (i = 0; i < *total; i++) {
			for (j = 0; j < *classes; j++) {
				fscanf(fps, "%f", &probs[i][j]);
			}
		}
	}
}

#endif /* LOG_PROCESSING_H_ */
