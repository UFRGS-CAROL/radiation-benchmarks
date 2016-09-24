/*
 * log_processing.h
 *
 *  Created on: 22/09/2016
 *      Author: fernando
 */

#ifndef LOG_PROCESSING_H_
#define LOG_PROCESSING_H_

typedef struct prob_arry {
	float **probs;
	long classes;
	long total_size;
} ProbArray;

//to store all gold filenames
typedef struct gold_pointers {
	box *boxes;
	box *boxes_gold;
	ProbArray pb_array;
	ProbArray *pb_gold;
	long plist_size;
	FILE* gold;
} GoldPointers;

/**
 * only to save gold
 * first need set classes and total_size parameters
 * not NULL = OK, NULL shit happened
 */
ProbArray new_prob_array(int classes, int total_size) {
	int i;
	ProbArray pb;
	pb.classes = classes;
	pb.total_size = total_size;
	pb.probs = calloc(pb.total_size, sizeof(float*));
	if (pb.probs == NULL) {
		error("ERROR ON ALLOCATING ProbArray\n");
	}
	for (i = 0; i < pb.total_size; i++) {
		pb.probs[i] = calloc(pb.classes, sizeof(float));
		if (pb.probs[i] == NULL)
			error("ERROR ON ALLOCATING ProbArray\n");
	}
	return pb;
}

void free_prob_array(ProbArray *pb) {
	int i;

	for (i = 0; i < pb->total_size; i++) {
		if (pb->probs[i] != NULL)
			free(pb->probs[i]);
	}
	if (pb->probs != NULL)
		free(pb->probs);
	pb->classes = pb->total_size = 0;
	pb = NULL;
}

GoldPointers new_gold_pointers(ProbArray pb, const int plist_size, char *file_path, char *open_mode){
	GoldPointers gp;
	gp.pb_array = pb;
	long int total_size = pb.total_size;
	long int classes = pb.classes;
	gp.plist_size = plist_size;
	//it is a normal box array
	gp.boxes = calloc(total_size, sizeof(box));
	//gold box array
	gp.boxes_gold = calloc(total_size * plist_size, sizeof(box));
	if(gp.boxes == NULL || gp.boxes_gold == NULL){
		error("ERROR ON ALLOCATING boxes array\n");
	}

	gp.pb_gold = calloc(plist_size, sizeof(ProbArray));
	if(gp.pb_gold == NULL){
		error("ERROR ON ALLOCATING ProbArray list\n");
	}
	int i;
	for(i = 0; i < plist_size; i++){
		gp.pb_gold[i] = new_prob_array(classes, total_size);
	}

	if((gp.gold = fopen(file_path, open_mode)) == NULL){
		char buff[1000];
		sprintf(buff,"ERROR ON OPENING %s\n", buff);
		error(buff);
	}
	return gp;
}

//don't mess up with the memory in C, shit gonna happens
void free_gold_pointers(GoldPointers *gp){
	fclose(gp->gold);
	free(gp->boxes);
	free(gp->boxes_gold);
	int i;
	for(i = 0; i < gp->plist_size;i++){
		free_prob_array(&gp->pb_gold[i]);
	}
	free(gp->pb_gold);
	free_prob_array(&gp->pb_array);
}

void cp_to_gold(ProbArray pb, GoldPointers *gp, int iterator){

}

///**
// * The output will be stored in this order
// * boxes
// * probs
// */
//void write_yolo_gold(GoldPointers *gp) {
//	//write all boxes
//	int i, j;
////	for (i = 0; i < gp->plist_size; i++) {
//	fwrite(gp->boxes, sizeof(box), gp->total_size * gp->plist_size, gp->gold);
////	}
//	//write probs
//	for (i = 0; i < gp->plist_size; i++) {
//		for (j = 0; j < gp->total_size; j++) {
//			fwrite(gp->probs[i], sizeof(float), gp->classes * gp->total_size,
//					gp->gold);
//		}
//	}
//}
//
//void read_yolo_gold(GoldPointers *gp) {
//	//read all boxes
//	int i, j;
////	for (i = 0; i < gp->plist_size; i++) {
//	fread(gp->boxes, sizeof(box), gp->total * gp->plist_size, gp->gold);
////	}
//	for (i = 0; i < gp->plist_size; i++) {
//		for (j = 0; j < gp->total_size; j++) {
//			fread(gp->probs[i], sizeof(float), gp->classes * gp->total_size,
//					gp->gold);
//		}
//	}
//}
//
//int allocate_gold_memory(GoldPointers *gp) {
//	int z, j;
//	//plist_size * total_size
//	gp->boxes = calloc(gp->plist_size, sizeof(box*)); //= calloc(side * side * l.n, sizeof(box));
//	gp->boxes_gold = calloc(gp->plist_size, sizeof(box*));
//	gp->probs = calloc(gp->plist_size, sizeof(float**)); // = calloc(side * side * l.n, sizeof(float *));
//	gp->probs_gold = calloc(gp->plist_size, sizeof(float**));
//
//	if (gp->boxes == NULL || gp->boxes_gold == NULL || gp->probs == NULL
//			|| gp->probs_gold == NULL) {
//		return -1;
//	}
//
//	//man this shit sucks, I love C++ and JAVA
//	for (z = 0; z < gp->plist_size; z++) {
//		//probabilities
//		gp->probs[z] = calloc(gp->total_size, sizeof(float*));
//		gp->probs_gold[z] = calloc(gp->total_size, sizeof(float*));
//		//boxes
//		gp->boxes[z] = calloc(gp->total_size, sizeof(box));
//		gp->boxes_gold[z] = calloc(gp->total_size, sizeof(box));
//		if (gp->probs[z] == NULL || gp->probs_gold[z] == NULL
//				|| gp->boxes[z] == NULL || gp->boxes_gold[z] == NULL) {
//			return -1;
//		}
//		for (j = 0; j < gp->total_size; j++) {
//			gp->probs[z][j] = calloc(gp->classes, sizeof(float));
//			gp->probs_gold[z][j] = calloc(gp->classes, sizeof(float));
//			if (gp->probs[z][j] == NULL || gp->probs_gold[z][j] == NULL) {
//				return -1;
//			}
//		}
//	}
//	return 0;
//}
//
//void free_gold_memory(GoldPointers *gp) {
//	int j;
//	//closing the gold input/output
//	fclose(gp->gold);
//	int z = 0;
//	for (z = 0; z < gp->plist_size; z++) {
//		for (j = 0; j < gp->total_size; j++) {
//			free(gp->probs[z][j]);
//			free(gp->probs_gold[z][j]);
//		}
//		free(gp->probs[z]);
//		free(gp->probs_gold[z]);
//		free(gp->boxes[z]);
//		free(gp->boxes_gold[z]);
//	}
//	free(gp->probs);
//	free(gp->probs_gold);
//	free(gp->boxes);
//	free(gp->boxes_gold);
//}

#endif /* LOG_PROCESSING_H_ */
