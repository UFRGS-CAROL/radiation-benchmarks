/*
 * log_processing.cpp
 *
 *  Created on: 30/04/2017
 *      Author: fernando
 */

#include "log_processing.h"

#ifdef LOGS
#include "log_helper.h"
#endif

inline double mysecond() {
	struct timeval tp;
	struct timezone tzp;
	gettimeofday(&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

int prob_array_comparable_and_log(ProbArray gold, ProbArray pb,
		long plist_iteration) {
	char error_detail[1000];
	int i, j;
	unsigned long error_count = 0;
	float **gold_probs = gold.probs;
	float **pb_probs = pb.probs;
	//compare boxes

	for (i = 0; i < gold.total_size; ++i) {
		box tmp_gold = gold.boxes[i];
		box tmp_pb = pb.boxes[i];
		float x_diff = fabs(tmp_gold.x - tmp_pb.x);
		float y_diff = fabs(tmp_gold.y - tmp_pb.y);
		float w_diff = fabs(tmp_gold.w - tmp_pb.w);
		float h_diff = fabs(tmp_gold.h - tmp_pb.h);
		if (x_diff > THRESHOLD_ERROR || y_diff > THRESHOLD_ERROR
				|| w_diff > THRESHOLD_ERROR || h_diff > THRESHOLD_ERROR) {
			sprintf(error_detail, "image_list_position: [%ld] boxes: [%d] "
					" x_r: %1.16e x_e: %1.16e x_diff: %1.16e"
					" y_r: %1.16e y_e: %1.16e y_diff: %1.16e"
					" w_r: %1.16e w_e: %1.16e w_diff: %1.16e"
					" h_r: %1.16e h_e: %1.16e h_diff: %1.16e", plist_iteration,
					i, tmp_pb.x, tmp_gold.x, x_diff, tmp_pb.y, tmp_gold.y,
					y_diff, tmp_pb.w, tmp_gold.w, w_diff, tmp_pb.h, tmp_gold.h,
					h_diff);
#ifdef LOGS
			log_error_detail(error_detail);
#endif
//printf("passou box %d %d\n", i, j);
			for (j = 0; j < gold.classes; ++j) {
//printf("passou boxes %d %d\n", i, j);
				if (pb_probs[i][j] != 0 || gold_probs[i][j] != 0) {
					sprintf(error_detail,
							"image_list_position: [%ld] probs: [%d,%d] "
									" prob_r: %1.16e prob_e: %1.16e",
							plist_iteration, i, j, pb_probs[i][j],
							gold_probs[i][j]);
#ifdef LOGS
					log_error_detail(error_detail);
#endif
				}
			}
			error_count++;
		}
//printf("passou 1 %d %d\n", i, j);
	}
	//compare probs

	for (i = 0; i < gold.total_size; ++i) {
		int print_box = 0;
		for (j = 0; j < gold.classes; ++j) {

			float diff = fabs(gold_probs[i][j] - pb_probs[i][j]);
			if (diff > THRESHOLD_ERROR) {
//printf("passou 1 %d %d\n", i, j);
				sprintf(error_detail,
						"image_list_position: [%ld] probs: [%d,%d] "
								" prob_r: %1.16e prob_e: %1.16e",
						plist_iteration, i, j, pb_probs[i][j],
						gold_probs[i][j]);
				error_count++;
				print_box = 1;
#ifdef LOGS
				log_error_detail(error_detail);
#endif
			}
//printf("passou comp %d %d\n", i, j);
		}
		if (print_box == 1) {
			sprintf(error_detail, "image_list_position: [%ld] boxes: [%d] "
					" x_r: %1.16e x_e: %1.16e"
					" y_r: %1.16e y_e: %1.16e"
					" w_r: %1.16e w_e: %1.16e"
					" h_r: %1.16e h_e: %1.16e", plist_iteration, i,
					pb.boxes[i].x, gold.boxes[i].x, pb.boxes[i].y,
					gold.boxes[i].y, pb.boxes[i].w, gold.boxes[i].w,
					pb.boxes[i].h, gold.boxes[i].h);
#ifdef LOGS
			log_error_detail(error_detail);
#endif
		}

	}

//	printf("finish cout\n");

//	printf("finish cout after error count\n");
	return error_count;
}

unsigned long comparable_and_log(GoldPointers gold, GoldPointers current) {
	unsigned long error_count = 0;
	long i;
	for (i = 0; i < gold.plist_size; i++) {
//		printf("i on comp %d\n", i);
		error_count += prob_array_comparable_and_log(gold.pb_gold[i],
				current.pb_gold[i], i);
	}
	return error_count;
}


void saveLayer(network net, int iterator, int n) {
//	FILE* bin;
//	char* log_name;
//	char name[100];
//	char folderPath[100];
//	char a[5], b[8], c[8];
//	int i, j;
//
//	snprintf(b, 5, "%d", iterator); //iterator+n-1?
//	for (i = 0; i < 32; i++) {
//		snprintf(a, 3, "%d", i);
//#ifdef LOGS
//		log_name = get_log_file_name();
//#else
//		log_name = "gold";
//#endif
//		strcpy(name, log_name);
//		strcat(name, "_it_");
//		strcat(name, &b);
//		strcat(name, "_layer_");
//		strcat(name, &a);
//
//#ifdef LOGS
//		name[26] = 'd';
//		name[27] = 'a';
//		name[28] = 't';
//		name[29] = 'a';
//		name[30] = '/';
//#else
//
//		strcpy(folderPath, "/var/radiation-benchmarks/data/");
//		strcat(folderPath, name);
//		strcpy(name, folderPath);
//		printf("...saving %s \n", name);
//#endif
//
//		if ((bin = fopen(name, "wb")) == NULL) {
//			printf("LAYER: ERROR ON OPENING \n");
//		}
//
//		fwrite(layer_output[i], sizeof(float), net.layers[i].outputs, bin);
//
//		fclose(bin);
//		name[0] = '\0';
//
//	}
}

void compareLayer(layer l, int i) {
//	FILE* bin;
//	int error_count = 0;
//	char name[50];
//	char a[5];
//	snprintf(a, 3, "%d", i);
//	strcpy(name, "gold/layer");
//	strcat(name, &a);
//	strcat(name, ".bin");
//	//printf("1111\n");
//	if ((bin = fopen(name, "r")) == NULL) {
//		printf("ERROR ON OPENING \n");
//	}
//
//	float * r = (float*) calloc(l.outputs, sizeof(float));
//	cudaMemcpy(r, l.output_gpu, l.outputs * sizeof(float),
//			cudaMemcpyDeviceToHost);
//	float * s = (float*) calloc(l.outputs, sizeof(float));
//	fread(s, sizeof(float), l.outputs, bin);
//	int j;
//	for (j = 0; j < l.outputs; j++) {
//		if (s[j] != r[j]) {
//			error_count++;
//		}
//	}
//
//	fclose(bin);
}
