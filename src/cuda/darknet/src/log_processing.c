/*
 * log_processing.c
 *
 *  Created on: 25/09/2016
 *      Author: fernando
 */

/**
 * only to save gold
 * first need set classes and total_size parameters
 * not NULL = OK, NULL shit happened
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


ProbArray new_prob_array(int classes, int total_size) {
	int i;
	ProbArray pb;
	pb.classes = classes;
	pb.total_size = total_size;
	pb.probs = calloc(pb.total_size, sizeof(float*));
	pb.boxes = calloc(pb.total_size, sizeof(box));
	if (pb.probs == NULL || pb.boxes == NULL) {
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
	if (pb->boxes != NULL)
		free(pb->boxes);
	pb->classes = pb->total_size = 0;
	pb = NULL;
}

GoldPointers new_gold_pointers(int classes, int total_size,
		const int plist_size, char *file_path, char *open_mode) {
	GoldPointers gp;
	gp.plist_size = plist_size;
	gp.pb_gold = calloc(plist_size, sizeof(ProbArray));
	if (gp.pb_gold == NULL) {
		error("ERROR ON ALLOCATING ProbArray list\n");
	}
	int i;
	for (i = 0; i < plist_size; i++) {
		gp.pb_gold[i] = new_prob_array(classes, total_size);
	}

	if (strcmp(file_path, "not_open") == 0) {
		gp.has_file = 0;
		return gp;
	}
	gp.has_file = 1;
	if ((gp.gold = fopen(file_path, open_mode)) == NULL) {
		char buff[1000];
		sprintf(buff, "ERROR ON OPENING %s\n", file_path);
		error(buff);
	}

	return gp;
}

//don't mess up with the memory in C, shit gonna happens
void free_gold_pointers(GoldPointers *gp) {
	if (gp->has_file)
		fclose(gp->gold);
//	free(gp->boxes);
//	free(gp->boxes_gold);
	int i;
	for (i = 0; i < gp->plist_size; i++) {
		free_prob_array(&gp->pb_gold[i]);
	}
	free(gp->pb_gold);
//	free_prob_array(&gp->pb);
}

void prob_array_serialize(ProbArray pb, FILE *fp) {
	//writing boxes
	fwrite(pb.boxes, sizeof(box), pb.total_size, fp);
	if (ferror(fp))
		error("cannot write to file boxes array\n");
	clearerr(fp);
	//writing probs matrix
	int i;
	for (i = 0; i < pb.total_size; i++) {
		fwrite(pb.probs[i], sizeof(float), pb.classes, fp);
		if (ferror(fp))
			error("cannot write to file prob array");
		clearerr(fp);
	}
}

/**
 * The output will be stored in this order
 long plist_size;
 long classes;
 long total_size;
 for(<plist_size times>){
 -----pb_gold.boxes
 -----pb_gold.probs
 }
 */
void gold_pointers_serialize(GoldPointers gp) {
	if (!gp.has_file)
		return;
	fwrite(&gp.plist_size, sizeof(long), 1, gp.gold);
	if (ferror(gp.gold))
		error("cannot write to file plist_size\n");
	clearerr(gp.gold);

	fwrite(&gp.pb_gold[0].classes, sizeof(long), 1, gp.gold);
	if (ferror(gp.gold))
		error("cannot write to file classes\n");
	clearerr(gp.gold);

	fwrite(&gp.pb_gold[0].total_size, sizeof(long), 1, gp.gold);
	if (ferror(gp.gold))
		error("cannot write to file total_size\n");
	clearerr(gp.gold);

	int i;
	for (i = 0; i < gp.plist_size; i++) {
		prob_array_serialize(gp.pb_gold[i], gp.gold);
	}
	if (ferror(gp.gold))
		error("cannot write to file total_size\n");
	clearerr(gp.gold);
}

/**
 * This function assumes that pb is already allocated on memory
 */
ProbArray read_prob_array(FILE *fp, ProbArray pb) {
	//writing boxes
	if (ferror(fp)
			|| fread(pb.boxes, sizeof(box), pb.total_size, fp) < pb.total_size)
		error("cannot write to file boxes array\n");
	clearerr(fp);
	//writing probs matrix
	int i;
	for (i = 0; i < pb.total_size; i++) {
		if (ferror(fp)
				|| fread(pb.probs[i], sizeof(float), pb.classes, fp)
						< pb.classes)
			error("cannot write to file prob array");
		clearerr(fp);
	}
	return pb; //return itself?
}
/**
 *  the input must be read in this order
 long plist_size;
 long classes;
 long total_size;
 for(<plist_size times>){
 -----pb_gold.boxes
 -----pb_gold.probs
 }
 */
void read_yolo_gold(GoldPointers *gp) {
//	printf("passou dos gold ptr %d\n", gp->has_file);
	if (ferror(gp->gold)
			|| fread(&gp->plist_size, sizeof(long), 1, gp->gold) != 1)
		error("cannot read to file plist_size\n");
	clearerr(gp->gold);

	if (ferror(gp->gold)
			|| fread(&gp->pb_gold[0].classes, sizeof(long), 1, gp->gold) != 1)
		error("cannot read to file classes\n");
	clearerr(gp->gold);

	if (ferror(gp->gold)
			|| fread(&gp->pb_gold[0].total_size, sizeof(long), 1, gp->gold)
					!= 1)
		error("cannot read to file total_size\n");
	clearerr(gp->gold);

	int i;
	for (i = 0; i < gp->plist_size; i++) {
		gp->pb_gold[i] = read_prob_array(gp->gold, gp->pb_gold[i]);
	}
	if (ferror(gp->gold))
		error("cannot write to file total_size\n");
	clearerr(gp->gold);
}

/**
 * if some error happens the error_count will be != 0
 */
int prob_array_comparable_and_log(ProbArray gold, ProbArray pb, long plist_iteration) {
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
				if (pb_probs[i][j] != 0 || gold_probs[i][j] != 0)
				{
					sprintf(error_detail, "image_list_position: [%ld] probs: [%d,%d] "
									" prob_r: %1.16e prob_e: %1.16e", plist_iteration,
							i, j, pb_probs[i][j],
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
		if (print_box == 1)
		{
			sprintf(error_detail, "image_list_position: [%ld] boxes: [%d] "
					" x_r: %1.16e x_e: %1.16e"
					" y_r: %1.16e y_e: %1.16e"
					" w_r: %1.16e w_e: %1.16e"
					" h_r: %1.16e h_e: %1.16e", plist_iteration,
					i, pb.boxes[i].x, gold.boxes[i].x, pb.boxes[i].y, gold.boxes[i].y,
					pb.boxes[i].w, gold.boxes[i].w, pb.boxes[i].h, gold.boxes[i].h);
#ifdef LOGS
			log_error_detail(error_detail);
#endif
		}

	}

//	printf("finish cout\n");

//	printf("finish cout after error count\n");
	return error_count;
}

unsigned long comparable_and_log(GoldPointers gold, GoldPointers current){
	unsigned long error_count = 0;
	long i;
	for(i = 0; i < gold.plist_size; i++){
//		printf("i on comp %d\n", i);
		error_count += prob_array_comparable_and_log(gold.pb_gold[i], current.pb_gold[i], i);
	}
	return error_count;
}

void clear_vectors(GoldPointers *gp){
	int i;
	for(i = 0; i < gp->plist_size; i++){
		ProbArray tmp = gp->pb_gold[i];
		memset(tmp.boxes, 0, sizeof(box) * tmp.total_size);
		int j;
		for(j = 0; j < tmp.total_size; j++){
			memset(tmp.probs[j], 0, sizeof(float) * tmp.classes);
		}
	}
}

void saveLayer(network net, int iterator, int n)
{
	FILE* bin;
	char* log_name;
	char name[100];
	char a[5], b[8], c[8];
	int i, j;
	
	snprintf(b, 5,"%d",iterator+n-1);
	for (i = 0; i < 32; i++)
	{
		snprintf(a, 3,"%d",i);
#ifdef LOGS
		log_name = get_log_file_name();
#else
		log_name = "standard_name";
#endif
		strcpy(name, log_name);
		strcat(name, "_it_");
		strcat(name, &b);
		strcat(name, "_layer_");
		strcat(name, &a);

		for (j = strlen(name)+2; j >= 31; j--)
		{
			name[j] = name [j-1];
		}
		name[26] = 'd';
		name[27] = 'a';
		name[28] = 't';
		name[29] = 'a';
		name[30] = '/';
		//printf("%s\n\n\n", name);

		if ((bin = fopen(name, "wb")) == NULL) {
			printf("ERROR ON OPENING \n");
		}
		//printf("1112\n");
		//printf("1113\n");
		//printf("%s\n", name);
		//printf("%f\n", l.output_gpu[0]);
		//printf("%f\n", l.output_gpu[1]);
		//printf("%f\n", l.output_gpu[2]);
		//printf("%d\n", l.batch);

		fwrite(layer_output[i], sizeof(float), net.layers[i].outputs, bin);
		//printf("1114\n");
		fclose(bin);
		name[0] = '\0';
		//Lucas always free the memory
		//free(layer_output[i]);
	}
}

void compareLayer(layer l, int i)
{
	FILE* bin;
	int error_count = 0;
	char name[50];
	char a[5];
	snprintf(a, 3,"%d",i);
	strcpy(name, "gold/layer");
	strcat(name, &a);
	strcat(name, ".bin");	
	//printf("1111\n");
	if ((bin = fopen(name, "r")) == NULL) {
		printf("ERROR ON OPENING \n");
	}
	//printf("1112\n");
	//printf("1113\n");
	//printf("%d\n", l.outputs);
	//printf("%f\n", l.output_gpu[0]);
	//printf("%f\n", l.output_gpu[1]);
	//printf("%f\n", l.output_gpu[2]);
	//printf("%d\n", l.batch);
	float * r = (float*)calloc(l.outputs, sizeof(float));
    cudaMemcpy ( r, l.output_gpu, l.outputs*sizeof(float), cudaMemcpyDeviceToHost);
    float * s = (float*)calloc(l.outputs, sizeof(float));
	fread(s, sizeof(float), l.outputs, bin);
	int j;
	for (j = 0; j < l.outputs; j++)
	{
		if (s[j] != r[j])
		{
			error_count++;
		}
	}
	//printf("error count:%d\n", error_count);
	fclose(bin);
}	
