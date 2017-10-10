//====================================================================================================100
//====================================================================================================100
//	INCLUDE/DEFINE
//====================================================================================================100
//====================================================================================================100

// #include <stdlib.h>
// #include <string.h>
#include <stdio.h>
// #include <math.h>
// #include <time.h>
// #include <sys/types.h>
// #include <dirent.h>
// #include <errno.h>

#include "image.h"
//====================================================================================================100
//====================================================================================================100
//	WRITE FUNCTION
//====================================================================================================100
//====================================================================================================100

void write_graphics(char* filename, fp* input, int data_rows, int data_cols,
		int major, int data_range) {

	//================================================================================80
	//	VARIABLES
	//================================================================================80

	FILE* fid;
	int i, j;

	//================================================================================80
	//	CREATE/OPEN FILE FOR WRITING
	//================================================================================80

	fid = fopen(filename, "w");
	if (fid == NULL) {
		printf("The file was not created/opened for writing\n");
		return;
	}

	//================================================================================80
	//	WRITE PGM FILE HEADER
	//================================================================================80

	fprintf(fid, "P2\n");
	fprintf(fid, "%d %d\n", data_cols, data_rows);
	fprintf(fid, "%d\n", data_range);

	//================================================================================80
	//	WRITE VALUES TO THE FILE
	//================================================================================80

	// if matrix is saved row major in memory (C)
	if (major == 0) {
		for (i = 0; i < data_rows; i++) {
			for (j = 0; j < data_cols; j++) {
				fprintf(fid, "%d ", (int) input[i * data_cols + j]);
			}
			fprintf(fid, "\n");
		}
	}
	// if matrix is saved column major in memory (MATLAB)
	else {
		for (i = 0; i < data_rows; i++) {
			for (j = 0; j < data_cols; j++) {
				fprintf(fid, "%d ", (int) input[j * data_rows + i]);
			}
			fprintf(fid, "\n");
		}
	}

	//================================================================================80
	//	CLOSE FILE
	//================================================================================80

	fclose(fid);

}

//====================================================================================================100
//====================================================================================================100
//	READ FUNCTION
//====================================================================================================100
//====================================================================================================100

void read_graphics(char* filename, fp* input, int data_rows, int data_cols,
		int major) {

	//================================================================================80
	//	VARIABLES
	//================================================================================80

	FILE* fid;
	int i, j;
	char c;
	int temp;

	//================================================================================80
	//	OPEN FILE FOR READING
	//================================================================================80

	fid = fopen(filename, "r");
	if (fid == NULL) {
		printf("The file was not opened for reading\n");
		return;
	}

	//================================================================================80
	//	SKIP PGM FILE HEADER
	//================================================================================80

	i = 0;
	while (i < 3) {
		c = fgetc(fid);
		if (c == '\n') {
			i = i + 1;
		}
	};

	//================================================================================80
	//	READ VALUES FROM THE FILE
	//================================================================================80

	if (major == 0) {			// if matrix is saved row major in memory (C)
		for (i = 0; i < data_rows; i++) {
			for (j = 0; j < data_cols; j++) {
				fscanf(fid, "%d", &temp);
				input[i * data_cols + j] = (fp) temp;
			}
		}
	} else {			// if matrix is saved column major in memory (MATLAB)
		for (i = 0; i < data_rows; i++) {
			for (j = 0; j < data_cols; j++) {
				fscanf(fid, "%d", &temp);
				input[j * data_rows + i] = (fp) temp;
			}
		}
	}

	//================================================================================80
	//	CLOSE FILE
	//================================================================================80

	fclose(fid);

}

void read_pgm_image(char* filename, PGMImage *img, int major) {

	//================================================================================80
	//	VARIABLES
	//================================================================================80

	FILE* fid;
	int i, j;
	char c;
	int temp;

	//================================================================================80
	//	OPEN FILE FOR READING
	//================================================================================80

	fid = fopen(filename, "r");
	if (fid == NULL) {
		printf("The file was not opened for reading\n");
		return;
	}

	//================================================================================80
	//	SKIP PGM FILE HEADER
	//================================================================================80

	i = 0;
	char line[256];
	for (i = 0; i <= 3; i++) {
		fgets(line, sizeof(line), fid);

		if (i == 0) {
			img->magic_number[0] = line[0];
			img->magic_number[1] = line[1];
		}
		if (i == 1) {

			sscanf(line, "%d %d", &img->width, &img->height);
		}
		if (i == 2) {
			img->max_gray_value = atoi(line);
		}

	}

	img->data = (fp*) calloc(img->width * img->height, sizeof(fp));
	//================================================================================80
	//	READ VALUES FROM THE FILE
	//================================================================================80

	if (major == 0) {			// if matrix is saved row major in memory (C)
		for (i = 0; i < img->height; i++) {
			for (j = 0; j < img->width; j++) {
				fscanf(fid, "%d", &temp);
				img->data[i * img->width + j] = (fp) temp;
			}
		}
	} else {			// if matrix is saved column major in memory (MATLAB)
		for (i = 0; i < img->height; i++) {
			for (j = 0; j < img->width; j++) {
				fscanf(fid, "%d", &temp);
				img->data[j * img->height + i] = (fp) temp;
			}
		}
	}

	//================================================================================80
	//	CLOSE FILE
	//================================================================================80

	fclose(fid);

}

PGMImage inline make_pgm_img(fp* data, int h, int w, unsigned max_value, char *magic){
	PGMImage img;
	img.data = data;
	img.height = h;
	img.width = w;
	img.max_gray_value = max_value;
	img.magic_number[0] = magic[0];
	img.magic_number[1] = magic[1];
	return img;
}


void free_img(PGMImage *img){
	if (img->data)
		free(img->data);

}

void malloc_img_data(PGMImage *img){
	img->data = (fp*) calloc(img->width * img->height, sizeof(fp));
}

void print_image(PGMImage *img){
	printf("Width %d\n"
			"Height %d\nMagic number %s\n"
			"Max value %d\n", img->width, img->height, img->magic_number, img->max_gray_value);
}

