#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>


#include "common.h"

int rows, cols;
int* data;
int** wall;
int* result;
int pyramid_height;

extern int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols,
		int pyramid_height, int blockCols, int borderCols);

void init(int argc, char** argv) {
	if (argc == 4) {

		cols = atoi(argv[1]);

		rows = atoi(argv[2]);

		pyramid_height = atoi(argv[3]);
	} else {
		printf("Usage: dynproc row_len col_len pyramid_height\n");
		exit(0);
	}
	data = new int[rows * cols];

	wall = new int*[rows];

	for (int n = 0; n < rows; n++)

		wall[n] = data + cols * n;

	result = new int[cols];

	int seed = M_SEED;

	srand(seed);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {

			wall[i][j] = rand() % 10;
		}
	}

#ifdef BENCH_PRINT

	for (int i = 0; i < rows; i++) {

		for (int j = 0; j < cols; j++) {
			printf("%d ", wall[i][j]);
		}
		printf("\n");
	}

#endif
}

void fatal(char *s) {
	fprintf(stderr, "error: %s\n", s);

}


int main(int argc, char** argv) {
	int num_devices;
	cudaGetDeviceCount(&num_devices);
	if (num_devices > 1)
		cudaSetDevice(DEVICE);

	run(argc, argv);

	return EXIT_SUCCESS;
}

void run(int argc, char** argv) {
	init(argc, argv);

	/* --------------- pyramid parameters --------------- */
	int borderCols = (pyramid_height) * HALO;
	int smallBlockCol = BLOCK_SIZE - (pyramid_height) * HALO * 2;
	int blockCols = cols / smallBlockCol
			+ ((cols % smallBlockCol == 0) ? 0 : 1);

	printf(
			"pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
			pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols,
			smallBlockCol);

	int *gpuWall, *gpuResult[2];
	int size = rows * cols;

	cudaMalloc((void**) &gpuResult[0], sizeof(int) * cols);
	cudaMalloc((void**) &gpuResult[1], sizeof(int) * cols);
	cudaMemcpy(gpuResult[0], data, sizeof(int) * cols, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &gpuWall, sizeof(int) * (size - cols));
	cudaMemcpy(gpuWall, data + cols, sizeof(int) * (size - cols),
			cudaMemcpyHostToDevice);

	int final_ret = calc_path(gpuWall, gpuResult, rows, cols, pyramid_height,
			blockCols, borderCols);

	cudaMemcpy(result, gpuResult[final_ret], sizeof(int) * cols,
			cudaMemcpyDeviceToHost);

#ifdef BENCH_PRINT

	for (int i = 0; i < cols; i++)

		printf("%d ", data[i]);

	printf("\n");

	for (int i = 0; i < cols; i++)

		printf("%d ", result[i]);

	printf("\n");

#endif

	cudaFree(gpuWall);
	cudaFree(gpuResult[0]);
	cudaFree(gpuResult[1]);

	delete[] data;
	delete[] wall;
	delete[] result;

}

