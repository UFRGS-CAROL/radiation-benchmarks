#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include <vector>

#include "common.h"
#include "Parameters.h"
#include "device_vector.h"

template<typename T>
using matrix = std::vector<std::vector<T>>;

template<typename T>
using vector = std::vector<T>;

extern int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols,
		int pyramid_height, int blockCols, int borderCols);

void init(vector<int*>& wall, vector<int>& data, vector<int>& result,
		int pyramid_height, int rows, int cols) {

//	data = new int[rows * cols];	data.resize(rows * cols);
	wall.resize(rows);

	for (int n = 0; n < rows; n++) {
		wall[n] = data.data() + cols * n;
	}

//	result = new int[cols];
	result.resize(cols);

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

void run(int argc, char** argv) {
	int pyramid_height, cols, rows;
	if (argc == 4) {
		cols = atoi(argv[1]);
		rows = atoi(argv[2]);
		pyramid_height = atoi(argv[3]);
	} else {
		throw_line("Usage: dynproc row_len col_len pyramid_height\n");
	}

	vector<int*> wall;
	vector<int> data;
	vector<int> result;

	init(wall, data, result, pyramid_height, rows, cols);

	/* --------------- pyramid parameters --------------- */
	int borderCols = (pyramid_height) * HALO;
	int smallBlockCol = BLOCK_SIZE - (pyramid_height) * HALO * 2;
	int blockCols = cols / smallBlockCol
			+ ((cols % smallBlockCol == 0) ? 0 : 1);

	printf(
			"pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
			pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols,
			smallBlockCol);

	//int *gpuWall, *gpuResult[2];
	rad::DeviceVector<int> gpuWall;
	rad::DeviceVector<int> gpuResult[2];

	int size = rows * cols;
//	cudaMalloc((void**) &gpuResult[0], sizeof(int) * cols);
//	cudaMalloc((void**) &gpuResult[1], sizeof(int) * cols);
	gpuResult[0].resize(cols);
	gpuResult[1].resize(cols);

//	cudaMemcpy(gpuResult[0], data, sizeof(int) * cols, cudaMemcpyHostToDevice);
	gpuResult[0].fill_n(data.begin(), cols);

//	cudaMalloc((void**) &gpuWall, sizeof(int) * (size - cols));
	gpuWall.resize(size - cols);
//	cudaMemcpy(gpuWall, data + cols, sizeof(int) * (size - cols),
//			cudaMemcpyHostToDevice);

	gpuWall.fill_n(data.begin() + cols, (size - cols));

	int *gpuResult_ptr[2] = { gpuResult[0].data(), gpuResult[1].data() };

	int final_ret = calc_path(gpuWall.data(), gpuResult_ptr, rows, cols,
			pyramid_height, blockCols, borderCols);

//	cudaMemcpy(result, gpuResult[final_ret], sizeof(int) * cols,
//			cudaMemcpyDeviceToHost);
	gpuResult[final_ret].to_vector(result);

#ifdef BENCH_PRINT

	for (int i = 0; i < cols; i++)

		printf("%d ", data[i]);

	printf("\n");

	for (int i = 0; i < cols; i++)

		printf("%d ", result[i]);

	printf("\n");

#endif

//	cudaFree(gpuWall);
//	cudaFree(gpuResult[0]);
//	cudaFree(gpuResult[1]);
//
//	delete[] data;
//	delete[] wall;
//	delete[] result;

}

