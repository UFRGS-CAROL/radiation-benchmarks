

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "common.h"

//#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>
//#include <assert.h>

#include <vector>
#include <fstream>
#include <string>

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
		int pyramid_height, int rows, int cols,
		std::string output_file) {

//	data = new int[rows * cols];
	data.resize(rows * cols);
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

	std::ofstream of(output_file);
	if (of.good()) {
		for (int i = 0; i < rows; i++) {

			for (int j = 0; j < cols; j++) {
//			printf("%d ", wall[i][j]);
				of << wall[i][j] << " ";
			}
//		printf("\n");
			of << std::endl;
		}
	} else {
		throw_line("Could not open file " + output_file);
	}
}

void run(int argc, char** argv) {
	Parameters parameters(argc, argv);

	vector<int*> wall;
	vector<int> data;
	vector<int> result;
	std::string output_file = "result.txt";

	init(wall, data, result, parameters.pyramid_height, parameters.rows,
			parameters.cols, output_file);

	/* --------------- pyramid parameters --------------- */
	int borderCols = (parameters.pyramid_height) * HALO;
	int smallBlockCol = BLOCK_SIZE - (parameters.pyramid_height) * HALO * 2;
	int blockCols = parameters.cols / smallBlockCol
			+ ((parameters.cols % smallBlockCol == 0) ? 0 : 1);

	printf(
			"pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
			parameters.pyramid_height, parameters.cols, borderCols, BLOCK_SIZE,
			blockCols, smallBlockCol);

	//int *gpuWall, *gpuResult[2];
	rad::DeviceVector<int> gpuWall;
	rad::DeviceVector<int> gpuResult[2];

	int size = parameters.rows * parameters.cols;
//	cudaMalloc((void**) &gpuResult[0], sizeof(int) * cols);
//	cudaMalloc((void**) &gpuResult[1], sizeof(int) * cols);
	gpuResult[0].resize(parameters.cols);
	gpuResult[1].resize(parameters.cols);

//	cudaMemcpy(gpuResult[0], data, sizeof(int) * cols, cudaMemcpyHostToDevice);
	gpuResult[0].fill_n(data.begin(), parameters.cols);

//	cudaMalloc((void**) &gpuWall, sizeof(int) * (size - cols));
	gpuWall.resize(size - parameters.cols);
//	cudaMemcpy(gpuWall, data + cols, sizeof(int) * (size - cols),
//			cudaMemcpyHostToDevice);

	gpuWall.fill_n(data.begin() + parameters.cols, (size - parameters.cols));

	int *gpuResult_ptr[2] = { gpuResult[0].data(), gpuResult[1].data() };

	int final_ret = calc_path(gpuWall.data(), gpuResult_ptr, parameters.rows,
			parameters.cols, parameters.pyramid_height, blockCols, borderCols);

//	cudaMemcpy(result, gpuResult[final_ret], sizeof(int) * cols,
//			cudaMemcpyDeviceToHost);
	gpuResult[final_ret].to_vector(result);

	std::ofstream of(output_file, std::ios::app);
	if (of.good()) {

		for (int i = 0; i < parameters.cols; i++)
			of << data[i] << " ";
//		printf("%d ", data[i]);

//	printf("\n");
		of << std::endl;

		for (int i = 0; i < parameters.cols; i++)
//		printf("%d ", result[i]);
			of << result[i] << " ";

//	printf("\n");
		of << std::endl;
	} else {
		throw_line("Could not open file " + output_file);
	}

}



int main(int argc, char** argv) {
	int num_devices;
	cudaGetDeviceCount(&num_devices);
	if (num_devices > 1)
		cudaSetDevice(DEVICE);

	run(argc, argv);

	return EXIT_SUCCESS;
}
