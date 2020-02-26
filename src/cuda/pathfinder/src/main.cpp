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

void generate_input(vector<int*>& wall, vector<int>& data, vector<int>& result,
		int pyramid_height, int rows, int cols, std::string output_file) {

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

	std::ofstream of(output_file, std::ios::binary);
	if (of.good()) {
		of.write(reinterpret_cast<char*>(data.data()),
				rows * cols * sizeof(int));
		of.close();
	} else {
		throw_line("Could not open file " + output_file);
	}
}

void read_input(vector<int*>& wall, vector<int>& data, vector<int>& result,
		vector<int>& gold, int pyramid_height, int rows, int cols,
		std::string input_file) {
	data.resize(rows * cols);
	wall.resize(rows);
	result.resize(cols);
	gold.resize(cols);

	std::ifstream ifp(input_file, std::ios::binary);
	if (ifp.good()) {
		ifp.read(reinterpret_cast<char*>(data.data()),
				rows * cols * sizeof(int));
		ifp.read(reinterpret_cast<char*>(gold.data()), cols * sizeof(int));
		ifp.close();
	} else {
		throw_line("Could not open file " + input_file);
	}

	for (int n = 0; n < rows; n++) {
		wall[n] = data.data() + cols * n;
	}

}

void run(int argc, char** argv) {
	Parameters parameters(argc, argv);

	vector<int*> wall;
	vector<int> data;
	vector<int> result;
	vector<int> gold;

	std::string& output_file = parameters.gold;

	if (parameters.generate) {
		generate_input(wall, data, result, parameters.pyramid_height,
				parameters.rows, parameters.cols, output_file);
	} else {
		read_input(wall, data, result, gold, parameters.pyramid_height,
				parameters.rows, parameters.cols, output_file);
	}

	/* --------------- pyramid parameters --------------- */
	int borderCols = (parameters.pyramid_height) * HALO;
	int smallBlockCol = BLOCK_SIZE - (parameters.pyramid_height) * HALO * 2;
	int blockCols = parameters.cols / smallBlockCol
			+ ((parameters.cols % smallBlockCol == 0) ? 0 : 1);

	if (parameters.verbose) {
		std::cout << parameters << std::endl;
		std::cout << "pyramidHeight: " << parameters.pyramid_height
				<< std::endl;
		std::cout << "gridSize: [" << parameters.cols << "]" << std::endl;
		std::cout << "border:[" << borderCols << "]" << std::endl;
		std::cout << "blockSize: " << BLOCK_SIZE << std::endl;
		std::cout << "blockGrid:[" << blockCols << "]" << std::endl;
		std::cout << "targetBlock:[" << smallBlockCol << "]" << std::endl;

	}

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

	if (parameters.generate) {
		std::cout << "Saving the output at " << output_file << std::endl;
		std::ofstream of(output_file, std::ios::binary | std::ios::app);
		if (of.good()) {
			of.write(reinterpret_cast<char*>(result.data()),
					parameters.cols * sizeof(int));
			of.close();
		} else {
			throw_line("Could not open file " + output_file);
		}
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
