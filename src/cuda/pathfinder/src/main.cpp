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
#include "cuda_utils.h"
#include "generic_log.h"

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
size_t compare_output(vector<int>& output, vector<int>& gold, rad::Log& log) {
	auto stream_number = output.size();
	auto no_of_nodes = gold.size();
	// acc errors
	size_t errors = 0;
	/*
	 static std::vector<bool> equal_array(stream_number, false);
	 //first check if output is ok
	 //by not doing it the comparison time increases 20%
	 #pragma omp parallel for default(shared)
	 for (auto i = 0; i < stream_number; i++) {
	 equal_array[i] = (output[i] != gold);
	 }

	 auto falses = std::count(equal_array.begin(), equal_array.end(), false);

	 if (falses != equal_array.size()) {
	 #pragma omp parallel for default(shared)
	 for (auto stream = 0; stream < stream_number; stream++) {
	 for (auto node = 0; node < no_of_nodes; node++) {
	 auto g = gold[node];
	 auto f = output[stream][node];
	 if (g != f) {
	 auto cost_e = std::to_string(g);
	 auto cost_r = std::to_string(f);
	 auto stream_str = std::to_string(stream);
	 auto node_str = std::to_string(node);

	 std::string error_detail = "Stream:" + stream_str;
	 error_detail += " Node: " + node_str;
	 error_detail += " cost_e: " + cost_e;
	 error_detail += " cost_r: " + cost_r;
	 #pragma omp critical
	 {
	 std::cout << error_detail << std::endl;
	 log.log_error_detail(error_detail);
	 errors++;
	 }
	 }

	 }
	 }
	 }
	 */
	log.update_errors();
	return errors;
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

	auto streams = 1;
	std::string test_info = "rows:" + std::to_string(parameters.rows);
	test_info += " cols:" + std::to_string(parameters.cols);
	test_info += " pyramid_height:" + std::to_string(parameters.pyramid_height);
	test_info += " streams:" + std::to_string(streams);

	rad::Log log("cudaPATHFINDER", test_info);


	//int *gpuWall, *gpuResult[2];
	rad::DeviceVector<int> gpuWall;
	rad::DeviceVector<int> gpuResult[2];

	int size = parameters.rows * parameters.cols;
	gpuResult[0].resize(parameters.cols);
	gpuResult[1].resize(parameters.cols);

	gpuWall.resize(size - parameters.cols);
	gpuWall.fill_n(data.begin() + parameters.cols, (size - parameters.cols));

	for (size_t iteration = 0; iteration < parameters.iterations; iteration++) {
		//reset the memory on gpu to default before restarting
		auto set_time = rad::mysecond();
		gpuResult[0].fill_n(data.begin(), parameters.cols);
		gpuResult[1].clear();
		set_time = rad::mysecond() - set_time;

		//Kernel processing
		auto kernel_time = rad::mysecond();
		log.start_iteration();
		int *gpuResult_ptr[2] = { gpuResult[0].data(), gpuResult[1].data() };
		int final_ret = calc_path(gpuWall.data(), gpuResult_ptr,
				parameters.rows, parameters.cols, parameters.pyramid_height,
				blockCols, borderCols);

		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		;
		rad::checkFrameworkErrors(cudaGetLastError());

		log.end_iteration();
		kernel_time = rad::mysecond() - kernel_time;

		/*
		 * SETUP things
		 */
		auto copy_time = rad::mysecond();
		gpuResult[final_ret].to_vector(result);
		copy_time = rad::mysecond() - copy_time;

		auto compare_time = rad::mysecond();
		auto errors = compare_output(result, gold, log);
		compare_time = rad::mysecond() - compare_time;

		if (parameters.verbose) {
			auto wasted_time = copy_time + set_time + compare_time;
			auto overall_time = wasted_time + kernel_time;
			std::cout << "Iteration " << iteration << " - ERRORS: " << errors
					<< std::endl;
			std::cout << "Overall time " << overall_time;
			std::cout << " Set time " << set_time;
			std::cout << " Kernel time " << kernel_time;
			std::cout << " Compare time " << compare_time;
			std::cout << " Copy time " << copy_time << std::endl;
			std::cout << "Wasted time " << wasted_time << " ("
					<< int(wasted_time / overall_time * 100.0f) << "%)\n"
					<< std::endl;
		}
	}

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
