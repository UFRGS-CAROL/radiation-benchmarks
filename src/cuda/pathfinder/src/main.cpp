#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <fstream>
#include <string>
#include <omp.h>
#include <algorithm>

#include "common.h"
#include "Parameters.h"
#include "generic_log.h"

void generate_input(vector<int*>& wall, vector<int>& data, int pyramid_height,
		int rows, int cols, std::string output_file) {

//	data = new int[rows * cols];
	data.resize(rows * cols);
	wall.resize(rows);

	for (int n = 0; n < rows; n++) {
		wall[n] = data.data() + cols * n;
	}

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

void read_input(vector<int*>& wall, vector<int>& data, vector<int>& gold,
		int pyramid_height, int rows, int cols, std::string input_file) {
	data.resize(rows * cols);
	wall.resize(rows);
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
size_t compare_output(matrix_hst<int>& output, vector<int>& gold,
		rad::Log& log) {
	auto stream_number = output.size();
	auto size_of_solution = gold.size();
	// acc errors
	size_t errors = 0;

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
			for (auto solution = 0; solution < size_of_solution; solution++) {
				auto g = gold[solution];
				auto f = output[stream][solution];
				if (g != f) {
					auto e = std::to_string(g);
					auto r = std::to_string(f);
					auto stream_str = std::to_string(stream);
					auto solution_str = std::to_string(solution);

					std::string error_detail = "Stream:" + stream_str;
					error_detail += " solution_n: " + solution_str;
					error_detail += " e: " + e;
					error_detail += " r: " + r;
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

	log.update_errors();
	return errors;
}

void run(int argc, char** argv) {
	Parameters parameters(argc, argv);

	vector<int*> wall;
	vector<int> data;
	vector<int> gold;

	std::string& output_file = parameters.gold;

	if (parameters.generate) {
		generate_input(wall, data, parameters.pyramid_height, parameters.rows,
				parameters.cols, output_file);
	} else {
		read_input(wall, data, gold, parameters.pyramid_height, parameters.rows,
				parameters.cols, output_file);
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

	auto streams = parameters.sm_count;
	std::string test_info = "rows:" + std::to_string(parameters.rows);
	test_info += " cols:" + std::to_string(parameters.cols);
	test_info += " pyramid_height:" + std::to_string(parameters.pyramid_height);
	test_info += " streams:" + std::to_string(streams);

	rad::Log log("cudaPATHFINDER", test_info);

	//multi streams execution
	vector<cuda_stream> streams_vec(streams);
	vector<int> streams_final_ret(streams);
	matrix_hst<int> all_results(streams);
	vector<matrix_dev<int>> gpu_result_streams(streams);

	for (auto stream = 0; stream < streams; stream++) {
		//for each stream allocate 2 DeviceVectors
		gpu_result_streams[stream].resize(2);
		gpu_result_streams[stream][0].resize(parameters.cols);
		gpu_result_streams[stream][1].resize(parameters.cols);
		all_results[stream].resize(parameters.cols);
	}

	//int *gpuWall, *gpuResult[2];
	rad::DeviceVector<int> gpu_wall;

	int size = parameters.rows * parameters.cols;

	gpu_wall.resize(size - parameters.cols);
	gpu_wall.fill_n(data.begin() + parameters.cols, (size - parameters.cols));

	for (size_t iteration = 0; iteration < parameters.iterations; iteration++) {
		//reset the memory on gpu to default before restarting
		auto set_time = rad::mysecond();
		for (auto& gpu_result : gpu_result_streams) {
			gpu_result[0].fill_n(data.begin(), parameters.cols);
			gpu_result[1].clear();
		}
		set_time = rad::mysecond() - set_time;

		//Kernel processing
		auto kernel_time = rad::mysecond();
		log.start_iteration();

#pragma omp parallel for default(shared)
		for (auto stream = 0; stream < streams; stream++) {
			int *gpuResult_ptr[2] = {
			//split streams input
					gpu_result_streams[stream][0].data(),	//0
					gpu_result_streams[stream][1].data() //1
					};

			streams_final_ret[stream] = calc_path(gpu_wall.data(), gpuResult_ptr,
					parameters.rows, parameters.cols, parameters.pyramid_height,
					blockCols, borderCols, streams_vec[stream]);

		}

		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		;
		rad::checkFrameworkErrors(cudaGetLastError());

		log.end_iteration();
		kernel_time = rad::mysecond() - kernel_time;

		/*
		 * SETUP things
		 */
		auto copy_time = rad::mysecond();
		for (auto stream = 0; stream < streams; stream++) {
			auto final_ret = streams_final_ret[stream];
			gpu_result_streams[stream][final_ret].to_vector(all_results[stream]);
		}
		copy_time = rad::mysecond() - copy_time;

		auto compare_time = rad::mysecond();
		auto errors = compare_output(all_results, gold, log);
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
					<< ceil(wasted_time / overall_time * 100.0f) << "%)\n"
					<< std::endl;
		}
	}

	if (parameters.generate) {
		std::cout << "Saving the output of stream 0 at " << output_file
				<< std::endl;
		std::ofstream of(output_file, std::ios::binary | std::ios::app);
		if (of.good()) {
			auto ptr = reinterpret_cast<char*>(all_results[0].data());
			of.write(ptr, parameters.cols * sizeof(int));
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
