#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <iostream>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>
#include <vector>
#include <fstream>
#include <omp.h>
#include <numeric>

//helper kernels
#include "include/cuda_utils.h"
#include "include/device_vector.h"

// includes, kernels
#include "needle.h"
//================== log include
#ifdef LOGS
#include "log_helper.h"
#endif
//====================================

#define GCHK_BLOCK_SIZE 32
#define MAX_VALUE_NW 24
#define LIMIT -999
#define N_ERRORS_LOG 500
#define ITERATIONS 100
#define CHAR_CAST(x) (reinterpret_cast<char*>(x))
#define CONST_CAST(x) (const_cast<char*>(x))

int blosum62[24][24] = { { 4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2,
		-1, 1, 0, -3, -2, 0, -2, -1, 0, -4 }, { -1, 5, 0, -2, -3, 1, 0, -2, 0,
		-3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1, -4 },
		{ -2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3,
				3, 0, -1, -4 }, { -2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1,
				-3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1, -4 }, { 0, -3, -3, -3,
				9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1,
				-3, -3, -2, -4 }, { -1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0,
				-3, -1, 0, -1, -2, -1, -2, 0, 3, -1, -4 },
		{ -1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2,
				-2, 1, 4, -1, -4 }, { 0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4,
				-2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1, -4 }, { -2, 0, 1,
				-1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3,
				0, 0, -1, -4 }, { -1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3,
				1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1, -4 }, { -1, -2, -3, -4,
				-1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4,
				-3, -1, -4 }, { -1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1,
				-3, -1, 0, -1, -3, -2, -2, 0, 1, -1, -4 }, { -1, -1, -2, -3, -1,
				0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1,
				-1, -4 }, { -2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6,
				-4, -2, -2, 1, 3, -1, -3, -3, -1, -4 }, { -1, -2, -2, -1, -3,
				-1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2,
				-1, -2, -4 }, { 1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2,
				-1, 4, 1, -3, -2, -2, 0, 0, 0, -4 },
		{ 0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2,
				-2, 0, -1, -1, 0, -4 }, { -3, -3, -4, -4, -2, -2, -3, -2, -2,
				-3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2, -4 }, {
				-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2,
				-2, 2, 7, -1, -3, -2, -1, -4 }, { 0, -3, -3, -3, -1, -2, -2, -3,
				-3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1, -4 }, {
				-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4,
				-3, -3, 4, 1, -1, -4 }, { -1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3,
				1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4 }, { 0, -1, -1,
				-1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1,
				-1, -1, -1, -1, -4 }, { -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
				-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1 } };

template<typename T>
bool read_from_file(std::string& path, std::vector<T>& array) {
	std::ifstream input(path, std::ios::binary);
	if (input.good()) {
		input.read(CHAR_CAST(array.data()), array.size() * sizeof(T));
		input.close();
		return true;
	}
	return false;
}

template<typename T>
bool write_to_file(std::string& path, std::vector<T>& array) {
	std::ofstream output(path, std::ios::binary);
	if (output.good()) {
		output.write(CHAR_CAST(array.data()), array.size() * sizeof(T));
		output.close();
		return true;
	}
	return false;
}

void GenerateInputFile(std::vector<int>& input_itemsets,
		std::string filenameinput) {
	std::cout << "Generating a random array  with size == "
			<< input_itemsets.size() << std::endl;
	for (auto& i : input_itemsets) {
		i = rand() % MAX_VALUE_NW; //24 is from blosum size
	}

	if (write_to_file(filenameinput, input_itemsets) == false) {
		std::cout << "error generating input.\n";
		exit(-3);
	}
}

void WriteGoldToFile(std::vector<int>& gold_array, std::string gold_name,
		int n) {
	if (write_to_file(gold_name, gold_array) == false) {
		std::cout << "error writing gold.\n";
		exit(-3);
	}
}

void ReadArrayFromFile(std::vector<int>& input_itemsets,
		std::vector<int>& gold_itemsets, std::string filenameinput,
		std::string filenamegold) {
	double time = rad::mysecond();
	std::cout << "open array...\n";

	if (read_from_file(filenameinput, input_itemsets) == false
			|| read_from_file(filenamegold, gold_itemsets) == false) {
		std::cout << "error.\n";
		exit(-3);
	}

	std::cout << "read...";
	printf("ok in %f\n", rad::mysecond() - time);
}

//int badass_memcmp(std::vector<int>&  gold, std::vector<int>& found) {
//	int result = 0;
//	unsigned long chunk = ceil(double(gold.size()) / double(omp_get_max_threads()));
//	// printf("size %d max threads %d chunk %d\n", n, omp_get_max_threads(), chunk);
//	double time = rad::mysecond();
//#pragma omp parallel for default(shared) schedule(static,chunk) reduction(+:result)
//	for (size_t i = 0; i < gold.size(); i++)
//		result = result + (gold[i] - found[i]);
//
////	printf("comparing took %lf seconds, diff %d\n", rad::mysecond() - time, result);
//	return result;
//}
bool inline badass_memcmp(std::vector<int>& gold_vector,
		std::vector<int>& found_vector) {
	auto badass_time = rad::mysecond();
	uint32_t n = gold_vector.size();
	uint32_t numthreads = omp_get_max_threads();
	uint32_t chunk = ceil(float(n) / float(numthreads));
	static std::vector<uint32_t> reduction_array(numthreads);

	int *gold = gold_vector.data();
	int *found = found_vector.data();

#pragma omp parallel default(shared)
	{
		uint32_t tid = omp_get_thread_num();
		uint32_t i = tid * chunk;
		reduction_array[tid] = std::equal(gold + i, gold + i + chunk,
				found + i);
	}
	auto result = std::accumulate(reduction_array.begin(), reduction_array.end(), 0);
	badass_time = rad::mysecond() - badass_time;
	std::cout << "BADASS TIME " << badass_time << std::endl;
	return (result != 0);
}

void usage(int argc, char **argv) {
	fprintf(stderr,
			"Usage: %s <max_rows/max_cols> <penalty> <input_array> <gold_array> <iterations> "
					"<to generate gold 0 or 1>\n", argv[0]);
	fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
	fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
	exit(1);
}

void runTest(int argc, char** argv) {
	int max_rows, max_cols, penalty;
//	const KErrorsType zero = 0;
	int iterations = 1;
	double timeG;
	bool generate = false;
	std::string array_path, gold_path;

	// the lengths of the two sequences should be able to divided by 16.
	// And at current stage  max_rows needs to equal max_cols
	if (argc == 7) {
		max_rows = atoi(argv[1]);
		max_cols = atoi(argv[1]);
		penalty = atoi(argv[2]);
		array_path = std::string(argv[3]);
		gold_path = std::string(argv[4]);
		iterations = atoi(argv[5]);
		generate = atoi(argv[6]);
		if (generate)
			iterations = 1;
	} else {
		usage(argc, argv);
	}

	int n = atoi(argv[1]) + 1;

	if (atoi(argv[1]) % 16 != 0) {
		std::cerr << "The dimension values must be a multiple of 16\n";
		exit(1);
	}

	//////////BLOCK and GRID size for goldchk////////////
	int gchk_gridsize = n / GCHK_BLOCK_SIZE < 1 ? 1 : n / GCHK_BLOCK_SIZE;
	int gchk_blocksize = n / GCHK_BLOCK_SIZE < 1 ? n : GCHK_BLOCK_SIZE;
	dim3 gchk_dimBlock(gchk_blocksize, gchk_blocksize);
	dim3 gchk_dimGrid(gchk_gridsize, gchk_gridsize);
	////////////////////////////////////////////////////

	// Log files
	/*FILE* file;
	 FILE* log_file;
	 */
	//================== Init logs
#ifdef LOGS
//"max_rows:%d max_cols:%d penalty:%d"
	std::string test_info = "";
	test_info += "max_rows:" + std::to_string(max_rows) + " ";
	test_info += "max_cols:" + std::to_string(max_cols) + " ";
	test_info += "penalty:" + std::to_string(penalty);
	start_log_file(CONST_CAST("cudaNW"), CONST_CAST(test_info.c_str()));
#endif
	//====================================
	KErrorsType ea = 0; //wrong integers in the current loop
	KErrorsType t_ea = 0; //total number of wrong integers
//	KErrorsType old_ea = 0;

	double total_time = 0.0;

	max_rows++;
	max_cols++;
	int size = max_cols * max_rows;

	std::vector<int> referrence(size);
	std::vector<int> input_itemsets(size);
	std::vector<int> output_itemsets(size);
	std::vector<int> gold_itemsets(size);

	rad::DeviceVector<int> referrence_cuda = referrence;
	rad::DeviceVector<int> matrix_cuda = input_itemsets;
	rad::DeviceVector<int> output_itemsets_cuda = output_itemsets;
	rad::DeviceVector<int> gold_itemsets_cuda = gold_itemsets;
//	int *referrence = (int *) malloc(max_rows * max_cols * sizeof(int));
//	int *input_itemsets = (int *) malloc(max_rows * max_cols * sizeof(int));
//	int *output_itemsets = (int *) malloc(max_rows * max_cols * sizeof(int));
//	int *gold_itemsets = (int *) malloc(max_rows * max_cols * sizeof(int));
//	int *matrix_cuda, *referrence_cuda;
//	std::cout << "Allocating matrixes on GPU...";
//	cudaMalloc((void**) &referrence_cuda, sizeof(int) * size);
//	cudaMalloc((void**) &matrix_cuda, sizeof(int) * size);
	//	if ((referrence_cuda == NULL) || (matrix_cuda == NULL)) {
	//		std::cout << "error.\n";
	//		exit(-3);
	//	}
//	std::cout << "Done\n";

//	KErrorsType kerrors = 0;

//	if (!input_itemsets)
//		fprintf(stderr, "error: can not allocate memory");

	std::cout << "Start Needleman-Wunsch" << std::endl;

	if (generate) {
		GenerateInputFile(input_itemsets, array_path);
	} else {
		ReadArrayFromFile(input_itemsets, gold_itemsets, array_path, gold_path);
	}

	for (int i = 1; i < max_cols; i++) {
		for (int j = 1; j < max_rows; j++) {
			referrence[i * max_cols + j] =
					blosum62[input_itemsets[i * max_cols]][input_itemsets[j]];
		}
	}
	for (int i = 1; i < max_rows; i++)
		input_itemsets[i * max_cols] = -i * penalty;
	for (int j = 1; j < max_cols; j++)
		input_itemsets[j] = -j * penalty;

	for (int loop2 = 0; loop2 < iterations; loop2++) {
		auto mem_cpy_time = rad::mysecond();
		matrix_cuda = input_itemsets;
		mem_cpy_time = rad::mysecond() - mem_cpy_time;

		dim3 dimGrid;
		dim3 dimBlock(BLOCK_SIZE, 1);
		int block_width = (max_cols - 1) / BLOCK_SIZE;

		timeG = rad::mysecond();
#ifdef LOGS
		start_iteration();
#endif
		//printf("Processing top-left matrix\n");
		//process top-left matrix
		for (int i = 1; i <= block_width; i++) {
			dimGrid.x = i;
			dimGrid.y = 1;
			needle_cuda_shared_1<<<dimGrid, dimBlock>>>(referrence_cuda.data(),
					matrix_cuda.data(), max_cols, penalty, i, block_width);
		}
		//printf("Processing bottom-right matrix\n");
		//process bottom-right matrix
		for (int i = block_width - 1; i >= 1; i--) {
			dimGrid.x = i;
			dimGrid.y = 1;
			needle_cuda_shared_2<<<dimGrid, dimBlock>>>(referrence_cuda.data(),
					matrix_cuda.data(), max_cols, penalty, i, block_width);
		}
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		rad::checkFrameworkErrors(cudaPeekAtLastError());

#ifdef LOGS
		end_iteration();
#endif
		timeG = rad::mysecond() - timeG;
		total_time += timeG;

		//std::cout << "Done in " << timeG << "s.\n";
		if (generate == false) {
			/*			kerrors = 0;
			 // Check errors on GPU...
			 //std::cout << "Sending gold matrix to GPU...";
			 timeG = rad::mysecond();
			 //			referrence_cuda = gold_itemsets;
			 // Using referrence just to avoid reallocation for gold
			 rad::checkFrameworkErrors(
			 cudaMemcpyToSymbol(gpukerrors, &zero, sizeof(KErrorsType)));
			 timeG = rad::mysecond() - timeG;
			 //std::cout << "Done in " << timeG << "s.\nRunning GoldChk...";
			 timeG = rad::mysecond();

			 GoldChkKernel<<<gchk_dimGrid, gchk_dimBlock>>>(
			 gold_itemsets_cuda.data(), output_itemsets_cuda.data(), n);

			 timeG = rad::mysecond() - timeG;
			 rad::checkFrameworkErrors(cudaDeviceSynchronize());
			 rad::checkFrameworkErrors((cudaPeekAtLastError()));
			 rad::checkFrameworkErrors(
			 cudaMemcpyFromSymbol(&kerrors, gpukerrors,
			 sizeof(KErrorsType)));
			 std::cout << "Done in " << timeG << "s.";
			 output_itemsets = output_itemsets_cuda.to_vector();
			 */
			ea = 0;
			uint32_t host_errors = 0;

			auto copy_time = rad::mysecond();
			matrix_cuda.to_vector(output_itemsets);
			copy_time = rad::mysecond() - copy_time;

			auto cmp_time = rad::mysecond();
			if (badass_memcmp(gold_itemsets, output_itemsets)) {
//#ifdef LOGS
////				sprintf(error_info, "Error detected! kerrors = %d" ,kerrors);
//				std::string error_info = "Error detected! kerrors = " + std::to_string(kerrors);
//				log_error_detail(CONST_CAST(error_info.c_str()));
//#endif

//				rad::checkFrameworkErrors(
//						cudaMemcpy(output_itemsets, matrix_cuda,
//								sizeof(int) * size, cudaMemcpyDeviceToHost));
				for (int i = 0; (i < n) && (ea < N_ERRORS_LOG); i++) {
					for (int j = 0; (j < n) && (ea < N_ERRORS_LOG); j++) {
						if (output_itemsets[i + n * j]
								!= gold_itemsets[i + n * j]) {
							ea++;
//							char error_detail[200];
							std::string error_detail = "";
							error_detail += " p: [" + std::to_string(i) + ", "
									+ std::to_string(j) + "],";
							error_detail += " r: "
									+ std::to_string(output_itemsets[i + n * j])
									+ ",";
							error_detail += " e: "
									+ std::to_string(gold_itemsets[i + n * j])
									+ ",";
							error_detail += " error: " + std::to_string(ea);

//							sprintf(error_detail,
//									" p: [%d, %d], r: %i, e: %i, error: %d", i,
//									j, output_itemsets[i + n * j],
//									gold_itemsets[i + n * j], ea);
#ifdef LOGS
							log_error_detail(CONST_CAST(error_detail.c_str()));
							host_errors++;
#endif

						}
					}
				}
				t_ea += host_errors;

#ifdef LOGS
				log_error_count(host_errors);
#endif
			}
			cmp_time = rad::mysecond() - cmp_time;

			if (host_errors > 0 || (loop2 % 10 == 0)) {
				std::cout << "iteration: " << loop2;
				std::cout << " kernel time: " << timeG;
				std::cout << " total time: " << total_time;
				std::cout << " errors: " << host_errors;
				std::cout << " matrix cuda set: " << mem_cpy_time;
				std::cout << " cpy time: " << copy_time;
				std::cout << " cmp time: " << cmp_time;
				std::cout << " total errors: " << t_ea << std::endl;
			} else {
				std::cout << "." << std::flush;
			}
		} else {
			output_itemsets = matrix_cuda.to_vector();
			WriteGoldToFile(output_itemsets, gold_path, max_rows);
		}

	}

#ifdef LOGS
	end_log_file();
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	std::cout << "WG size of kernel = " << BLOCK_SIZE << std::endl;

	runTest(argc, argv);

	return EXIT_SUCCESS;
}

