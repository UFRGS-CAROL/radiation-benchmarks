#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <iostream>
#include <math.h>
#include "needle.h"
#include <cuda.h>
#include <sys/time.h>

#define GCHK_BLOCK_SIZE 32
#define MAX_VALUE_NW 24

// includes, kernels
#include "needle_kernel.cu"


///////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);

#define N_ERRORS_LOG 500
#define ITERATIONS 1

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort =
		true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
				line);
		if (abort)
			exit(code);
	}
}
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

double gettime() {
	struct timeval t;
	
	gettimeofday(&t, NULL);
	return t.tv_sec + t.tv_usec * 1e-6;
}

double mysecond() {
	struct timeval tp;
	struct timezone tzp;
	int i = gettimeofday(&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

void ReadArrayFromFile(int* input_itemsets,  char** argv, std::string array = "") {
	double time = mysecond();
	int n = atoi(argv[1]) + 1;

	//"/home/carol/TestGPU/GenerateNeedleArray/NeedleInput_"
	if(array == ""){
		std::cout << "Input array path is null so I'm generating a random array  with size == " << n * n << std::endl;
		for(int i = 0; i < n * n; i++){
			input_itemsets[i] = rand() % MAX_VALUE_NW; //24 is from blosum size
		}

		FILE *f_a;
		std::string filenameinput(array);
		filenameinput += std::string("input_") + argv[1];
		f_a = fopen(filenameinput.c_str(), "wb");
		if (f_a == NULL) {
			std::cout << "error.\n";
			exit(-3);
		}
		fwrite(input_itemsets, sizeof(int) * n * n, 1, f_a);
		fclose(f_a);
	}else{
		FILE *f_a;
		f_a = fopen(array.c_str(), "rb");
		if (f_a == NULL) {
			std::cout << "error.\n";
			exit(-3);
		}
		std::cout << "read...";
		fread(input_itemsets, sizeof(int) * n * n, 1, f_a);
		fclose(f_a);
	}
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	printf("WG size of kernel = %d \n", BLOCK_SIZE);

	runTest(argc, argv);

	return EXIT_SUCCESS;
}

void UpdateTimestamp() {
	time_t timestamp = time(NULL);
	char time_s[50];
	sprintf(time_s, "%d", int(timestamp));

	char string[100] = "echo ";
	strcat(string, time_s);
	strcat(string, " > /home/carol/TestGPU/timestamp.txt");
	system(string);
}

void usage(int argc, char **argv) {
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> <input, if it exist>\n", argv[0]);
	fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
	fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
	exit(1);
}

void runTest(int argc, char** argv) {
	int max_rows, max_cols, penalty;
	int *input_itemsets, *output_itemsets,  *referrence;
	int *matrix_cuda, *referrence_cuda;
	int size;
	//int zero = 0;
	double timeG;
	std::string array_path;

	// the lengths of the two sequences should be able to divided by 16.
	// And at current stage  max_rows needs to equal max_cols
	if (argc >= 3) {
		max_rows = atoi(argv[1]);
		max_cols = atoi(argv[1]);
		penalty = atoi(argv[2]);
		if(argc > 3)
			array_path = std::string(argv[3]);
	} else {
		usage(argc, argv);
	}

	int n = atoi(argv[1]) + 1;

	if (atoi(argv[1]) % 16 != 0) {
		fprintf(stderr, "The dimension values must be a multiple of 16\n");
		exit(1);
	}

	//////////BLOCK and GRID size for goldchk////////////
	int gchk_gridsize = n / GCHK_BLOCK_SIZE < 1 ? 1 : n / GCHK_BLOCK_SIZE;
	int gchk_blocksize = n / GCHK_BLOCK_SIZE < 1 ? n : GCHK_BLOCK_SIZE;
	dim3 gchk_dimBlock(gchk_blocksize, gchk_blocksize);
	dim3 gchk_dimGrid(gchk_gridsize, gchk_gridsize);
	////////////////////////////////////////////////////

	//====================================
	//int ea = 0; //wrong integers in the current loop
	//int t_ea = 0; //total number of wrong integers
	//int old_ea = 0;

	double total_time = 0.0;

	max_rows = max_rows + 1;
	max_cols = max_cols + 1;
	referrence = (int *) malloc(max_rows * max_cols * sizeof(int));
	input_itemsets = (int *) malloc(max_rows * max_cols * sizeof(int));
	output_itemsets = (int *) malloc(max_rows * max_cols * sizeof(int));
	//gold_itemsets = (int *) malloc(max_rows * max_cols * sizeof(int));

	//int *kerrors;
	//kerrors = (int*) malloc(sizeof(int));

	if (!input_itemsets)
		fprintf(stderr, "error: can not allocate memory");

	printf("Start Needleman-Wunsch\n");

	ReadArrayFromFile(input_itemsets, argv, array_path);

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

	size = max_cols * max_rows;

	for (int loop2 = 0; loop2 < ITERATIONS; loop2++) {
		//file = fopen(file_name, "a");
		//std::cout << "Allocating matrixes on GPU...";
		cudaMalloc((void**) &referrence_cuda, sizeof(int) * size);
		cudaMalloc((void**) &matrix_cuda, sizeof(int) * size);
		if ((referrence_cuda == NULL) || (matrix_cuda == NULL)) {
			std::cout << "error.\n";
			exit(-3);
		}
		//std::cout << "Done\n";
		//std::cout << "Sending matrixes to GPU...";

		timeG = mysecond();

		cudaMemcpy(referrence_cuda, referrence, sizeof(int) * size,
				cudaMemcpyHostToDevice);
		cudaMemcpy(matrix_cuda, input_itemsets, sizeof(int) * size,
				cudaMemcpyHostToDevice);
		timeG = mysecond() - timeG;

		//std::cout << "Done in " << timeG << "s.\nRunning Needleman-Wunsch...";

		dim3 dimGrid;
		dim3 dimBlock(BLOCK_SIZE, 1);
		int block_width = (max_cols - 1) / BLOCK_SIZE;

		timeG = mysecond();

		//printf("Processing top-left matrix\n");
		//process top-left matrix
		for (int i = 1; i <= block_width; i++) {
			dimGrid.x = i;
			dimGrid.y = 1;
			needle_cuda_shared_1<<<dimGrid, dimBlock>>>(referrence_cuda,
					matrix_cuda, max_cols, penalty, i, block_width);
		}
		//printf("Processing bottom-right matrix\n");
		//process bottom-right matrix
		for (int i = block_width - 1; i >= 1; i--) {
			dimGrid.x = i;
			dimGrid.y = 1;
			needle_cuda_shared_2<<<dimGrid, dimBlock>>>(referrence_cuda,
					matrix_cuda, max_cols, penalty, i, block_width);
		}

		timeG = mysecond() - timeG;
		total_time += timeG;

		std::cout << "Done in " << timeG << "s.\nSaving gold\n";
		//retrieve information to save as gold
		cudaError_t mcpy = cudaMemcpy(input_itemsets, matrix_cuda,  sizeof(int) * size, cudaMemcpyDeviceToHost );
		FILE *f_a;
		std::string gold_name("gold_" + std::to_string(n-1));

		f_a = fopen(gold_name.c_str(), "wb");
		fwrite(input_itemsets, sizeof(int) * n * n, 1, f_a);
		fclose(f_a);

		cudaFree(referrence_cuda);
		cudaFree(matrix_cuda);
	}

	free(referrence);
	free(input_itemsets);
	free(output_itemsets);

}

