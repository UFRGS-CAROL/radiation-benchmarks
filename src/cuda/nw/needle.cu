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

// includes, kernels
#include "needle_kernel.cu"

//================== log include
#ifdef LOGS
#include "log_helper.h"
#endif
//====================================

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

void ReadArrayFromFile(int* input_itemsets, int* gold_itemsets, char** argv) {
	double time = mysecond();
	int n = atoi(argv[1]) + 1;

	std::string filenameinput(
			"/home/carol/TestGPU/GenerateNeedleArray/NeedleInput_");
	filenameinput += argv[1];
	filenameinput += ".array";
	std::string filenamegold(
			"/home/carol/TestGPU/GenerateNeedleArray/NeedleGOLD_");
	filenamegold += argv[1];
	filenamegold += ".array";

	std::cout << "open array...";

	FILE *f_a, *f_gold;
	f_a = fopen(filenameinput.c_str(), "rb");
	f_gold = fopen(filenamegold.c_str(), "rb");

	if ((f_a == NULL) || (f_gold == NULL)) {
		std::cout << "error.\n";
		exit(-3);
	}

	std::cout << "read...";
	fread(input_itemsets, sizeof(int) * n * n, 1, f_a);
	fread(gold_itemsets, sizeof(int) * n * n, 1, f_gold);
	fclose(f_a);
	fclose(f_gold);

	printf("ok in %f\n", mysecond() - time);
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
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> \n", argv[0]);
	fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
	fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
	exit(1);
}

void runTest(int argc, char** argv) {
	int max_rows, max_cols, penalty;
	int *input_itemsets, *output_itemsets, *gold_itemsets, *referrence;
	int *matrix_cuda, *referrence_cuda;
	int size;
	int zero = 0;
	double timeG;

	// the lengths of the two sequences should be able to divided by 16.
	// And at current stage  max_rows needs to equal max_cols
	if (argc == 3) {
		max_rows = atoi(argv[1]);
		max_cols = atoi(argv[1]);
		penalty = atoi(argv[2]);
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

	// Log files
	/*FILE* file;
	 FILE* log_file;
	 */
	//================== Init logs
#ifdef LOGS
	char test_info[90];
	snprintf(test_info, 90, "max_rows:%d max_cols:%d penalty:%d", max_rows, max_cols, penalty);
	start_log_file("cudaNW", test_info);
#endif
	//====================================
	int ea = 0; //wrong integers in the current loop
	int t_ea = 0; //total number of wrong integers
	int old_ea = 0;

	double total_time = 0.0;
	////////////////FILE NAME//////////////////////////////
	/*time_t file_time;
	 struct tm *ptm;
	 char day[2], month[2], year[4], hour[2], second[2], minute[2];
	 char file_name[60];
	 char file_name_log[60];

	 file_time = time(NULL);
	 ptm = gmtime(&file_time);

	 std::string logfile_matrixname("cudaNW");
	 logfile_matrixname += argv[1];

	 snprintf(day, sizeof(day + 1), "%d", ptm->tm_mday);
	 snprintf(month, sizeof(month + 1), "%d", ptm->tm_mon + 1);
	 snprintf(year, sizeof(year + 1), "%d", ptm->tm_year + 1900);
	 snprintf(hour, sizeof(hour + 1), "%d", ptm->tm_hour);
	 snprintf(minute, sizeof(minute + 1), "%d", ptm->tm_min);
	 snprintf(second, sizeof(second + 1), "%d", ptm->tm_sec);
	 strcpy(file_name, day);
	 strcat(file_name, "_");
	 strcat(file_name, month);
	 strcat(file_name, "_");
	 strcat(file_name, year);
	 strcat(file_name, "_");
	 strcat(file_name, hour);
	 strcat(file_name, "_");
	 strcat(file_name, minute);
	 strcat(file_name, "_");
	 strcat(file_name, second);
	 strcat(file_name, "_");
	 strcat(file_name, logfile_matrixname.c_str());
	 strcpy(file_name_log, file_name);

	 strcat(file_name, ".txt");
	 strcat(file_name_log, "log.txt");
	 *///////////////////////////////////////////////////////
	max_rows = max_rows + 1;
	max_cols = max_cols + 1;
	referrence = (int *) malloc(max_rows * max_cols * sizeof(int));
	input_itemsets = (int *) malloc(max_rows * max_cols * sizeof(int));
	output_itemsets = (int *) malloc(max_rows * max_cols * sizeof(int));
	gold_itemsets = (int *) malloc(max_rows * max_cols * sizeof(int));

	int *kerrors;
	kerrors = (int*) malloc(sizeof(int));

	if (!input_itemsets)
		fprintf(stderr, "error: can not allocate memory");

	printf("Start Needleman-Wunsch\n");

	ReadArrayFromFile(input_itemsets, gold_itemsets, argv);

	/*    	srand ( time(NULL) );
	 std::cout << "Original -  input : " << input_itemsets[1*max_cols+0] << "\treference : " << blosum62[input_itemsets[1*max_cols]][input_itemsets[1]] << "\n";
	 input_itemsets[1*max_cols+0] = rand () % 10 + 1;
	 std::cout << "Modified -  input : " << input_itemsets[1*max_cols+0] << "\treference : " << blosum62[input_itemsets[1*max_cols]][input_itemsets[1]] << "\n";
	 */
	/*for (int i = 0 ; i < max_cols; i++)
	 {
	 //getchar();
	 for (int j = 0 ; j < max_rows; j++)
	 std::cout << "[" << i << "][" << j << "] : " << input_itemsets[i*max_cols+j] << "\t";
	 }*/
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
#ifdef LOGS
		start_iteration();
#endif
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
#ifdef LOGS
		end_iteration();
#endif
		timeG = mysecond() - timeG;
		total_time += timeG;

		//std::cout << "Done in " << timeG << "s.\n";

		*kerrors = 0;
		// Check errors on GPU...
		//std::cout << "Sending gold matrix to GPU...";
		timeG = mysecond();
		cudaMemcpy(referrence_cuda, gold_itemsets, sizeof(int) * size,
				cudaMemcpyHostToDevice);
		// Using referrence just to avoid reallocation for gold
		cudaMemcpyToSymbol(gpukerrors, &zero, sizeof(int));
		timeG = mysecond() - timeG;
		//std::cout << "Done in " << timeG << "s.\nRunning GoldChk...";
		timeG = mysecond();

		GoldChkKernel<<<gchk_dimGrid, gchk_dimBlock>>>(referrence_cuda,
				matrix_cuda, n);

		timeG = mysecond() - timeG;
		//std::cout << "Done in " << timeG << "s.";
		gpuErrchk(cudaPeekAtLastError());
		cudaMemcpyFromSymbol(kerrors, gpukerrors, sizeof(unsigned int));

		//std::cout << "Errors : " << *kerrors << "\n";

		///////////UPDATE FILE//////////////////////
		/*file_time = time(NULL);
		 ptm = gmtime(&file_time);
		 snprintf(hour, sizeof(hour + 1), "%d", ptm->tm_hour);
		 snprintf(minute, sizeof(minute + 1), "%d", ptm->tm_min);
		 snprintf(second, sizeof(second + 1), "%d", ptm->tm_sec);
		 fprintf(file, "\n start time: %s/%s_%s:%s:%s", day, month, hour, minute,
		 second);
		 fclose (file);
		 */

		ea = 0;

		/////////////UPDATE TIMESTAMP///////////////////
		//UpdateTimestamp();
		////////////////////////////////////////////////

		if (*kerrors > 0) {
			//file = fopen(file_name, "a");
			//std::cout <<  << *kerrors << "\n";
			std::stringstream sstm;
			sstm << "Error detected! kerrors = " << *kerrors;
#ifdef LOGS
			log_error_detail(sstm.str().c_str());
#endif
			cudaMemcpy(output_itemsets, matrix_cuda, sizeof(int) * size,
					cudaMemcpyDeviceToHost);
			int host_errors =0;
			for (int i = 0; (i < n) && (ea < N_ERRORS_LOG); i++) {
				for (int j = 0; (j < n) && (ea < N_ERRORS_LOG); j++) {
					if (output_itemsets[i + n * j]
							!= gold_itemsets[i + n * j]) {
						ea++;
						char error_detail[200];
						sprintf(error_detail,
								"\n p: [%d, %d], r: %i, e: %i, error: %d\n", i,
								j, output_itemsets[i + n * j],
								gold_itemsets[i + n * j], ea);
#ifdef LOGS
						log_error_detail(error_detail);
						host_errors++;
#endif

					}
				}
			}
			t_ea += *kerrors;

			///////////UPDATE LOG FILE//////////////////////
			/*log_file = fopen(file_name_log, "a");
			fprintf(log_file, "\ntest number: %d", loop2);
			fprintf(log_file, "\ntime: %f", timeG);
			fprintf(log_file, "\ntotal time: %f", total_time);
			fprintf(log_file, "\nerrors: %d", *kerrors);
			fprintf(log_file, "\ntotal errors: %d", t_ea);
			fclose (log_file);
			fclose (file);
			*/
#ifdef LOGS
			log_error_count(host_errors);
#endif
		}

		if (*kerrors > 0 || (loop2 % 10 == 0)) {
			printf("\ntest number: %d", loop2);
			printf("\ntotal time: %f", total_time);
			printf("\nerrors: %d", *kerrors);
			printf("\ntotal errors: %d\n", t_ea);
			if ((*kerrors != 0) && (*kerrors == old_ea)) {
				old_ea = 0;
				return;
			}

			old_ea = *kerrors;
		} else {
			printf(".");
		}
		cudaFree(referrence_cuda);
		cudaFree(matrix_cuda);
	}
#ifdef LOGS
	end_log_file();
#endif
	free(referrence);
	free(input_itemsets);
	free(output_itemsets);

}

