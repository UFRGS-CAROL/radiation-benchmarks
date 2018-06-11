#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>

int generate;

// Helper functions
#include "helper_cuda.h"
#include "helper_string.h"

#ifdef LOGS
#include "log_helper.h"
#endif

#ifdef SAFE_MALLOC
#include "safe_memory/safe_memory.h"
#endif

#define BLOCK_SIZE 16

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

typedef struct parameters_t {
	int grid_cols, grid_rows;
	float *FilesavingTemp1, *FilesavingPower1, *MatrixOut1, *GoldMatrix1;
	float *FilesavingTemp2, *FilesavingPower2, *MatrixOut2;
	float *FilesavingTemp3, *FilesavingPower3, *MatrixOut3;

	char *tfile, *pfile, *ofile;
	int nstreams;
	int sim_time;
	int pyramid_height;
	int setup_loops;
	int verbose;
	int fault_injection;
	int generate;
} parameters;

void run(int argc, char** argv);
int check_output_errors(parameters *setup_parameters, int streamIdx);

double mysecond() {
	struct timeval tp;
	struct timezone tzp;
	int i = gettimeofday(&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

void fatal(const char *s) {
	fprintf(stderr, "error: %s\n", s);
#ifdef LOGS
	if (!generate) {end_log_file();}
#endif
	exit(1);
}

void readInput(parameters *params) {
	// =================== Read all files
	int i, j;
	FILE *ftemp, *fpower, *fgold;
	char str[STR_SIZE];
	float val;
	int num_zeros = 0;
	int num_nans = 0;

	if ((ftemp = fopen(params->tfile, "r")) == 0)
		fatal("The temp file was not opened");
	if ((fpower = fopen(params->pfile, "r")) == 0)
		fatal("The power file was not opened");

	if (!(params->generate))
		if ((fgold = fopen(params->ofile, "r")) == 0)
			fatal("The gold was not opened");

	for (i = 0; i <= (params->grid_rows) - 1; i++) {
		for (j = 0; j <= (params->grid_cols) - 1; j++) {
			fgets(str, STR_SIZE, ftemp);
			if (feof(ftemp)) {
				printf("[%d,%d] size: %d ", i, j, params->grid_rows);
				fatal("not enough lines in temp file");
			}
			if ((sscanf(str, "%f", &val) != 1))
				fatal("invalid temp file format");

			// =======================
			//HARDENING AGAINST BAD BOARDS
			//-----------------------------------------------------------------------------------

			params->FilesavingTemp1[i * (params->grid_cols) + j] = val;
			params->FilesavingTemp2[i * (params->grid_cols) + j] = val;
			params->FilesavingTemp3[i * (params->grid_cols) + j] = val;

			//-----------------------------------------------------------------------------------

			if (val == 0)
				num_zeros++;
			if (isnan(val))
				num_nans++;

			fgets(str, STR_SIZE, fpower);
			if (feof(fpower))
				fatal("not enough lines in power file");
			if ((sscanf(str, "%f", &val) != 1))
				fatal("invalid power file format");
			// =======================
			//HARDENING AGAINST BAD BOARDS
			//-----------------------------------------------------------------------------------

			params->FilesavingPower1[i * (params->grid_cols) + j] = val;
			params->FilesavingPower2[i * (params->grid_cols) + j] = val;
			params->FilesavingPower3[i * (params->grid_cols) + j] = val;

			//-----------------------------------------------------------------------------------

			if (val == 0)
				num_zeros++;
			if (isnan(val))
				num_nans++;

			if (!(params->generate)) {
				fgets(str, STR_SIZE, fgold);
				if (feof(fgold))
					fatal("not enough lines in gold file");
				if ((sscanf(str, "%f", &val) != 1))
					fatal("invalid gold file format");

				// =======================
				//HARDENING AGAINST BAD BOARDS
				//-----------------------------------------------------------------------------------
				params->GoldMatrix1[i * (params->grid_cols) + j] = val;
				//-----------------------------------------------------------------------------------

			}
		}
	}

	printf("Zeros in the input: %d\n", num_zeros);
	printf("NaNs in the input: %d\n", num_nans);

	// =================== FAULT INJECTION
	if (params->fault_injection) {
		params->FilesavingTemp1[32] = 6.231235;
		printf("!!!!!!!!! Injected error: FilesavingTemp[32] = %f\n",
				params->FilesavingTemp1[32]);
	}
	// ==================================

	fclose(ftemp);
	fclose(fpower);
	if (!(params->generate))
		fclose(fgold);
}

void writeOutput(parameters *params) {
	// =================== Write output to gold file
	int i, j;
	FILE *fgold;
	char str[STR_SIZE];
	int num_zeros = 0;
	int num_nans = 0;

	if ((fgold = fopen(params->ofile, "w")) == 0)
		fatal("The gold was not opened");

	for (i = 0; i <= (params->grid_rows) - 1; i++) {
		for (j = 0; j <= (params->grid_cols) - 1; j++) {
			// =======================
			//HARDENING AGAINST BAD BOARDS
			//-----------------------------------------------------------------------------------

			if (params->MatrixOut1[i * (params->grid_cols) + j] == 0)
				num_zeros++;

			if (isnan(params->MatrixOut1[i * (params->grid_cols) + j]))
				num_nans++;

			//-----------------------------------------------------------------------------------
			sprintf(str, "%f\n",
					params->MatrixOut1[i * (params->grid_cols) + j]);
			fputs(str, fgold);
		}
	}
	fclose(fgold);
	printf("Zeros in the output: %d\n", num_zeros);
	printf("NaNs in the output: %d\n", num_nans);
}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__device__ unsigned long long int is_memory_bad = 0;

__device__ float inline read_voter(float *v1, float *v2, float *v3,
		int offset) {

	register float in1 = v1[offset];
	register float in2 = v2[offset];
	register float in3 = v3[offset];

	if (in1 == in2 || in1 == in3) {
		return in1;
	}

	if (in2 == in3) {
		return in2;
	}

	if (in1 != in2 && in2 != in3 && in1 != in3) {
		atomicAdd(&is_memory_bad, 1);
	}

	return in1;
}

__device__ float inline read_voter_2d(float v1[][BLOCK_SIZE],
		float v2[][BLOCK_SIZE], float v3[][BLOCK_SIZE], int x, int y) {
	register float in1 = v1[x][y];
	register float in2 = v2[x][y];
	register float in3 = v3[x][y];

	if (in1 == in2 || in1 == in3) {
		return in1;
	}

	if (in2 == in3) {
		return in2;
	}

	if (in1 != in2 && in2 != in3 && in1 != in3) {
		atomicAdd(&is_memory_bad, 1);
	}

	return in1;
}

__global__ void calculate_temp(int iteration,  //number of iteration
		//Hardening against bad boards
		float* power1,   //power input
		float* temp_src1,    //temperature input/output
		float* temp_dst1,    //temperature input/output
		//---------------------------------------------------
		float* power2,   //power input
		float* temp_src2,    //temperature input/output
		float* temp_dst2,    //temperature input/output
		//---------------------------------------------------
		float* power3,   //power input
		float* temp_src3,    //temperature input/output
		float* temp_dst3,    //temperature input/output
		//---------------------------------------------------

		int grid_cols,  //Col of grid
		int grid_rows,  //Row of grid
		int border_cols,  // border offset
		int border_rows,  // border offset
		float Cap,      //Capacitance
		float Rx, float Ry, float Rz, float step, float time_elapsed) {

	//----------------------------------------------------
	__shared__ float temp_on_cuda_1[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float power_on_cuda_1[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float temp_t_1[BLOCK_SIZE][BLOCK_SIZE]; // saving temporary temperature result

	//----------------------------------------------------
	__shared__ float temp_on_cuda_2[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float power_on_cuda_2[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float temp_t_2[BLOCK_SIZE][BLOCK_SIZE]; // saving temporary temperature result

	//----------------------------------------------------
	__shared__ float temp_on_cuda_3[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float power_on_cuda_3[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float temp_t_3[BLOCK_SIZE][BLOCK_SIZE]; // saving temporary temperature result
	//---------------------------------------------------

	float amb_temp = 80.0;
	float step_div_Cap;
	float Rx_1, Ry_1, Rz_1;

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	step_div_Cap = step / Cap;

	Rx_1 = 1 / Rx;
	Ry_1 = 1 / Ry;
	Rz_1 = 1 / Rz;

	// each block finally computes result for a small block
	// after N iterations.
	// it is the non-overlapping small blocks that cover
	// all the input data

	// calculate the small block size
	int small_block_rows = BLOCK_SIZE - iteration * 2;    //EXPAND_RATE
	int small_block_cols = BLOCK_SIZE - iteration * 2;    //EXPAND_RATE

	// calculate the boundary for the block according to
	// the boundary of its small block
	int blkY = small_block_rows * by - border_rows;
	int blkX = small_block_cols * bx - border_cols;
	int blkYmax = blkY + BLOCK_SIZE - 1;
	int blkXmax = blkX + BLOCK_SIZE - 1;

	// calculate the global thread coordination
	int yidx = blkY + ty;
	int xidx = blkX + tx;

	// load data if it is within the valid input range
	int loadYidx = yidx, loadXidx = xidx;
	int index = grid_cols * loadYidx + loadXidx;

	if (IN_RANGE(loadYidx, 0,
			grid_rows - 1) && IN_RANGE(loadXidx, 0, grid_cols - 1)) {

		//v1
		temp_on_cuda_1[ty][tx] = read_voter(temp_src1, temp_src2, temp_src3,
				index); // Load the temperature data from global memory to shared memory
		power_on_cuda_1[ty][tx] = read_voter(power1, power2, power3, index); // Load the power data from global memory to shared memory

		//v2
		temp_on_cuda_2[ty][tx] = read_voter(temp_src1, temp_src2, temp_src3,
				index); // Load the temperature data from global memory to shared memory
		power_on_cuda_2[ty][tx] = read_voter(power1, power2, power3, index); // Load the power data from global memory to shared memory

		//v3
		temp_on_cuda_3[ty][tx] = read_voter(temp_src1, temp_src2, temp_src3,
				index); // Load the temperature data from global memory to shared memory
		power_on_cuda_3[ty][tx] = read_voter(power1, power2, power3, index); // Load the power data from global memory to shared memory

	}
	__syncthreads();

	// effective range within this block that falls within
	// the valid range of the input data
	// used to rule out computation outside the boundary.
	int validYmin = (blkY < 0) ? -blkY : 0;
	int validYmax = (blkYmax > grid_rows - 1) ?
	BLOCK_SIZE - 1 - (blkYmax - grid_rows + 1) :
												BLOCK_SIZE - 1;
	int validXmin = (blkX < 0) ? -blkX : 0;
	int validXmax = (blkXmax > grid_cols - 1) ?
	BLOCK_SIZE - 1 - (blkXmax - grid_cols + 1) :
												BLOCK_SIZE - 1;

	int N = ty - 1;
	int S = ty + 1;
	int W = tx - 1;
	int E = tx + 1;

	N = (N < validYmin) ? validYmin : N;
	S = (S > validYmax) ? validYmax : S;
	W = (W < validXmin) ? validXmin : W;
	E = (E > validXmax) ? validXmax : E;

	bool computed;
	for (int i = 0; i < iteration; i++) {
		computed = false;
		if ( IN_RANGE(tx, i + 1, BLOCK_SIZE-i-2) &&
		IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&
		IN_RANGE(tx, validXmin, validXmax) &&
		IN_RANGE(ty, validYmin, validYmax)) {
			computed = true;
			register float calculated = read_voter_2d(temp_on_cuda_1,
					temp_on_cuda_2, temp_on_cuda_3, ty, tx)
					+ step_div_Cap
							* (read_voter_2d(power_on_cuda_1, power_on_cuda_2,
									power_on_cuda_3, ty, tx)
									+ (read_voter_2d(temp_on_cuda_1,
											temp_on_cuda_2, temp_on_cuda_3, S,
											tx)
											+ read_voter_2d(temp_on_cuda_1,
													temp_on_cuda_2,
													temp_on_cuda_3, N, tx)

											- 2.0
													* read_voter_2d(
															temp_on_cuda_1,
															temp_on_cuda_2,
															temp_on_cuda_3, ty,
															tx)) * Ry_1
									+ (read_voter_2d(temp_on_cuda_1,
											temp_on_cuda_2, temp_on_cuda_3, ty,
											E)
											+ read_voter_2d(temp_on_cuda_1,
													temp_on_cuda_2,
													temp_on_cuda_3, ty, W)
											- 2.0
													* read_voter_2d(
															temp_on_cuda_1,
															temp_on_cuda_2,
															temp_on_cuda_3, ty,
															tx)) * Rx_1
									+ (amb_temp
											- read_voter_2d(temp_on_cuda_1,
													temp_on_cuda_2,
													temp_on_cuda_3, ty, tx))
											* Rz_1);
			temp_t_1[ty][tx] = calculated;

			//--------------------------------------------------------------------------------------------------------------------------
			temp_t_2[ty][tx] = calculated;

			//--------------------------------------------------------------------------------------------------------------------------
			temp_t_3[ty][tx] = calculated;

		}
		__syncthreads();
		if (i == iteration - 1)
			break;
		if (computed) {	 //Assign the computation range

			temp_on_cuda_1[ty][tx] = read_voter_2d(temp_t_1, temp_t_2, temp_t_3,
					ty, tx);
			temp_on_cuda_2[ty][tx] = read_voter_2d(temp_t_1, temp_t_2, temp_t_3,
					ty, tx);
			temp_on_cuda_3[ty][tx] = read_voter_2d(temp_t_1, temp_t_2, temp_t_3,
					ty, tx);

		}
		__syncthreads();
	}

	// update the global memory
	// after the last iteration, only threads coordinated within the
	// small block perform the calculation and switch on ``computed''
	if (computed) {
		//--------------------------------------------------------------------------------------------------------------------------

		temp_dst1[index] = read_voter_2d(temp_t_1, temp_t_2, temp_t_3, ty, tx);
		//--------------------------------------------------------------------------------------------------------------------------

		temp_dst2[index] = read_voter_2d(temp_t_1, temp_t_2, temp_t_3, ty, tx);
		//--------------------------------------------------------------------------------------------------------------------------

		temp_dst3[index] = read_voter_2d(temp_t_1, temp_t_2, temp_t_3, ty, tx);
	}
}

/*
 compute N time steps
 */
long long int flops = 0;

int compute_tran_temp(
		//Memory triplication
		float *MatrixPower1, float *MatrixPower2, float *MatrixPower3,
		float *MatrixTemp1[2], float *MatrixTemp2[2], float *MatrixTemp3[2],
		//-------------------------------------------------------------
		int col, int row, int sim_time, int num_iterations, int blockCols,
		int blockRows, int borderCols, int borderRows, cudaStream_t stream) {
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(blockCols, blockRows);

	float grid_height = chip_height / row;
	float grid_width = chip_width / col;

	float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	float Rz = t_chip / (K_SI * grid_height * grid_width);

	float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	float step = PRECISION / max_slope;
	float t;
	float time_elapsed;
	time_elapsed = 0.001;

	int src = 1, dst = 0;
	for (t = 0; t < sim_time; t += num_iterations) {
		int temp = src;
		src = dst;
		dst = temp;
		//printf("[%d]", omp_get_thread_num());
		calculate_temp<<<dimGrid, dimBlock, 0, stream>>>(
				MIN(num_iterations, sim_time - t),
				//memory hardening --------------------------------
				MatrixPower1, MatrixTemp1[src],
				MatrixTemp1[dst], //default copy
				MatrixPower2, MatrixTemp2[src],
				MatrixTemp2[dst], //second copy
				MatrixPower3, MatrixTemp3[src],
				MatrixTemp3[dst], //third copy

				col, row, borderCols, borderRows, Cap, Rx, Ry, Rz, step,
				time_elapsed);
		flops += col * row * MIN(num_iterations, sim_time - t) * 15;
	}
	cudaStreamSynchronize(stream);
	return dst;
}

void usage(int argc, char** argv) {
	printf(
			"Usage: %s -size=N [-generate] [-sim_time=N] [-temp_file=<path>] [-power_file=<path>] [-gold_file=<path>] [-iterations=N] [-streams=N] [-debug] [-verbose]\n",
			argv[0]);
}

void getParams(int argc, char** argv, parameters *params) {
	params->nstreams = 1;
	params->sim_time = 1000;
	params->pyramid_height = 1;
	params->setup_loops = 10000000;
	params->verbose = 0;
	params->fault_injection = 0;
	params->generate = 0;
	generate = 0;

	if (argc < 2) {
		usage(argc, argv);
		exit (EXIT_FAILURE);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "size")) {
		params->grid_cols = getCmdLineArgumentInt(argc, (const char **) argv,
				"size");
		params->grid_rows = params->grid_cols;

		if ((params->grid_cols <= 0) || (params->grid_cols % 16 != 0)) {
			printf("Invalid input size given on the command-line: %d\n",
					params->grid_cols);
			exit (EXIT_FAILURE);
		}
	} else {
		usage(argc, argv);
		exit (EXIT_FAILURE);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "generate")) {
		params->generate = 1;
		generate = 1;
		printf(
				">> Output will be written to file. Only stream #0 output will be considered.\n");
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "sim_time")) {
		params->sim_time = getCmdLineArgumentInt(argc, (const char **) argv,
				"sim_time");

		if (params->sim_time < 1) {
			printf("Invalid sim_time given on the command-line: %d\n",
					params->sim_time);
			exit (EXIT_FAILURE);
		}
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "temp_file")) {
		getCmdLineArgumentString(argc, (const char **) argv, "temp_file",
				&(params->tfile));
	} else {
		params->tfile = new char[100];
		snprintf(params->tfile, 100, "temp_%i", params->grid_rows);
		printf("Using default temp_file path: %s\n", params->tfile);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "power_file")) {
		getCmdLineArgumentString(argc, (const char **) argv, "power_file",
				&(params->pfile));
	} else {
		params->pfile = new char[100];
		snprintf(params->pfile, 100, "power_%i", params->grid_rows);
		printf("Using default power_file path: %s\n", params->pfile);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "gold_file")) {
		getCmdLineArgumentString(argc, (const char **) argv, "gold_file",
				&(params->ofile));
	} else {
		params->ofile = new char[100];
		snprintf(params->ofile, 100, "gold_float_%i_%i", params->grid_rows,
				params->sim_time);
		printf("Using default gold path: %s\n", params->ofile);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "iterations")) {
		params->setup_loops = getCmdLineArgumentInt(argc, (const char **) argv,
				"iterations");
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "streams")) {
		params->nstreams = getCmdLineArgumentInt(argc, (const char **) argv,
				"streams");
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "verbose")) {
		params->verbose = 1;
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "debug")) {
		params->fault_injection = 1;
		printf("!! Will be injected an input error\n");
	}
}

int main(int argc, char** argv) {
	printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

	run(argc, argv);

	return EXIT_SUCCESS;
}

void run(int argc, char** argv) {
	//int streamIdx;
	double timestamp, globaltime;

	parameters *setupParams = (parameters *) malloc(sizeof(parameters));

	// =============== Get setup parameters from command line
	getParams(argc, argv, setupParams);
	// =======================

	// ===============  pyramid parameters
# define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline
	int borderCols = (setupParams->pyramid_height) * EXPAND_RATE / 2;
	int borderRows = (setupParams->pyramid_height) * EXPAND_RATE / 2;
	int smallBlockCol = BLOCK_SIZE - (setupParams->pyramid_height) * EXPAND_RATE;
	int smallBlockRow = BLOCK_SIZE - (setupParams->pyramid_height) * EXPAND_RATE;
	int blockCols = setupParams->grid_cols / smallBlockCol
			+ ((setupParams->grid_cols % smallBlockCol == 0) ? 0 : 1);
	int blockRows = setupParams->grid_rows / smallBlockRow
			+ ((setupParams->grid_rows % smallBlockRow == 0) ? 0 : 1);

	int size = (setupParams->grid_cols) * (setupParams->grid_rows);
	// =======================
	//HARDENING AGAINST BAD BOARDS
	//-----------------------------------------------------------------------------------
	setupParams->FilesavingTemp1 = (float *) malloc(size * sizeof(float));
	setupParams->FilesavingPower1 = (float *) malloc(size * sizeof(float));
	setupParams->MatrixOut1 = (float *) calloc(size, sizeof(float));
	setupParams->GoldMatrix1 = (float *) calloc(size, sizeof(float));

	setupParams->FilesavingTemp2 = (float *) malloc(size * sizeof(float));
	setupParams->FilesavingPower2 = (float *) malloc(size * sizeof(float));
	setupParams->MatrixOut2 = (float *) calloc(size, sizeof(float));

	setupParams->FilesavingTemp3 = (float *) malloc(size * sizeof(float));
	setupParams->FilesavingPower3 = (float *) malloc(size * sizeof(float));
	setupParams->MatrixOut3 = (float *) calloc(size, sizeof(float));

	if (!(setupParams->FilesavingPower1) || !(setupParams->FilesavingTemp1)
			|| !(setupParams->MatrixOut1) || !(setupParams->GoldMatrix1))
		fatal("unable to allocate memory");

	if (!(setupParams->FilesavingPower2) || !(setupParams->FilesavingTemp2)
			|| !(setupParams->MatrixOut2))
		fatal("unable to allocate memory");

	if (!(setupParams->FilesavingPower3) || !(setupParams->FilesavingTemp3)
			|| !(setupParams->MatrixOut3))
		fatal("unable to allocate memory");

	//-----------------------------------------------------------------------------------

	printf("cudaTripHOTSPOT\nstreams:%d size:%d pyramidHeight:%d simTime:%d\n",
			setupParams->nstreams, setupParams->grid_rows,
			setupParams->pyramid_height, setupParams->sim_time);

#ifdef LOGS
	char test_info[90];
	snprintf(test_info, 90, "streams:%d size:%d pyramidHeight:%d simTime:%d", setupParams -> nstreams, setupParams -> grid_rows, setupParams -> pyramid_height, setupParams -> sim_time);
	if (!(setupParams->generate)) start_log_file("cudaHotspot", test_info);
#endif

	timestamp = mysecond();
	readInput(setupParams);
	if (setupParams->verbose)
		printf("readInput time: %.4fs\n", mysecond() - timestamp);
	fflush (stdout);

	cudaStream_t *streams = (cudaStream_t *) malloc(
			(setupParams->nstreams) * sizeof(cudaStream_t));

	// =======================
	//HARDENING AGAINST BAD BOARDS
	//-----------------------------------------------------------------------------------
	float *MatrixTemp1[setupParams->nstreams][2],
			*MatrixPower1[setupParams->nstreams];

	float *MatrixTemp2[setupParams->nstreams][2],
			*MatrixPower2[setupParams->nstreams];

	float *MatrixTemp3[setupParams->nstreams][2],
			*MatrixPower3[setupParams->nstreams];
	//-----------------------------------------------------------------------------------
	for (int streamIdx = 0; streamIdx < (setupParams->nstreams); streamIdx++) {
		checkCudaErrors(
				cudaStreamCreateWithFlags(&(streams[streamIdx]),
						cudaStreamNonBlocking));

#ifdef SAFE_MALLOC
		// =======================
		//HARDENING AGAINST BAD BOARDS
		//-----------------------------------------------------------------------------------
		for(int z = 0; z < 2; z++) {
			safe_cuda_malloc_cover((void**)&(MatrixTemp1[streamIdx][z]), sizeof(float)*size);
//		safe_cuda_malloc_cover((void**)&(MatrixTemp1[streamIdx][1]), sizeof(float)*size);

			safe_cuda_malloc_cover((void**)&(MatrixTemp2[streamIdx][z]), sizeof(float)*size);
//		safe_cuda_malloc_cover((void**)&(MatrixTemp2[streamIdx][1]), sizeof(float)*size);

			safe_cuda_malloc_cover((void**)&(MatrixTemp3[streamIdx][z]), sizeof(float)*size);
//		safe_cuda_malloc_cover((void**)&(MatrixTemp3[streamIdx][1]), sizeof(float)*size);
		}
		//-----------------------------------------------------------------------------------

#else
		// =======================
		//HARDENING AGAINST BAD BOARDS
		//-----------------------------------------------------------------------------------
		checkCudaErrors(
				cudaMalloc((void**) &(MatrixTemp1[streamIdx][0]),
						sizeof(float) * size));
		checkCudaErrors(
				cudaMalloc((void**) &(MatrixTemp1[streamIdx][1]),
						sizeof(float) * size));

		checkCudaErrors(
				cudaMalloc((void**) &(MatrixTemp2[streamIdx][0]),
						sizeof(float) * size));
		checkCudaErrors(
				cudaMalloc((void**) &(MatrixTemp2[streamIdx][1]),
						sizeof(float) * size));

		checkCudaErrors(
				cudaMalloc((void**) &(MatrixTemp3[streamIdx][0]),
						sizeof(float) * size));
		checkCudaErrors(
				cudaMalloc((void**) &(MatrixTemp3[streamIdx][1]),
						sizeof(float) * size));

		//-----------------------------------------------------------------------------------

#endif

#ifdef SAFE_MALLOC
		// =======================
		//HARDENING AGAINST BAD BOARDS
		//-----------------------------------------------------------------------------------
		safe_cuda_malloc_cover((void**)&(MatrixPower1[streamIdx]), sizeof(float)*size);
		safe_cuda_malloc_cover((void**)&(MatrixPower2[streamIdx]), sizeof(float)*size);
		safe_cuda_malloc_cover((void**)&(MatrixPower3[streamIdx]), sizeof(float)*size);

		//-----------------------------------------------------------------------------------

#else
		// =======================
		//HARDENING AGAINST BAD BOARDS
		//-----------------------------------------------------------------------------------

		checkCudaErrors(
				cudaMalloc((void**) &(MatrixPower1[streamIdx]),
						sizeof(float) * size));

		checkCudaErrors(
				cudaMalloc((void**) &(MatrixPower2[streamIdx]),
						sizeof(float) * size));

		checkCudaErrors(
				cudaMalloc((void**) &(MatrixPower3[streamIdx]),
						sizeof(float) * size));
		//-----------------------------------------------------------------------------------

#endif

	}
	for (int loop1 = 0; loop1 < (setupParams->setup_loops); loop1++) {
		globaltime = mysecond();

		int ret[setupParams->nstreams];
//		// =======================
//		//HARDENING AGAINST BAD BOARDS
//		//-----------------------------------------------------------------------------------
//		float *MatrixTemp1[setupParams->nstreams][2],
//				*MatrixPower1[setupParams->nstreams];
//
//		float *MatrixTemp2[setupParams->nstreams][2],
//				*MatrixPower2[setupParams->nstreams];
//
//		float *MatrixTemp3[setupParams->nstreams][2],
//				*MatrixPower3[setupParams->nstreams];
//		//-----------------------------------------------------------------------------------

		timestamp = mysecond();
		for (int streamIdx = 0; streamIdx < (setupParams->nstreams);
				streamIdx++) {
//			CHANGED
//			checkCudaErrors(
//					cudaStreamCreateWithFlags(&(streams[streamIdx]),
//							cudaStreamNonBlocking));
//
//#ifdef SAFE_MALLOC
//			// =======================
//			//HARDENING AGAINST BAD BOARDS
//			//-----------------------------------------------------------------------------------
//
//			safe_cuda_malloc_cover((void**)&(MatrixTemp1[streamIdx][0]), sizeof(float)*size);
//			safe_cuda_malloc_cover((void**)&(MatrixTemp1[streamIdx][1]), sizeof(float)*size);
//
//			safe_cuda_malloc_cover((void**)&(MatrixTemp2[streamIdx][0]), sizeof(float)*size);
//			safe_cuda_malloc_cover((void**)&(MatrixTemp2[streamIdx][1]), sizeof(float)*size);
//
//			safe_cuda_malloc_cover((void**)&(MatrixTemp3[streamIdx][0]), sizeof(float)*size);
//			safe_cuda_malloc_cover((void**)&(MatrixTemp3[streamIdx][1]), sizeof(float)*size);
//			//-----------------------------------------------------------------------------------
//
//#else
//			// =======================
//			//HARDENING AGAINST BAD BOARDS
//			//-----------------------------------------------------------------------------------
//			checkCudaErrors(
//					cudaMalloc((void**) &(MatrixTemp1[streamIdx][0]),
//							sizeof(float) * size));
//			checkCudaErrors(
//					cudaMalloc((void**) &(MatrixTemp1[streamIdx][1]),
//							sizeof(float) * size));
//
//			checkCudaErrors(
//					cudaMalloc((void**) &(MatrixTemp2[streamIdx][0]),
//							sizeof(float) * size));
//			checkCudaErrors(
//					cudaMalloc((void**) &(MatrixTemp2[streamIdx][1]),
//							sizeof(float) * size));
//
//			checkCudaErrors(
//					cudaMalloc((void**) &(MatrixTemp3[streamIdx][0]),
//							sizeof(float) * size));
//			checkCudaErrors(
//					cudaMalloc((void**) &(MatrixTemp3[streamIdx][1]),
//							sizeof(float) * size));
//
//			//-----------------------------------------------------------------------------------
//
//#endif
			// =======================
			//HARDENING AGAINST BAD BOARDS
			//-----------------------------------------------------------------------------------

			cudaMemcpy(MatrixTemp1[streamIdx][0], setupParams->FilesavingTemp1,
					sizeof(float) * size, cudaMemcpyHostToDevice);
			cudaMemset(MatrixTemp2[streamIdx][1], 0.0, sizeof(float) * size);

			cudaMemcpy(MatrixTemp2[streamIdx][0], setupParams->FilesavingTemp2,
					sizeof(float) * size, cudaMemcpyHostToDevice);
			cudaMemset(MatrixTemp2[streamIdx][1], 0.0, sizeof(float) * size);

			cudaMemcpy(MatrixTemp3[streamIdx][0], setupParams->FilesavingTemp3,
					sizeof(float) * size, cudaMemcpyHostToDevice);
			cudaMemset(MatrixTemp3[streamIdx][1], 0.0, sizeof(float) * size);

			//-----------------------------------------------------------------------------------
//
//#ifdef SAFE_MALLOC
//			// =======================
//			//HARDENING AGAINST BAD BOARDS
//			//-----------------------------------------------------------------------------------
//			safe_cuda_malloc_cover((void**)&(MatrixPower1[streamIdx]), sizeof(float)*size);
//			safe_cuda_malloc_cover((void**)&(MatrixPower2[streamIdx]), sizeof(float)*size);
//			safe_cuda_malloc_cover((void**)&(MatrixPower3[streamIdx]), sizeof(float)*size);
//
//			//-----------------------------------------------------------------------------------
//
//#else
//			// =======================
//			//HARDENING AGAINST BAD BOARDS
//			//-----------------------------------------------------------------------------------
//
//			checkCudaErrors(
//					cudaMalloc((void**) &(MatrixPower1[streamIdx]),
//							sizeof(float) * size));
//
//			checkCudaErrors(
//					cudaMalloc((void**) &(MatrixPower2[streamIdx]),
//							sizeof(float) * size));
//
//			checkCudaErrors(
//					cudaMalloc((void**) &(MatrixPower3[streamIdx]),
//							sizeof(float) * size));
//			//-----------------------------------------------------------------------------------
//
//#endif
			//-----------------------------------------------------------------------------------

			cudaMemcpy(MatrixPower1[streamIdx], setupParams->FilesavingPower1,
					sizeof(float) * size, cudaMemcpyHostToDevice);

			cudaMemcpy(MatrixPower2[streamIdx], setupParams->FilesavingPower2,
					sizeof(float) * size, cudaMemcpyHostToDevice);

			cudaMemcpy(MatrixPower3[streamIdx], setupParams->FilesavingPower3,
					sizeof(float) * size, cudaMemcpyHostToDevice);

			//-----------------------------------------------------------------------------------

		}
		if (setupParams->verbose)
			printf("[Iteration #%i] GPU prepare time: %.4fs\n", loop1,
					mysecond() - timestamp);

		//printf("Start computing the transient temperature\n");
		double kernel_time = mysecond();
#ifdef LOGS
		if (!(setupParams->generate)) start_iteration();
#endif
#pragma omp parallel for
		for (int streamIdx = 0; streamIdx < (setupParams->nstreams);
				streamIdx++) {
			unsigned long long int is_memory_bad_host = 0;

			cudaMemcpyToSymbol("is_memory_bad", &is_memory_bad_host,
					sizeof(unsigned long long int), 0, cudaMemcpyHostToDevice);

			ret[streamIdx] = compute_tran_temp(
					///compute_tran_temp(
					//float *MatrixPower1, float *MatrixPower2, float *MatrixPower3,
					MatrixPower1[streamIdx], MatrixPower2[streamIdx],
					MatrixPower3[streamIdx],
					//float *MatrixTemp1[2], float *MatrixTemp2[2], float *MatrixTemp3[2],
					MatrixTemp1[streamIdx], MatrixTemp2[streamIdx],
					MatrixTemp3[streamIdx],
					//-----------------------------
					setupParams->grid_cols, setupParams->grid_rows,
					setupParams->sim_time, setupParams->pyramid_height,
					blockCols, blockRows, borderCols, borderRows,
					streams[streamIdx]);
		}
		for (int streamIdx = 0; streamIdx < (setupParams->nstreams);
				streamIdx++) {
			cudaStreamSynchronize(streams[streamIdx]);
		}
#ifdef LOGS
		if (!(setupParams->generate)) end_iteration();
#endif
		kernel_time = mysecond() - kernel_time;

		/////////// PERF
		if (setupParams->verbose) {

			double outputpersec = (double) ((setupParams->grid_rows
					* setupParams->grid_rows * setupParams->nstreams)
					/ kernel_time);
			printf("[Iteration #%i] kernel time: %.4lfs\n", loop1, kernel_time);
			printf(
					"[Iteration #%i] SIZE:%d OUTPUT/S:%f FLOPS: %f (GFLOPS: %.2f)\n",
					loop1, setupParams->grid_rows, outputpersec,
					(double) flops / kernel_time,
					(double) flops / (kernel_time * 1000000000));
		}
		flops = 0;

		//printf("Ending simulation\n");
		timestamp = mysecond();
		int kernel_errors = 0;
		if (setupParams->generate) {
			// =======================
			//HARDENING AGAINST BAD BOARDS
			//-----------------------------------------------------------------------------------
			cudaMemcpy(setupParams->MatrixOut1, MatrixTemp1[0][ret[0]],
					sizeof(float) * size, cudaMemcpyDeviceToHost);
			cudaMemcpy(setupParams->MatrixOut2, MatrixTemp2[0][ret[0]],
					sizeof(float) * size, cudaMemcpyDeviceToHost);
			cudaMemcpy(setupParams->MatrixOut3, MatrixTemp3[0][ret[0]],
					sizeof(float) * size, cudaMemcpyDeviceToHost);
			//-----------------------------------------------------------------------------------

			writeOutput(setupParams);
		} else {
			for (int streamIdx = 0; streamIdx < (setupParams->nstreams);
					streamIdx++) {
				cudaMemcpy(setupParams->MatrixOut1,
						MatrixTemp1[streamIdx][ret[streamIdx]],
						sizeof(float) * size, cudaMemcpyDeviceToHost);
				cudaMemcpy(setupParams->MatrixOut2,
						MatrixTemp2[streamIdx][ret[streamIdx]],
						sizeof(float) * size, cudaMemcpyDeviceToHost);
				cudaMemcpy(setupParams->MatrixOut3,
						MatrixTemp3[streamIdx][ret[streamIdx]],
						sizeof(float) * size, cudaMemcpyDeviceToHost);

				kernel_errors = check_output_errors(setupParams, streamIdx);
			}
			//			for (streamIdx = 0; streamIdx < setupParams->nstreams;
//					streamIdx++) {
//				memset(setupParams->MatrixOut, 0, sizeof(float) * size);
//				cudaMemcpy(setupParams->MatrixOut,
//						MatrixTemp[streamIdx][ret[streamIdx]],
//						sizeof(float) * size, cudaMemcpyDeviceToHost);
//				char error_detail[150];
//				if (memcmp(setupParams->GoldMatrix, setupParams->MatrixOut,
//						sizeof(float) * size)) {
//#pragma omp parallel for
//					for (int i = 0; i < (setupParams->grid_rows); i++) {
//						register float *ptrGold = &(setupParams->GoldMatrix[i
//								* (setupParams->grid_rows) + 0]);
//						register float *ptrOut = &(setupParams->MatrixOut[i
//								* (setupParams->grid_rows) + 0]);
//						for (int j = 0; j < (setupParams->grid_cols); j++) {
//							if (ptrGold[j] != ptrOut[j])
//#pragma omp critical
//									{
//								kernel_errors++;
//								snprintf(error_detail, 150,
//										"stream: %d, p: [%d, %d], r: %1.16e, e: %1.16e",
//										streamIdx, i, j,
//										setupParams->GoldMatrix[i
//												* (setupParams->grid_rows) + j],
//										setupParams->MatrixOut[i
//												* (setupParams->grid_rows) + j]);
//								printf(
//										"stream: %d, p: [%d, %d], r: %1.16e, e: %1.16e\n",
//										streamIdx, i, j,
//										setupParams->GoldMatrix[i
//												* (setupParams->grid_rows) + j],
//										setupParams->MatrixOut[i
//												* (setupParams->grid_rows) + j]);
//#ifdef LOGS
//								if (!(setupParams->generate)) log_error_detail(error_detail);
//#endif
//							}
//						}
//					}
//				}
//			}
#ifdef LOGS
			if (!(setupParams->generate)) log_error_count(kernel_errors);
#endif
		}

		if (setupParams->verbose)
			printf("[Iteration #%i] Gold check time: %.4fs\n", loop1,
					mysecond() - timestamp);
		if (kernel_errors != 0)
			printf("ERROR detected.\n");
		else
			printf(".");

		fflush(stdout);

//		for (streamIdx = 0; streamIdx < setupParams->nstreams; streamIdx++) {
//			// =======================
//			//HARDENING AGAINST BAD BOARDS
//			//-----------------------------------------------------------------------------------
//
//			cudaFree(MatrixPower1[streamIdx]);
//			cudaFree(MatrixTemp1[streamIdx][0]);
//			cudaFree(MatrixTemp1[streamIdx][1]);
//
//			cudaFree(MatrixPower2[streamIdx]);
//			cudaFree(MatrixTemp2[streamIdx][0]);
//			cudaFree(MatrixTemp2[streamIdx][1]);
//
//			cudaFree(MatrixPower3[streamIdx]);
//			cudaFree(MatrixTemp3[streamIdx][0]);
//			cudaFree(MatrixTemp3[streamIdx][1]);
//
////			cudaStreamDestroy(streams[streamIdx]);
//
//			//-----------------------------------------------------------------------------------
//
//		}
		if (setupParams->verbose)
			printf("[Iteration #%i] elapsed time: %.4fs\n", loop1,
					mysecond() - globaltime);
	}

	for (int streamIdx = 0; streamIdx < setupParams->nstreams; streamIdx++) {
		// =======================
		//HARDENING AGAINST BAD BOARDS
		//-----------------------------------------------------------------------------------
		cudaFree(MatrixPower1[streamIdx]);
		cudaFree(MatrixTemp1[streamIdx][0]);
		cudaFree(MatrixTemp1[streamIdx][1]);

		cudaFree(MatrixPower2[streamIdx]);
		cudaFree(MatrixTemp2[streamIdx][0]);
		cudaFree(MatrixTemp2[streamIdx][1]);

		cudaFree(MatrixPower3[streamIdx]);
		cudaFree(MatrixTemp3[streamIdx][0]);
		cudaFree(MatrixTemp3[streamIdx][1]);
		cudaStreamDestroy(streams[streamIdx]);
		//-----------------------------------------------------------------------------------
	}

#ifdef LOGS
	if (!(setupParams->generate)) end_log_file();
#endif
}

// Returns true if no errors are found. False if otherwise.
// Set votedOutput pointer to retrieve the voted matrix
int check_output_errors(parameters *setup_parameters, int streamIdx) {
	int host_errors = 0;
	int memory_errors = 0;

//#pragma omp parallel for shared(host_errors)
	for (int i = 0; i < setup_parameters->grid_rows; i++) {
		for (int j = 0; j < setup_parameters->grid_cols; j++) {
			int index = i * setup_parameters->grid_rows + j;
			register bool checkFlag = true;
			register float valGold = setup_parameters->GoldMatrix1[index];
			register float valOutput1 = setup_parameters->MatrixOut1[index];
			register float valOutput2 = setup_parameters->MatrixOut2[index];
			register float valOutput3 = setup_parameters->MatrixOut3[index];
			register float valOutput = valOutput1;

			if ((valOutput1 != valOutput2) || (valOutput2 != valOutput3)) {
#pragma omp critical
				{
					char info_detail[150];
					snprintf(info_detail, 150,
							"stream: %d, m: [%d, %d], r0: %1.16e, r1: %1.16e, r2: %1.16e",
							streamIdx, i, j, valOutput1, valOutput2,
							valOutput3);
					if (setup_parameters->verbose && (memory_errors < 10))
						printf("%s\n", info_detail);

#ifdef LOGS
					if (!generate)
					log_info_detail(info_detail);
#endif
					memory_errors += 1;
				}
				if ((valOutput1 != valOutput2) && (valOutput2 != valOutput3)) {
					// All 3 values diverge
					if (valOutput1 == valGold) {
						valOutput = valOutput1;
					} else if (valOutput2 == valGold) {
						valOutput = valOutput2;
					} else if (valOutput3 == valGold) {
						valOutput = valOutput3;
					} else {
						// NO VALUE MATCHES THE GOLD AND ALL 3 DIVERGE!
						checkFlag = false;
#pragma omp critical
						{
							char error_detail[150];
							snprintf(error_detail, 150,
									"stream: %d, f: [%d, %d], r0: %1.16e, r1: %1.16e, r2: %1.16e, e: %1.16e",
									streamIdx, i, j, valOutput1, valOutput2,
									valOutput3, valGold);

							if (setup_parameters->verbose && (host_errors < 10))
								printf("%s\n", error_detail);

#ifdef LOGS
							if (!generate)
							log_error_detail(error_detail);
#endif
							host_errors++;
						}
					}
				} else if (valOutput2 == valOutput3) {
					// Only value 0 diverge
					valOutput = valOutput2;
				} else if (valOutput1 == valOutput3) {
					// Only value 1 diverge
					valOutput = valOutput1;
				} else if (valOutput1 == valOutput2) {
					// Only value 2 diverge
					valOutput = valOutput1;
				}
			}
//			if (votedOutput != NULL)
//				votedOutput[i] = valOutput;
			// if ((fabs((tested_type_host)(valOutput-valGold)/valGold) > 1e-10)||(fabs((tested_type_host)(valOutput-valGold)/valGold) > 1e-10)) {
//			if (!(generate && (votedOutput != NULL))) {
			if (valGold != valOutput && checkFlag) {

#pragma omp critical
				{
					char error_detail[150];
					snprintf(error_detail, 150,
							"stream: %d, p: [%d, %d], r: %1.16e, e: %1.16e",
							streamIdx, i, j, valOutput, valGold);
					if (setup_parameters->verbose && (host_errors < 10))
						printf("%s\n", error_detail);
#ifdef LOGS
					if (!generate)
					log_error_detail(error_detail);
#endif
					host_errors++;

				}
			}
		}
	}

#ifdef LOGS
	unsigned long long int is_memory_bad_host = 0;
	cudaMemcpyFromSymbol(&is_memory_bad_host, "is_memory_bad",
			sizeof(unsigned long long int), 0,
			cudaMemcpyDeviceToHost);
	if(is_memory_bad_host != 0) {
		char error_info[150];

		sprintf(error_info, "For stream %d times that memory diverged %ld\n", streamIdx,
				is_memory_bad_host);
		log_info_detail(error_info);
	}
#endif

	printf("numErrors:%d\n", host_errors);

	return host_errors;
}
