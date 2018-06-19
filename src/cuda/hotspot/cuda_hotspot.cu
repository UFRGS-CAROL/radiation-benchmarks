#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>

#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef PRECISION_HALF
#include <cuda_fp16.h>
#include "half.hpp"
#endif

// Helper functions
#include "helper_cuda.h"
#include "helper_string.h"

#ifdef LOGS
#include "log_helper.h"
#endif
// The timestamp is updated on every log_helper function call.

#ifndef DEFAULT_SIM_TIME
#define DEFAULT_SIM_TIME 10000
#endif

//=========== DEFINE TESTED TYPE
#if defined(PRECISION_DOUBLE)
	const char test_precision_description[] = "double";
	typedef double tested_type;
	typedef double tested_type_host;
#elif defined(PRECISION_SINGLE)
	const char test_precision_description[] = "single";
	typedef float tested_type;
	typedef float tested_type_host;
#elif defined(PRECISION_HALF)
	const char test_precision_description[] = "half";
	typedef half tested_type;
	typedef half_float::half tested_type_host;
#else 
	#error TEST TYPE NOT DEFINED OR INCORRECT. USE PRECISION=<double|single|half>.
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
tested_type_host t_chip(0.0005);
tested_type_host chip_height(0.016);
tested_type_host chip_width(0.016);
/* ambient temperature, assuming no package at all	*/
tested_type_host amb_temp(80.0);

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

typedef struct parameters_t {
	int grid_cols, grid_rows;
	tested_type_host *in_temperature;
	tested_type_host *in_power;
	tested_type_host *out_temperature;
	tested_type_host *gold_temperature;

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

void fatal(parameters *params, const char *s) {
	fprintf(stderr, "error: %s\n", s);
#ifdef LOGS
	if (!params->generate) {end_log_file();}
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
		fatal(params, "The temp file was not opened");
	if ((fpower = fopen(params->pfile, "r")) == 0)
		fatal(params, "The power file was not opened");

	if (!(params->generate))
		if ((fgold = fopen(params->ofile, "rb")) == 0)
			fatal(params, "The gold was not opened");

	for (i = 0; i <= (params->grid_rows) - 1; i++) {
		for (j = 0; j <= (params->grid_cols) - 1; j++) {
			if (!fgets(str, STR_SIZE, ftemp)) {
				fatal(params, "not enough lines in temp file");
			}
			if (feof(ftemp)) {
				printf("[%d,%d] size: %d ", i, j, params->grid_rows);
				fatal(params, "not enough lines in temp file");
			}
			if ((sscanf(str, "%f", &val) != 1))
				fatal(params, "invalid temp file format");

			params->in_temperature[i * (params->grid_cols) + j] = tested_type_host(val);

			if (tested_type_host(val) == 0)
				num_zeros++;
			if (isnan(tested_type_host(val)))
				num_nans++;

			if (!fgets(str, STR_SIZE, fpower)) {
				fatal(params, "not enough lines in power file");
			}
			if (feof(fpower))
				fatal(params, "not enough lines in power file");
			if ((sscanf(str, "%f", &val) != 1))
				fatal(params, "invalid power file format");

			params->in_power[i * (params->grid_cols) + j] = tested_type_host(val);

			if (tested_type_host(val) == 0)
				num_zeros++;
			if (isnan(tested_type_host(val)))
				num_nans++;

			if (!(params->generate)) {
				assert( 
					fread(&(params->gold_temperature[i * (params->grid_cols) + j]), sizeof(tested_type), 1, fgold) == 1 
				);
			}
		}
	}

	printf("Zeros in the input: %d\n", num_zeros);
	printf("NaNs in the input: %d\n", num_nans);

	// =================== FAULT INJECTION
	if (params->fault_injection) {
		params->in_temperature[32] = 6.231235;
		printf("!!!!!!!!! Injected error: in_temperature[32] = %f\n",
				(double)params->in_temperature[32]);
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
	// char str[STR_SIZE];
	int num_zeros = 0;
	int num_nans = 0;

	if ((fgold = fopen(params->ofile, "wb")) == 0)
		fatal(params, "The gold was not opened");

	for (i = 0; i <= (params->grid_rows) - 1; i++) {
		for (j = 0; j <= (params->grid_cols) - 1; j++) {
			// =======================
			//HARDENING AGAINST BAD BOARDS
			//-----------------------------------------------------------------------------------

			if (params->out_temperature[i * (params->grid_cols) + j] == 0)
				num_zeros++;

			if (isnan(params->out_temperature[i * (params->grid_cols) + j]))
				num_nans++;

			//-----------------------------------------------------------------------------------
			fwrite(&(params->out_temperature[i * (params->grid_cols) + j]), sizeof(tested_type), 1, fgold);
		}
	}
	fclose(fgold);
	printf("Zeros in the output: %d\n", num_zeros);
	printf("NaNs in the output: %d\n", num_nans);
}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__global__ void calculate_temp(int iteration,  //number of iteration
		tested_type* power,   //power input
		tested_type* temp_src,    //temperature input/output
		tested_type* temp_dst,    //temperature input/output
		int grid_cols,  //Col of grid
		int grid_rows,  //Row of grid
		int border_cols,  // border offset
		int border_rows,  // border offset
		float Cap,      //Capacitance
		float Rx, float Ry, float Rz, float step, float time_elapsed) {

	//----------------------------------------------------
	__shared__ tested_type temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ tested_type power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ tested_type t_temp[BLOCK_SIZE][BLOCK_SIZE]; // saving temporary temperature result
	//----------------------------------------------------

	tested_type amb_temp(80.0);
	tested_type step_div_Cap;
	tested_type Rx_1, Ry_1, Rz_1;

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	step_div_Cap = step / Cap;

	Rx_1 = tested_type(1 / Rx);
	Ry_1 = tested_type(1 / Ry);
	Rz_1 = tested_type(1 / Rz);

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

	if (IN_RANGE(loadYidx, 0, grid_rows - 1) && 
		IN_RANGE(loadXidx, 0, grid_cols - 1)) {

		temp_on_cuda[ty][tx] = temp_src[index];
		power_on_cuda[ty][tx] = power[index];

	}
	__syncthreads();

	// effective range within this block that falls within
	// the valid range of the input data
	// used to rule out computation outside the boundary.
	int validYmin = (blkY < 0) ? -blkY : 0;
	int validYmax = (blkYmax > grid_rows - 1) ? BLOCK_SIZE - 1 - (blkYmax - grid_rows + 1) : BLOCK_SIZE - 1;
	int validXmin = (blkX < 0) ? -blkX : 0;
	int validXmax = (blkXmax > grid_cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - grid_cols + 1) : BLOCK_SIZE - 1;

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
			register tested_type calculated = 
				temp_on_cuda[ty][tx]
				+ 
				step_div_Cap
				* 
				(
					power_on_cuda[ty][tx]
					+ 
					(
						temp_on_cuda[S][tx]
						+ 
						temp_on_cuda[N][tx]
						- 
						tested_type(2.0)
						* 
						temp_on_cuda[ty][tx]
					) 
					* 
					Ry_1
					+ 
					(
						temp_on_cuda[ty][E]
						+ 
						temp_on_cuda[ty][W]
						- 
						tested_type(2.0)
						* 
						temp_on_cuda[ty][tx]
					) 
					* 
					Rx_1
					+ 
					(
						amb_temp
						- 
						temp_on_cuda[ty][tx]
					)
					* 
					Rz_1
				);
			t_temp[ty][tx] = calculated;
		}
		__syncthreads();
		if (i == iteration - 1)
			break;

		if (computed) {	 //Assign the computation range
			temp_on_cuda[ty][tx] = t_temp[ty][tx];
		}
		__syncthreads();
	}

	// update the global memory
	// after the last iteration, only threads coordinated within the
	// small block perform the calculation and switch on ``computed''
	if (computed) {
		temp_dst[index] = t_temp[ty][tx];
	}
}

/*
 compute N time steps
 */
long long int flops = 0;

int compute_tran_temp(
		tested_type_host *MatrixPower,
		tested_type_host *MatrixTemp[2],
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
				(tested_type*)MatrixPower, 
				(tested_type*)MatrixTemp[src],
				(tested_type*)MatrixTemp[dst],
				col, row, borderCols, borderRows, Cap, Rx, Ry, Rz, step,
				time_elapsed);
		flops += col * row * MIN(num_iterations, sim_time - t) * 15;
	}
	cudaStreamSynchronize(stream);
	return dst;
}

void usage(int argc, char** argv) {
	printf(
			"Usage: %s [-size=N] [-generate] [-sim_time=N] [-input_temp=<path>] [-input_power=<path>] [-gold_temp=<path>] [-iterations=N] [-streams=N] [-debug] [-verbose]\n",
			argv[0]);
}

void getParams(int argc, char** argv, parameters *params) {
	params->nstreams = 1;
	params->sim_time = DEFAULT_SIM_TIME;
	params->pyramid_height = 1;
	params->setup_loops = 10000000;
	params->verbose = 0;
	params->fault_injection = 0;
	params->generate = 0;

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

	if (checkCmdLineFlag(argc, (const char **) argv, "input_temp")) {
		getCmdLineArgumentString(argc, (const char **) argv, "input_temp",
				&(params->tfile));
	} else {
		params->tfile = new char[100];
		snprintf(params->tfile, 100, "temp_%i", params->grid_rows);
		printf("Using default input_temp path: %s\n", params->tfile);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "input_power")) {
		getCmdLineArgumentString(argc, (const char **) argv, "input_power",
				&(params->pfile));
	} else {
		params->pfile = new char[100];
		snprintf(params->pfile, 100, "power_%i", params->grid_rows);
		printf("Using default input_power path: %s\n", params->pfile);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "gold_temp")) {
		getCmdLineArgumentString(argc, (const char **) argv, "gold_temp",
				&(params->ofile));
	} else {
		params->ofile = new char[100];
		snprintf(params->ofile, 100, "gold_temp_%s_%i_%i", test_precision_description, params->grid_rows,
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
	setupParams->in_temperature = (tested_type_host *) malloc(size * sizeof(tested_type));
	setupParams->in_power = (tested_type_host *) malloc(size * sizeof(tested_type));
	setupParams->out_temperature = (tested_type_host *) calloc(size, sizeof(tested_type));
	setupParams->gold_temperature = (tested_type_host *) calloc(size, sizeof(tested_type));

	if (!(setupParams->in_power) || !(setupParams->in_temperature)
			|| !(setupParams->out_temperature) || !(setupParams->gold_temperature))
		fatal(setupParams, "unable to allocate memory");

	//-----------------------------------------------------------------------------------

	char test_info[150];
	char test_name[90];
	snprintf(test_info, 150, "streams:%d precision:%s size:%d pyramidHeight:%d simTime:%d", setupParams -> nstreams, test_precision_description, setupParams -> grid_rows, setupParams -> pyramid_height, setupParams -> sim_time);
	snprintf(test_name, 90, "cuda_hotspot_%s", test_precision_description);
#ifdef LOGS
	if (!(setupParams->generate)) start_log_file(test_name, test_info);
#endif
	printf("\n=================================\n%s\n%s\n=================================\n\n", test_name, test_info);

	timestamp = mysecond();
	readInput(setupParams);
	if (setupParams->verbose)
		printf("readInput time: %.4fs\n", mysecond() - timestamp);
	fflush (stdout);

	cudaStream_t *streams = (cudaStream_t *) malloc((setupParams->nstreams) * sizeof(cudaStream_t));

	tested_type_host *MatrixTemp[setupParams->nstreams][2];
	tested_type_host *MatrixPower[setupParams->nstreams];
			
	for (int streamIdx = 0; streamIdx < (setupParams->nstreams); streamIdx++) {
		checkCudaErrors(
				cudaStreamCreateWithFlags(&(streams[streamIdx]),
						cudaStreamNonBlocking));

		checkCudaErrors(
				cudaMalloc((void**) &(MatrixTemp[streamIdx][0]),
						sizeof(tested_type) * size));
		checkCudaErrors(
				cudaMalloc((void**) &(MatrixTemp[streamIdx][1]),
						sizeof(tested_type) * size));

		checkCudaErrors(
				cudaMalloc((void**) &(MatrixPower[streamIdx]),
						sizeof(tested_type) * size));

	}

	// ====================== MAIN BENCHMARK CYCLE ======================
	for (int loop1 = 0; loop1 < (setupParams->setup_loops); loop1++) {
		if (setupParams->verbose) 
			printf("======== Iteration #%06u ========\n", loop1);
		globaltime = mysecond();

		// ============ PREPARE ============
		int ret[setupParams->nstreams];
		timestamp = mysecond();
		for (int streamIdx = 0; streamIdx < (setupParams->nstreams); streamIdx++) {

			// Setup inputs (Power and Temperature)
			cudaMemcpy(MatrixTemp[streamIdx][0], setupParams->in_temperature, sizeof(tested_type) * size, cudaMemcpyHostToDevice);

			cudaMemcpy(MatrixPower[streamIdx], setupParams->in_power, sizeof(tested_type) * size, cudaMemcpyHostToDevice);

			// Setup output (Temperature)
			cudaMemset(MatrixTemp[streamIdx][1], 0.0, sizeof(tested_type) * size);

		}
		if (setupParams->verbose)
			printf("GPU prepare time: %.4fs\n", mysecond() - timestamp);

		// ============ COMPUTE ============
		double kernel_time = mysecond();
#ifdef LOGS
		if (!(setupParams->generate)) start_iteration();
#endif
#pragma omp parallel for
		for (int streamIdx = 0; streamIdx < (setupParams->nstreams);
				streamIdx++) {
			ret[streamIdx] = compute_tran_temp(
					MatrixPower[streamIdx],
					MatrixTemp[streamIdx],
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

		// ============ MEASURE PERFORMANCE ============
		if (setupParams->verbose) {

			double outputpersec = (double) ((setupParams->grid_rows
					* setupParams->grid_rows * setupParams->nstreams)
					/ kernel_time);
			printf("Kernel time: %.4lfs\n", kernel_time);
			printf("Performance - SIZE:%d OUTPUT/S:%f FLOPS: %f (GFLOPS: %.2f)\n",
					setupParams->grid_rows, outputpersec,
					(double) flops / kernel_time,
					(double) flops / (kernel_time * 1000000000));
		}
		flops = 0;

		// ============ VALIDATE OUTPUT ============
		timestamp = mysecond();
		int kernel_errors = 0;
		if (setupParams->generate) {
			cudaMemcpy(setupParams->out_temperature, MatrixTemp[0][ret[0]],
					sizeof(tested_type) * size, cudaMemcpyDeviceToHost);

			writeOutput(setupParams);
		} else {
			for (int streamIdx = 0; streamIdx < (setupParams->nstreams);
					streamIdx++) {
				cudaMemcpy(setupParams->out_temperature,
						MatrixTemp[streamIdx][ret[streamIdx]],
						sizeof(tested_type) * size, cudaMemcpyDeviceToHost);

				check_output_errors(setupParams, streamIdx);
			}
		}

		if (setupParams->verbose)
			printf("Gold check time: %.4fs\n", mysecond() - timestamp);
		if ((kernel_errors != 0) && !(setupParams->verbose))
			printf(".");
		
		double iteration_time = mysecond() - globaltime;
		if (setupParams->verbose)
			printf("Iteration time: %.4fs (%3.1f%% Device)\n", iteration_time, (kernel_time / iteration_time) * 100.0);
		if (setupParams->verbose)
			printf("===================================\n");

		fflush(stdout);
	}

	for (int streamIdx = 0; streamIdx < setupParams->nstreams; streamIdx++) {
		cudaFree(MatrixPower[streamIdx]);
		cudaFree(MatrixTemp[streamIdx][0]);
		cudaFree(MatrixTemp[streamIdx][1]);

		cudaStreamDestroy(streams[streamIdx]);
	}

#ifdef LOGS
	if (!(setupParams->generate)) end_log_file();
#endif
}

// Returns true if no errors are found. False if otherwise.
int check_output_errors(parameters *setup_parameters, int streamIdx) {
	int host_errors = 0;

#pragma omp parallel for shared(host_errors)
	for (int i = 0; i < setup_parameters->grid_rows; i++) {
		for (int j = 0; j < setup_parameters->grid_cols; j++) {
			int index = i * setup_parameters->grid_rows + j;
			
			register tested_type_host valGold = setup_parameters->gold_temperature[index];
			register tested_type_host valOutput = setup_parameters->out_temperature[index];

			
			if (valGold != valOutput) {
#pragma omp critical
				{
					char error_detail[150];
					snprintf(error_detail, 150,
							"stream: %d, p: [%d, %d], r: %1.20e, e: %1.20e",
							streamIdx, i, j, 
							(double)valOutput, (double)valGold);
					if (setup_parameters->verbose && (host_errors < 10))
						printf("%s\n", error_detail);
#ifdef LOGS
					if (!setup_parameters->generate)
						log_error_detail(error_detail);
#endif
					host_errors++;

				}
			}
		}
	}

#ifdef LOGS
	if (!setup_parameters->generate) {
		log_error_count(host_errors);
	}
#endif
	if ((host_errors != 0) && (!setup_parameters->verbose)) printf("#");
	if ((host_errors != 0) && (setup_parameters->verbose)) printf("Output errors: %d\n", host_errors);

	return (host_errors == 0);
}
