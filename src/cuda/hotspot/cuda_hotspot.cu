#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <vector>

#ifdef USE_OMP
#include <omp.h>
#endif

//#ifdef PRECISION_HALF
//#include <cuda_fp16.h>
//#include "half.hpp"
//#endif

// Helper functions
//#include "helper_cuda.h"
#include "helper_string.h"

#include "generic_log.h"
#include "cuda_utils.h"
#include "multi_compiler_analysis.h"

// The timestamp is updated on every log_helper function call.

#ifndef DEFAULT_SIM_TIME
#define DEFAULT_SIM_TIME 10000
#endif

#define BLOCK_SIZE 32

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

# define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

template<typename float_type>
struct test_arrays {
	std::vector<float_type> in_temperature;
	std::vector<float_type> in_power;
	std::vector<float_type> out_temperature;
	std::vector<float_type> gold_temperature;

	void allocate(size_t size) {
		this->in_temperature = std::vector < float_type > (size, 0); //(float_type *) malloc(size * sizeof(float_type));
		this->in_power = std::vector < float_type > (size, 0); //(float_type *) malloc(size * sizeof(float_type));
		this->out_temperature = std::vector < float_type > (size, 0); //(float_type *) calloc(size, sizeof(float_type));
		this->gold_temperature = std::vector < float_type > (size, 0); //(float_type *) calloc(size, sizeof(float_type));
	}
};

struct parameters {
	int grid_cols, grid_rows;
	std::string tested_type;
	std::string tfile, pfile, ofile;
	int nstreams;
	int sim_time;
	int pyramid_height;
	int setup_loops;
	int verbose;
	int fault_injection;
	int generate;

	void usage(int argc, char** argv) {
		printf("Usage: %s [-size=N] [-generate] [-sim_time=N] [-input_temp=<path>]"
				" [-input_power=<path>] [-gold_temp=<path>] [-iterations=N] [-streams=N]"
				" [-debug] [-verbose] [-precision=<float|double>]\n", argv[0]);
	}

	parameters(int argc, char** argv) {
		this->nstreams = 1;
		this->sim_time = DEFAULT_SIM_TIME;
		this->pyramid_height = 1;
		this->setup_loops = 10000000;
		this->verbose = 0;
		this->fault_injection = 0;
		this->generate = 0;

		if (argc < 2) {
			usage(argc, argv);
			exit (EXIT_FAILURE);
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "precision")) {
			char *tmp;
			getCmdLineArgumentString(argc, (const char **) argv, "precision", &(tmp));
			this->tested_type = std::string(tmp);
			printf("Precision: %s\n", this->tested_type.c_str());
		} else {
			this->tested_type = "float";
			printf("Using default precision float\n");
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "input_temp")) {
			char *tmp;
			getCmdLineArgumentString(argc, (const char **) argv, "input_temp", &(tmp));
			this->tfile = std::string(tmp);
			printf("Input temp: %s\n", this->tfile.c_str());
		} else {
			this->tfile = "temp_" + std::to_string(this->grid_rows);
			printf("Using default input_temp path: %s\n", this->tfile.c_str());
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "input_power")) {
			char *tmp;
			getCmdLineArgumentString(argc, (const char **) argv, "input_power", &(tmp));
			this->pfile = std::string(tmp);
			printf("Input power: %s\n", this->pfile.c_str());
		} else {
			this->pfile = "power_" + std::to_string(this->grid_rows);
			printf("Using default input_power path: %s\n", this->pfile.c_str());
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "gold_temp")) {
			char *tmp;
			getCmdLineArgumentString(argc, (const char **) argv, "gold_temp", &(tmp));
			this->ofile = std::string(tmp);
			printf("Gold/output file: %s\n", this->ofile.c_str());
		} else {
			this->ofile = "gold_temp_" + this->tested_type + "_" + std::to_string(this->grid_rows)
					+ "_" + std::to_string(this->sim_time);
			printf("Using default gold path: %s\n", this->ofile.c_str());
		}

		//-------------------------------------------------------

		if (checkCmdLineFlag(argc, (const char **) argv, "size")) {
			this->grid_cols = getCmdLineArgumentInt(argc, (const char **) argv, "size");
			this->grid_rows = this->grid_cols;

			if ((this->grid_cols <= 0) || (this->grid_cols % 16 != 0)) {
				printf("Invalid input size given on the command-line: %d\n", this->grid_cols);
				exit (EXIT_FAILURE);
			}
		} else {
			usage(argc, argv);
			exit (EXIT_FAILURE);
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "sim_time")) {
			this->sim_time = getCmdLineArgumentInt(argc, (const char **) argv, "sim_time");

			if (this->sim_time < 1) {
				printf("Invalid sim_time given on the command-line: %d\n", this->sim_time);
				exit (EXIT_FAILURE);
			}
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "iterations")) {
			this->setup_loops = getCmdLineArgumentInt(argc, (const char **) argv, "iterations");
			printf("Iterations: %d\n", this->setup_loops);
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "streams")) {
			this->nstreams = getCmdLineArgumentInt(argc, (const char **) argv, "streams");
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "verbose")) {
			this->verbose = 1;
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "debug")) {
			this->fault_injection = 1;
			printf("!! Will be injected an input error\n");
		}

		if (checkCmdLineFlag(argc, (const char **) argv, "generate")) {
			this->generate = 1;
			this->setup_loops = 1;
			printf("Output will be written to file. Only stream #0 output will be considered.\n");
		}

	}

};
// parameters;

void fatal(const char *s) {
	fprintf(stderr, "error: %s\n", s);
	exit(1);
}

template<typename float_type>
void readInput(parameters& params, test_arrays<float_type>& arrays) {
	// =================== Read all files
	int i, j;
	FILE *ftemp, *fpower, *fgold;
	char str[STR_SIZE];
	float val;
	int num_zeros = 0;
	int num_nans = 0;

	if ((ftemp = fopen(params.tfile.c_str(), "r")) == 0)
		fatal("The temp file was not opened");
	if ((fpower = fopen(params.pfile.c_str(), "r")) == 0)
		fatal("The power file was not opened");

	if (!(params.generate))
		if ((fgold = fopen(params.ofile.c_str(), "rb")) == 0)
			fatal("The gold was not opened");

	for (i = 0; i <= (params.grid_rows) - 1; i++) {
		for (j = 0; j <= (params.grid_cols) - 1; j++) {
			if (!fgets(str, STR_SIZE, ftemp)) {
				fatal("not enough lines in temp file");
			}
			if (feof(ftemp)) {
				printf("[%d,%d] size: %d ", i, j, params.grid_rows);
				fatal("not enough lines in temp file");
			}
			if ((sscanf(str, "%f", &val) != 1))
				fatal("invalid temp file format");

			arrays.in_temperature[i * (params.grid_cols) + j] = float_type(val);

			if (float_type(val) == 0)
				num_zeros++;
			if (isnan(float_type(val)))
				num_nans++;

			if (!fgets(str, STR_SIZE, fpower)) {
				fatal("not enough lines in power file");
			}
			if (feof(fpower))
				fatal("not enough lines in power file");
			if ((sscanf(str, "%f", &val) != 1))
				fatal("invalid power file format");

			arrays.in_power[i * (params.grid_cols) + j] = float_type(val);

			if (float_type(val) == 0)
				num_zeros++;
			if (isnan(float_type(val)))
				num_nans++;

			if (!(params.generate)) {
				assert(
						fread(&(arrays.gold_temperature[i * (params.grid_cols) + j]),
								sizeof(float_type), 1, fgold) == 1);
			}
		}
	}

	printf("Zeros in the input: %d\n", num_zeros);
	printf("NaNs in the input: %d\n", num_nans);

	// =================== FAULT INJECTION
	if (params.fault_injection) {
		arrays.in_temperature[32] = 6.231235;
		printf("!!!!!!!!! Injected error: in_temperature[32] = %f\n",
				(double) arrays.in_temperature[32]);
	}
	// ==================================

	fclose(ftemp);
	fclose(fpower);
	if (!(params.generate))
		fclose(fgold);
}

template<typename float_type>
void writeOutput(parameters& params, test_arrays<float_type>& arrays) {
	// =================== Write output to gold file
	int i, j;
	FILE *fgold;
	// char str[STR_SIZE];
	int num_zeros = 0;
	int num_nans = 0;

	if ((fgold = fopen(params.ofile.c_str(), "wb")) == 0)
		fatal("The gold was not opened");

	for (i = 0; i <= (params.grid_rows) - 1; i++) {
		for (j = 0; j <= (params.grid_cols) - 1; j++) {
			// =======================
			//HARDENING AGAINST BAD BOARDS
			//-----------------------------------------------------------------------------------

			if (arrays.out_temperature[i * (params.grid_cols) + j] == 0)
				num_zeros++;

			if (isnan(arrays.out_temperature[i * (params.grid_cols) + j]))
				num_nans++;

			//-----------------------------------------------------------------------------------
			fwrite(&(arrays.out_temperature[i * (params.grid_cols) + j]), sizeof(float_type), 1,
					fgold);
		}
	}
	fclose(fgold);
	printf("Zeros in the output: %d\n", num_zeros);
	printf("NaNs in the output: %d\n", num_nans);
}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

template<typename float_type>
__global__ void calculate_temp(int iteration,  //number of iteration
		float_type* power,   //power input
		float_type* temp_src,    //temperature input/output
		float_type* temp_dst,    //temperature input/output
		int grid_cols,  //Col of grid
		int grid_rows,  //Row of grid
		int border_cols,  // border offset
		int border_rows,  // border offset
		float_type Cap,      //Capacitance
		float_type Rx, float_type Ry, float_type Rz, float_type step, float_type time_elapsed) {

	//----------------------------------------------------
	__shared__ float_type temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float_type power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float_type t_temp[BLOCK_SIZE][BLOCK_SIZE]; // saving temporary temperature result
	//----------------------------------------------------

	float_type amb_temp(80.0);
	float_type step_div_Cap;
	float_type Rx_1, Ry_1, Rz_1;

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	step_div_Cap = step / Cap;

	Rx_1 = float_type(1.0 / Rx);
	Ry_1 = float_type(1.0 / Ry);
	Rz_1 = float_type(1.0 / Rz);

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

	if (IN_RANGE(loadYidx, 0, grid_rows - 1) && IN_RANGE(loadXidx, 0, grid_cols - 1)) {

		temp_on_cuda[ty][tx] = temp_src[index];
		power_on_cuda[ty][tx] = power[index];

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
			float_type calculated = temp_on_cuda[ty][tx]
					+ step_div_Cap
							* (power_on_cuda[ty][tx]
									+ (temp_on_cuda[S][tx] + temp_on_cuda[N][tx]
											- float_type(2.0) * temp_on_cuda[ty][tx]) * Ry_1
									+ (temp_on_cuda[ty][E] + temp_on_cuda[ty][W]
											- float_type(2.0) * temp_on_cuda[ty][tx]) * Rx_1
									+ (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
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

template<typename float_type>
int compute_tran_temp(float_type *MatrixPower, float_type *MatrixTemp[2], int col, int row,
		int sim_time, int num_iterations, int blockCols, int blockRows, int borderCols,
		int borderRows, cudaStream_t stream) {

	///* chip parameters	*/
	float_type t_chip(0.0005);
	float_type chip_height(0.016);
	float_type chip_width(0.016);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(blockCols, blockRows);

	float_type grid_height = chip_height / row;
	float_type grid_width = chip_width / col;

	float_type Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	float_type Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	float_type Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	float_type Rz = t_chip / (K_SI * grid_height * grid_width);

	float_type max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	float_type step = PRECISION / max_slope;
	float_type t;
	float_type time_elapsed;
	time_elapsed = 0.001;

	int src = 1, dst = 0;
	for (t = 0; t < sim_time; t += num_iterations) {
		int temp = src;
		src = dst;
		dst = temp;
		//printf("[%d]", omp_get_thread_num());
		calculate_temp<<<dimGrid, dimBlock, 0, stream>>>(MIN(num_iterations, sim_time - t),
				(float_type*) MatrixPower, (float_type*) MatrixTemp[src],
				(float_type*) MatrixTemp[dst], col, row, borderCols, borderRows, Cap, Rx, Ry, Rz,
				step, time_elapsed);
		flops += col * row * MIN(num_iterations, sim_time - t) * 15;
	}
//	cudaStreamSynchronize(stream);
	return dst;
}

// Returns true if no errors are found. False if otherwise.
template<typename float_type>
int check_output_errors(parameters setup_parameters, test_arrays<float_type>& arrays, int streamIdx,
		rad::Log& log) {
	int host_errors = 0;

#pragma omp parallel for shared(host_errors)
	for (int i = 0; i < setup_parameters.grid_rows; i++) {
		for (int j = 0; j < setup_parameters.grid_cols; j++) {
			int index = i * setup_parameters.grid_rows + j;

			auto valGold = arrays.gold_temperature[index];
			auto valOutput = arrays.out_temperature[index];

			if (valGold != valOutput) {
#pragma omp critical
				{
					char error_detail[150];
					snprintf(error_detail, 150, "stream: %d, p: [%d, %d], r: %1.20e, e: %1.20e",
							streamIdx, i, j, (double) valOutput, (double) valGold);
					if (setup_parameters.verbose && (host_errors < 10))
						printf("%s\n", error_detail);
					if (!setup_parameters.generate) {
						log.log_error_detail(std::string(error_detail));
					}
					host_errors++;

				}
			}
		}
	}

	if (!setup_parameters.generate) {
		log.update_errors();
	}
	if ((host_errors != 0) && (!setup_parameters.verbose))
		printf("#");
	if ((host_errors != 0) && (setup_parameters.verbose))
		printf("Output errors: %d\n", host_errors);

	return (host_errors == 0);
}

template<typename float_type>
void run(parameters& params, test_arrays<float_type>& arrays) {
	//int streamIdx;
	double timestamp, globaltime;

	// ===============  pyramid parameters
	int borderCols = (params.pyramid_height) * EXPAND_RATE / 2;
	int borderRows = (params.pyramid_height) * EXPAND_RATE / 2;
	int smallBlockCol = BLOCK_SIZE - (params.pyramid_height) * EXPAND_RATE;
	int smallBlockRow = BLOCK_SIZE - (params.pyramid_height) * EXPAND_RATE;
	int blockCols = params.grid_cols / smallBlockCol
			+ ((params.grid_cols % smallBlockCol == 0) ? 0 : 1);
	int blockRows = params.grid_rows / smallBlockRow
			+ ((params.grid_rows % smallBlockRow == 0) ? 0 : 1);

	int size = (params.grid_cols) * (params.grid_rows);
	// =======================
	//HARDENING AGAINST BAD BOARDS
	//-----------------------------------------------------------------------------------
	arrays.allocate(size);
//
//	if (!(arrays.in_power) || !(arrays.in_temperature) || !(arrays.out_temperature)
//			|| !(arrays.gold_temperature))
//		fatal("unable to allocate memory");

	//-----------------------------------------------------------------------------------
	auto test_info = "streams:" + std::to_string(params.nstreams);
	test_info += " precision:" + params.tested_type;
	test_info += " size:" + std::to_string(params.grid_rows);
	test_info += " pyramidHeight:" + std::to_string(params.pyramid_height);
	test_info += " simTime:" + std::to_string(params.sim_time);
	test_info += " nvcc_version:" + rad::get_cuda_cc_version();
	test_info += " nvcc_optimization_flags:" + rad::extract_nvcc_opt_flags_str();
	;

	rad::Log log("cuda_hotspot", test_info);
	if (params.verbose) {
		std::cout << log << std::endl;

	}
	printf("\n=================================\ncuda_hotspot"
			"\n%s\n=================================\n\n", test_info.c_str());

	timestamp = rad::mysecond();
	readInput(params, arrays);
	if (params.verbose)
		printf("readInput time: %.4fs\n", rad::mysecond() - timestamp);
	fflush (stdout);

	cudaStream_t *streams = (cudaStream_t *) malloc((params.nstreams) * sizeof(cudaStream_t));

	float_type *MatrixTemp[params.nstreams][2];
	float_type *MatrixPower[params.nstreams];

	for (int streamIdx = 0; streamIdx < (params.nstreams); streamIdx++) {
		rad::checkFrameworkErrors(
				cudaStreamCreateWithFlags(&(streams[streamIdx]), cudaStreamNonBlocking));

		rad::checkFrameworkErrors(
				cudaMalloc((void**) &(MatrixTemp[streamIdx][0]), sizeof(float_type) * size));
		rad::checkFrameworkErrors(
				cudaMalloc((void**) &(MatrixTemp[streamIdx][1]), sizeof(float_type) * size));

		rad::checkFrameworkErrors(
				cudaMalloc((void**) &(MatrixPower[streamIdx]), sizeof(float_type) * size));

	}

	// ====================== MAIN BENCHMARK CYCLE ======================
	for (int loop1 = 0; loop1 < (params.setup_loops); loop1++) {
		if (params.verbose)
			printf("======== Iteration #%06u ========\n", loop1);
		globaltime = rad::mysecond();

		// ============ PREPARE ============
		int ret[params.nstreams];
		timestamp = rad::mysecond();
		for (int streamIdx = 0; streamIdx < (params.nstreams); streamIdx++) {

			// Setup inputs (Power and Temperature)
			rad::checkFrameworkErrors(
					cudaMemcpy(MatrixTemp[streamIdx][0], arrays.in_temperature.data(),
							sizeof(float_type) * size, cudaMemcpyHostToDevice));

			rad::checkFrameworkErrors(
					cudaMemcpy(MatrixPower[streamIdx], arrays.in_power.data(),
							sizeof(float_type) * size, cudaMemcpyHostToDevice));

			// Setup output (Temperature)
			rad::checkFrameworkErrors(
					cudaMemset(MatrixTemp[streamIdx][1], 0.0, sizeof(float_type) * size));

		}
		if (params.verbose)
			printf("GPU prepare time: %.4fs\n", rad::mysecond() - timestamp);

		// ============ COMPUTE ============
		double kernel_time = rad::mysecond();
		log.start_iteration();
#pragma omp parallel for
		for (int streamIdx = 0; streamIdx < (params.nstreams); streamIdx++) {
			ret[streamIdx] = compute_tran_temp(MatrixPower[streamIdx], MatrixTemp[streamIdx],
					params.grid_cols, params.grid_rows, params.sim_time, params.pyramid_height,
					blockCols, blockRows, borderCols, borderRows, streams[streamIdx]);
		}
		for (int streamIdx = 0; streamIdx < (params.nstreams); streamIdx++) {
			rad::checkFrameworkErrors(cudaStreamSynchronize(streams[streamIdx]));
		}
		rad::checkFrameworkErrors(cudaGetLastError());
		log.end_iteration();
		kernel_time = rad::mysecond() - kernel_time;

		// ============ MEASURE PERFORMANCE ============
		if (params.verbose) {

			double outputpersec = (double) ((params.grid_rows * params.grid_rows * params.nstreams)
					/ kernel_time);
			printf("Kernel time: %.4lfs\n", kernel_time);
			printf("Performance - SIZE:%d OUTPUT/S:%f FLOPS: %f (GFLOPS: %.2f)\n", params.grid_rows,
					outputpersec, (double) flops / kernel_time,
					(double) flops / (kernel_time * 1000000000));
		}
		flops = 0;

		// ============ VALIDATE OUTPUT ============
		timestamp = rad::mysecond();
		int kernel_errors = 0;
		if (params.generate) {
			rad::checkFrameworkErrors(
					cudaMemcpy(arrays.out_temperature.data(), MatrixTemp[0][ret[0]],
							sizeof(float_type) * size, cudaMemcpyDeviceToHost));

			writeOutput(params, arrays);
		} else {
			for (int streamIdx = 0; streamIdx < (params.nstreams); streamIdx++) {
				rad::checkFrameworkErrors(
						cudaMemcpy(arrays.out_temperature.data(),
								MatrixTemp[streamIdx][ret[streamIdx]], sizeof(float_type) * size,
								cudaMemcpyDeviceToHost));

				check_output_errors(params, arrays, streamIdx, log);
			}
		}

		if (params.verbose)
			printf("Gold check time: %.4fs\n", rad::mysecond() - timestamp);
		if ((kernel_errors != 0) && !(params.verbose))
			printf(".");

		double iteration_time = rad::mysecond() - globaltime;
		if (params.verbose)
			printf("Iteration time: %.4fs (%3.1f%% Device)\n", iteration_time,
					(kernel_time / iteration_time) * 100.0);
		if (params.verbose)
			printf("===================================\n");

		fflush(stdout);
	}

	for (int streamIdx = 0; streamIdx < params.nstreams; streamIdx++) {
		rad::checkFrameworkErrors(cudaFree(MatrixPower[streamIdx]));
		rad::checkFrameworkErrors(cudaFree(MatrixTemp[streamIdx][0]));
		rad::checkFrameworkErrors(cudaFree(MatrixTemp[streamIdx][1]));
		rad::checkFrameworkErrors(cudaStreamDestroy(streams[streamIdx]));
	}
}

int main(int argc, char** argv) {
	printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

	parameters setup_parameters(argc, argv);

	if (setup_parameters.tested_type == "half") {
		fatal("Half precision not implemented");
	} else if (setup_parameters.tested_type == "double") {
		test_arrays<double> arrays;
		run(setup_parameters, arrays);
	} else {
		test_arrays<float> arrays;
		run(setup_parameters, arrays);
	}
	return EXIT_SUCCESS;
}
