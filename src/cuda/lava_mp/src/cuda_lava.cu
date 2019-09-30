//============================================================================
//	UPDATE
//============================================================================

//	14 APR 2011 Lukasz G. Szafaryn
//  2014-2018 Caio Lunardi
//  2018 Fernando Fernandes dos Santos

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdbool.h>

#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef LOGS

#include "log_helper.h"

#ifdef BUILDPROFILER

#ifdef FORJETSON
#include "include/JTX2Inst.h"
#define OBJTYPE JTX2Inst
#else
#include "include/NVMLWrapper.h"
#define OBJTYPE NVMLWrapper
#endif // FORJETSON

#endif // BUILDPROFILER

#endif // LOGHELPER

#include <cuda_fp16.h>
#include "cuda_utils.h"

#include "Parameters.h"
#include "Log.h"
#include "types.h"
#include "common.h"
#include "nondmr_kernels.h"

//=============================================================================
//	DEFINE TEST TYPE WITH INTRINSIC TYPES
//=============================================================================
#if defined(PRECISION_DOUBLE)

const char test_precision_description[] = "double";
typedef double tested_type;
typedef double tested_type;

#elif defined(PRECISION_SINGLE)

const char test_precision_description[] = "single";
typedef float tested_type;
typedef float tested_type;

#elif defined(PRECISION_HALF)

#define H2_DOT(A,B) (__hfma2((A.x), (B.x), __hfma2((A.y), (B.y), __hmul2((A.z), (B.z)))))

const char test_precision_description[] = "half";
typedef half tested_type;
//typedef half_float::half tested_type;
typedef half tested_type;

//#else
//#error TEST TYPE NOT DEFINED OR INCORRECT. USE PRECISION=<double|single|half>.
#endif

template<typename tested_type>
void generateInput(dim_str dim_cpu, const std::string& input_distances,
		FOUR_VECTOR<tested_type> **rv_cpu, const std::string& input_charges,
		tested_type **qv_cpu) {
	// random generator seed set to random value - time in this case
	FILE *fp;
	int i;

	printf("Generating input...\n");

	srand(time(NULL));

	// input (distances)
	if ((fp = fopen(input_distances.c_str(), "wb")) == 0) {
		printf("The file 'input_distances' was not opened\n");
		exit(EXIT_FAILURE);
	}
	*rv_cpu = (FOUR_VECTOR<tested_type>*) malloc(dim_cpu.space_mem);
	for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
		// get a number in the range 0.1 - 1.0
		(*rv_cpu)[i].v = (tested_type) (rand() % 10 + 1) / tested_type(10.0);
		fwrite(&((*rv_cpu)[i].v), 1, sizeof(tested_type), fp);
		// get a number in the range 0.1 - 1.0
		(*rv_cpu)[i].x = (tested_type) (rand() % 10 + 1) / tested_type(10.0);
		fwrite(&((*rv_cpu)[i].x), 1, sizeof(tested_type), fp);
		// get a number in the range 0.1 - 1.0
		(*rv_cpu)[i].y = (tested_type) (rand() % 10 + 1) / tested_type(10.0);
		fwrite(&((*rv_cpu)[i].y), 1, sizeof(tested_type), fp);
		// get a number in the range 0.1 - 1.0
		(*rv_cpu)[i].z = (tested_type) (rand() % 10 + 1) / tested_type(10.0);
		fwrite(&((*rv_cpu)[i].z), 1, sizeof(tested_type), fp);
	}
	fclose(fp);

	// input (charge)
	if ((fp = fopen(input_charges.c_str(), "wb")) == 0) {
		printf("The file 'input_charges' was not opened\n");
		exit(EXIT_FAILURE);
	}

	*qv_cpu = (tested_type*) malloc(dim_cpu.space_mem2);
	for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
		// get a number in the range 0.1 - 1.0
		(*qv_cpu)[i] = (tested_type) (rand() % 10 + 1) / tested_type(10.0);
		fwrite(&((*qv_cpu)[i]), 1, sizeof(tested_type), fp);
	}
	fclose(fp);
}

template<typename tested_type>
void readInput(dim_str dim_cpu, const std::string& input_distances,
		FOUR_VECTOR<tested_type> **rv_cpu, const std::string& input_charges,
		tested_type **qv_cpu, int fault_injection) {
	FILE *fp;
	int i;
	size_t return_value[4];
	// size_t return_value;

	// input (distances)
	if ((fp = fopen(input_distances.c_str(), "rb")) == 0) {
		printf("The file 'input_distances' was not opened\n");
		exit(EXIT_FAILURE);
	}

	*rv_cpu = (FOUR_VECTOR<tested_type>*) malloc(dim_cpu.space_mem);
	if (*rv_cpu == NULL) {
		printf("error rv_cpu malloc\n");
#ifdef LOGS
		log_error_detail((char *)"error rv_cpu malloc"); end_log_file();
#endif
		exit(1);
	}

	// return_value = fread(*rv_cpu, sizeof(FOUR_VECTOR), dim_cpu.space_elem, fp);
	// if (return_value != dim_cpu.space_elem) {
	// 	printf("error reading rv_cpu from file\n");
	// 	#ifdef LOGS
	// 		log_error_detail((char *)"error reading rv_cpu from file"); end_log_file();
	// 	#endif
	// 	exit(1);
	// }

	for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
		return_value[0] = fread(&((*rv_cpu)[i].v), 1, sizeof(tested_type), fp);
		return_value[1] = fread(&((*rv_cpu)[i].x), 1, sizeof(tested_type), fp);
		return_value[2] = fread(&((*rv_cpu)[i].y), 1, sizeof(tested_type), fp);
		return_value[3] = fread(&((*rv_cpu)[i].z), 1, sizeof(tested_type), fp);
		if (return_value[0] == 0 || return_value[1] == 0 || return_value[2] == 0
				|| return_value[3] == 0) {
			printf("error reading rv_cpu from file\n");
#ifdef LOGS
			log_error_detail((char *)"error reading rv_cpu from file"); end_log_file();
#endif
			exit(1);
		}
	}
	fclose(fp);

	// input (charge)
	if ((fp = fopen(input_charges.c_str(), "rb")) == 0) {
		printf("The file 'input_charges' was not opened\n");
		exit(EXIT_FAILURE);
	}

	*qv_cpu = (tested_type*) malloc(dim_cpu.space_mem2);
	if (*qv_cpu == NULL) {
		printf("error qv_cpu malloc\n");
#ifdef LOGS
		log_error_detail((char *)"error qv_cpu malloc"); end_log_file();
#endif
		exit(1);
	}

	// return_value = fread(*qv_cpu, sizeof(tested_type), dim_cpu.space_elem, fp);
	// if (return_value != dim_cpu.space_elem) {
	// 	printf("error reading qv_cpu from file\n");
	// 	#ifdef LOGS
	// 		log_error_detail((char *)"error reading qv_cpu from file"); end_log_file();
	// 	#endif
	// 	exit(1);
	// }

	for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
		return_value[0] = fread(&((*qv_cpu)[i]), 1, sizeof(tested_type), fp);
		if (return_value[0] == 0) {
			printf("error reading qv_cpu from file\n");
#ifdef LOGS
			log_error_detail((char *)"error reading qv_cpu from file"); end_log_file();
#endif
			exit(1);
		}
	}
	fclose(fp);

	// =============== Fault injection
	if (fault_injection) {
		(*qv_cpu)[2] = 0.732637263; // must be in range 0.1 - 1.0
		printf("!!> Fault injection: qv_cpu[2]=%f\n", (double) (*qv_cpu)[2]);
	}
	// ========================
}

template<typename tested_type>
void readGold(dim_str dim_cpu, const std::string& output_gold,
		FOUR_VECTOR<tested_type> *fv_cpu_GOLD) {
	FILE *fp;
	size_t return_value[4];
	// size_t return_value;
	int i;

	if ((fp = fopen(output_gold.c_str(), "rb")) == 0) {
		printf("The file 'output_forces' was not opened\n");
		exit(EXIT_FAILURE);
	}

	for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
		return_value[0] = fread(&(fv_cpu_GOLD[i].v), 1, sizeof(tested_type),
				fp);
		return_value[1] = fread(&(fv_cpu_GOLD[i].x), 1, sizeof(tested_type),
				fp);
		return_value[2] = fread(&(fv_cpu_GOLD[i].y), 1, sizeof(tested_type),
				fp);
		return_value[3] = fread(&(fv_cpu_GOLD[i].z), 1, sizeof(tested_type),
				fp);
		if (return_value[0] == 0 || return_value[1] == 0 || return_value[2] == 0
				|| return_value[3] == 0) {
			printf("error reading rv_cpu from file\n");
#ifdef LOGS
			log_error_detail((char *)"error reading rv_cpu from file"); end_log_file();
#endif
			exit(1);
		}
	}
	fclose(fp);
}

template<typename tested_type>
void writeGold(dim_str dim_cpu, const std::string& output_gold,
		FOUR_VECTOR<tested_type> **fv_cpu) {
	FILE *fp;
	int i;

	if ((fp = fopen(output_gold.c_str(), "wb")) == 0) {
		printf("The file 'output_forces' was not opened\n");
		exit(EXIT_FAILURE);
	}
	int number_zeros = 0;
	for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
		if ((*fv_cpu)[i].v == tested_type(0.0))
			number_zeros++;
		if ((*fv_cpu)[i].x == tested_type(0.0))
			number_zeros++;
		if ((*fv_cpu)[i].y == tested_type(0.0))
			number_zeros++;
		if ((*fv_cpu)[i].z == tested_type(0.0))
			number_zeros++;

		fwrite(&((*fv_cpu)[i].v), 1, sizeof(tested_type), fp);
		fwrite(&((*fv_cpu)[i].x), 1, sizeof(tested_type), fp);
		fwrite(&((*fv_cpu)[i].y), 1, sizeof(tested_type), fp);
		fwrite(&((*fv_cpu)[i].z), 1, sizeof(tested_type), fp);
	}
	fclose(fp);
}

template<typename tested_type>
void gpu_memory_setup(int nstreams, bool gpu_check, dim_str dim_cpu,
		box_str **d_box_gpu, box_str *box_cpu,
		FOUR_VECTOR<tested_type> **d_rv_gpu, FOUR_VECTOR<tested_type> *rv_cpu,
		tested_type **d_qv_gpu, tested_type *qv_cpu,
		FOUR_VECTOR<tested_type> **d_fv_gpu,
		FOUR_VECTOR<tested_type> *d_fv_gold_gpu,
		FOUR_VECTOR<tested_type> *fv_cpu_GOLD) {

	for (int streamIdx = 0; streamIdx < nstreams; streamIdx++) {
		//=====================================================================
		//	GPU SETUP MEMORY
		//=====================================================================

		//==================================================
		//	boxes
		//==================================================
		rad::checkFrameworkErrors(
				cudaMalloc((void **) &(d_box_gpu[streamIdx]), dim_cpu.box_mem));
		//==================================================
		//	rv
		//==================================================
		rad::checkFrameworkErrors(
				cudaMalloc((void **) &(d_rv_gpu[streamIdx]),
						dim_cpu.space_mem));
		//==================================================
		//	qv
		//==================================================
		rad::checkFrameworkErrors(
				cudaMalloc((void **) &(d_qv_gpu[streamIdx]),
						dim_cpu.space_mem2));

		//==================================================
		//	fv
		//==================================================
		rad::checkFrameworkErrors(
				cudaMalloc((void **) &(d_fv_gpu[streamIdx]),
						dim_cpu.space_mem));

		//=====================================================================
		//	GPU MEMORY			COPY
		//=====================================================================

		//==================================================
		//	boxes
		//==================================================

		rad::checkFrameworkErrors(
				cudaMemcpy(d_box_gpu[streamIdx], box_cpu, dim_cpu.box_mem,
						cudaMemcpyHostToDevice));
		//==================================================
		//	rv
		//==================================================

		rad::checkFrameworkErrors(
				cudaMemcpy(d_rv_gpu[streamIdx], rv_cpu, dim_cpu.space_mem,
						cudaMemcpyHostToDevice));
		//==================================================
		//	qv
		//==================================================

		rad::checkFrameworkErrors(
				cudaMemcpy(d_qv_gpu[streamIdx], qv_cpu, dim_cpu.space_mem2,
						cudaMemcpyHostToDevice));
		//==================================================
		//	fv
		//==================================================

		// This will be done with memset at the start of each iteration.
		// rad::checkFrameworkErrors( cudaMemcpy( d_fv_gpu[streamIdx], fv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice) );
	}

	//==================================================
	//	fv_gold for GoldChkKernel
	//==================================================
	if (gpu_check) {
		rad::checkFrameworkErrors(
				cudaMalloc((void**) &d_fv_gold_gpu, dim_cpu.space_mem));
		rad::checkFrameworkErrors(
				cudaMemcpy(d_fv_gold_gpu, fv_cpu_GOLD, dim_cpu.space_mem2,
						cudaMemcpyHostToDevice));
	}
}

template<typename tested_type>
void gpu_memory_unset(int nstreams, int gpu_check, box_str **d_box_gpu,
		FOUR_VECTOR<tested_type> **d_rv_gpu, tested_type **d_qv_gpu,
		FOUR_VECTOR<tested_type> **d_fv_gpu,
		FOUR_VECTOR<tested_type> *d_fv_gold_gpu) {

	//=====================================================================
	//	GPU MEMORY DEALLOCATION
	//=====================================================================
	for (int streamIdx = 0; streamIdx < nstreams; streamIdx++) {
		cudaFree(d_rv_gpu[streamIdx]);
		cudaFree(d_qv_gpu[streamIdx]);
		cudaFree(d_fv_gpu[streamIdx]);
		cudaFree(d_box_gpu[streamIdx]);
	}
	if (gpu_check) {
		cudaFree(d_fv_gold_gpu);
	}
}

// Returns true if no errors are found. False if otherwise.
// Set votedOutput pointer to retrieve the voted matrix
template<typename tested_type>
bool checkOutputErrors(int verbose, dim_str dim_cpu, int streamIdx,
		FOUR_VECTOR<tested_type>* fv_cpu,
		FOUR_VECTOR<tested_type>* fv_cpu_GOLD) {
	int host_errors = 0;

#pragma omp parallel for shared(host_errors)
	for (int i = 0; i < dim_cpu.space_elem; i = i + 1) {
		FOUR_VECTOR<tested_type> valGold = fv_cpu_GOLD[i];
		FOUR_VECTOR<tested_type> valOutput = fv_cpu[i];
		if (valGold != valOutput) {
#pragma omp critical
			{
				char error_detail[500];
				host_errors++;

				snprintf(error_detail, 500,
						"stream: %d, p: [%d], v_r: %1.20e, v_e: %1.20e, x_r: %1.20e, x_e: %1.20e, y_r: %1.20e, y_e: %1.20e, z_r: %1.20e, z_e: %1.20e\n",
						streamIdx, i, (double) valOutput.v, (double) valGold.v,
						(double) valOutput.x, (double) valGold.x,
						(double) valOutput.y, (double) valGold.y,
						(double) valOutput.z, (double) valGold.z);
				if (verbose && (host_errors < 10))
					printf("%s\n", error_detail);
#ifdef LOGS
				if ((host_errors<MAX_LOGGED_ERRORS_PER_STREAM))
				log_error_detail(error_detail);
#endif
			}
		}
	}

	// printf("numErrors:%d", host_errors);

#ifdef LOGS
	log_error_count(host_errors);
#endif
	if (host_errors != 0)
		printf("#");

	return (host_errors == 0);
}

template<typename tested_type>
void setup_execution(const Parameters& parameters, Log& log) {
	//=====================================================================
	//	CPU/MCPU VARIABLES
	//=====================================================================
	// timer
	double timestamp;
	// counters
//	int i, j, k, l, m, n;
//	int iterations;

	// system memory
	par_str<tested_type> par_cpu;
	dim_str dim_cpu;
	box_str* box_cpu;
	FOUR_VECTOR<tested_type>* rv_cpu;
	tested_type* qv_cpu;
	FOUR_VECTOR<tested_type>* fv_cpu_GOLD;
	int nh;
//	int nstreams, streamIdx;

	int number_nn = 0;
	//=====================================================================
	//	CHECK INPUT ARGUMENTS
	//=====================================================================

	dim_cpu.boxes1d_arg = parameters.boxes;

	//=====================================================================
	//	INPUTS
	//=====================================================================
	par_cpu.alpha = 0.5;
	//=====================================================================
	//	DIMENSIONS
	//=====================================================================
	// total number of boxes
	dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg
			* dim_cpu.boxes1d_arg;
	// how many particles space has in each direction
	dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
	dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR<tested_type> );
	dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(tested_type);
	// box array
	dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);
	//=====================================================================
	//	SYSTEM MEMORY
	//=====================================================================
	// prepare host memory to receive kernel output
	// output (forces)
	FOUR_VECTOR<tested_type>* fv_cpu[parameters.nstreams];
	for (int streamIdx = 0; streamIdx < parameters.nstreams; streamIdx++) {
		fv_cpu[streamIdx] = (FOUR_VECTOR<tested_type>*) malloc(
				dim_cpu.space_mem);
		if (fv_cpu[streamIdx] == NULL) {
			printf("error fv_cpu malloc\n");
#ifdef LOGS
			if (!parameters.generate) log_error_detail((char *)"error fv_cpu malloc"); end_log_file();
#endif
			exit(1);
		}
	}
	fv_cpu_GOLD = (FOUR_VECTOR<tested_type>*) (malloc(dim_cpu.space_mem));
	if (fv_cpu_GOLD == NULL) {
		printf("error fv_cpu_GOLD malloc\n");
#ifdef LOGS
		log_error_detail((char *)"error fv_cpu_GOLD malloc"); end_log_file();
#endif
		exit(1);
	}
	//=====================================================================
	//	BOX
	//=====================================================================
	// allocate boxes
	box_cpu = (box_str*) (malloc(dim_cpu.box_mem));
	if (box_cpu == NULL) {
		printf("error box_cpu malloc\n");
#ifdef LOGS
		if (!parameters.generate) log_error_detail((char *)"error box_cpu malloc"); end_log_file();
#endif
		exit(1);
	}
	// initialize number of home boxes
	nh = 0;
	// home boxes in z direction
	for (int i = 0; i < dim_cpu.boxes1d_arg; i++) {
		// home boxes in y direction
		for (int j = 0; j < dim_cpu.boxes1d_arg; j++) {
			// home boxes in x direction
			for (int k = 0; k < dim_cpu.boxes1d_arg; k++) {
				// current home box
				box_cpu[nh].x = k;
				box_cpu[nh].y = j;
				box_cpu[nh].z = i;
				box_cpu[nh].number = nh;
				box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;
				// initialize number of neighbor boxes
				box_cpu[nh].nn = 0;
				// neighbor boxes in z direction
				for (int l = -1; l < 2; l++) {
					// neighbor boxes in y direction
					for (int m = -1; m < 2; m++) {
						// neighbor boxes in x direction
						for (int n = -1; n < 2; n++) {
							// check if (this neighbor exists) and (it is not the same as home box)
							if ((((i + l) >= 0 && (j + m) >= 0 && (k + n) >= 0)
									== true
									&& ((i + l) < dim_cpu.boxes1d_arg
											&& (j + m) < dim_cpu.boxes1d_arg
											&& (k + n) < dim_cpu.boxes1d_arg)
											== true)
									&& (l == 0 && m == 0 && n == 0) == false) {
								// current neighbor box
								box_cpu[nh].nei[box_cpu[nh].nn].x = (k + n);
								box_cpu[nh].nei[box_cpu[nh].nn].y = (j + m);
								box_cpu[nh].nei[box_cpu[nh].nn].z = (i + l);
								box_cpu[nh].nei[box_cpu[nh].nn].number =
										(box_cpu[nh].nei[box_cpu[nh].nn].z
												* dim_cpu.boxes1d_arg
												* dim_cpu.boxes1d_arg)
												+ (box_cpu[nh].nei[box_cpu[nh].nn].y
														* dim_cpu.boxes1d_arg)
												+ box_cpu[nh].nei[box_cpu[nh].nn].x;
								box_cpu[nh].nei[box_cpu[nh].nn].offset =
										box_cpu[nh].nei[box_cpu[nh].nn].number
												* NUMBER_PAR_PER_BOX;
								// increment neighbor box
								box_cpu[nh].nn = box_cpu[nh].nn + 1;
								number_nn += box_cpu[nh].nn;
							}
						} // neighbor boxes in x direction
					} // neighbor boxes in y direction
				} // neighbor boxes in z direction
				  // increment home box
				nh = nh + 1;
			} // home boxes in x direction
		} // home boxes in y direction
	} // home boxes in z direction
	  //=====================================================================
	  //	PARAMETERS, DISTANCE, CHARGE AND FORCE
	  //=====================================================================
	if (parameters.generate) {
		generateInput(dim_cpu, parameters.input_distances, &rv_cpu,
				parameters.input_charges, &qv_cpu);
	} else {
		readInput(dim_cpu, parameters.input_distances, &rv_cpu,
				parameters.input_charges, &qv_cpu, parameters.fault_injection);
		readGold(dim_cpu, parameters.output_gold, fv_cpu_GOLD);
	}
	//=====================================================================
	//	EXECUTION PARAMETERS
	//=====================================================================
	dim3 threads;
	dim3 blocks;
	blocks.x = dim_cpu.number_boxes;
	blocks.y = 1;
	// define the number of threads in the block
	threads.x = NUMBER_THREADS;
	threads.y = 1;
	//=====================================================================
	//	GPU_CUDA
	//=====================================================================
	//=====================================================================
	//	STREAMS
	//=====================================================================
	cudaStream_t* streams = (cudaStream_t*) (malloc(
			parameters.nstreams * sizeof(cudaStream_t)));
	for (int streamIdx = 0; streamIdx < parameters.nstreams; streamIdx++) {
		rad::checkFrameworkErrors(
				cudaStreamCreateWithFlags(&(streams[streamIdx]),
						cudaStreamNonBlocking));
	}
	//=====================================================================
	//	VECTORS
	//=====================================================================
	box_str* d_box_gpu[parameters.nstreams];
	FOUR_VECTOR<tested_type>* d_rv_gpu[parameters.nstreams];
	tested_type* d_qv_gpu[parameters.nstreams];
	FOUR_VECTOR<tested_type>* d_fv_gpu[parameters.nstreams];
	FOUR_VECTOR<tested_type>* d_fv_gold_gpu = nullptr;
	//=====================================================================
	//	GPU MEMORY SETUP
	//=====================================================================
	gpu_memory_setup(parameters.nstreams, parameters.gpu_check, dim_cpu,
			d_box_gpu, box_cpu, d_rv_gpu, rv_cpu, d_qv_gpu, qv_cpu, d_fv_gpu,
			d_fv_gold_gpu, fv_cpu_GOLD);
	////////////// GOLD CHECK Kernel /////////////////
	// dim3 gck_blockSize = dim3(	GOLDCHK_BLOCK_SIZE, 
	// 	GOLDCHK_BLOCK_SIZE);
	// dim3 gck_gridSize = dim3(	k / (GOLDCHK_BLOCK_SIZE * GOLDCHK_TILE_SIZE), 
	// 	k / (GOLDCHK_BLOCK_SIZE * GOLDCHK_TILE_SIZE));
	// //////////////////////////////////////////////////
	//LOOP START
	for (int loop = 0; loop < parameters.iterations; loop++) {

		if (parameters.verbose)
			printf("======== Iteration #%06u ========\n", loop);

		double globaltimer = rad::mysecond();
		timestamp = rad::mysecond();

		// for(i=0; i<dim_cpu.space_elem; i=i+1) {
		// 	// set to 0, because kernels keeps adding to initial value
		// 	fv_cpu[i].v = tested_type(0.0);
		// 	fv_cpu[i].x = tested_type(0.0);
		// 	fv_cpu[i].y = tested_type(0.0);
		// 	fv_cpu[i].z = tested_type(0.0);
		// }

		//=====================================================================
		//	GPU SETUP
		//=====================================================================
		for (int streamIdx = 0; streamIdx < parameters.nstreams; streamIdx++) {
			memset(fv_cpu[streamIdx], 0x00, dim_cpu.space_elem);
			rad::checkFrameworkErrors(
					cudaMemset(d_fv_gpu[streamIdx], 0x00, dim_cpu.space_mem));
		}

		if (parameters.verbose)
			printf("Setup prepare time: %.4fs\n", rad::mysecond() - timestamp);

		//=====================================================================
		//	KERNEL
		//=====================================================================

		double kernel_time = rad::mysecond();
		log.start_iteration();

		// launch kernel - all boxes
		for (int streamIdx = 0; streamIdx < parameters.nstreams; streamIdx++) {
			kernel_gpu_cuda<<<blocks, threads, 0, streams[streamIdx]>>>(par_cpu,
					dim_cpu, d_box_gpu[streamIdx], d_rv_gpu[streamIdx],
					d_qv_gpu[streamIdx], d_fv_gpu[streamIdx]);
			rad::checkFrameworkErrors(cudaPeekAtLastError());
		}

		for (int streamIdx = 0; streamIdx < parameters.nstreams; streamIdx++) {
			rad::checkFrameworkErrors(
					cudaStreamSynchronize(streams[streamIdx]));
			rad::checkFrameworkErrors(cudaPeekAtLastError());
		}

		log.end_iteration();
		kernel_time = rad::mysecond() - kernel_time;

		//=====================================================================
		//	COMPARE OUTPUTS / WRITE GOLD
		//=====================================================================
		if (parameters.generate) {
			rad::checkFrameworkErrors(
					cudaMemcpy(fv_cpu_GOLD, d_fv_gpu[0], dim_cpu.space_mem,
							cudaMemcpyDeviceToHost));
			writeGold(dim_cpu, parameters.output_gold, &fv_cpu_GOLD);
		} else {
			timestamp = rad::mysecond();
			{
				bool reloadFlag = false;
#pragma omp parallel for shared(reloadFlag)
				for (int streamIdx = 0; streamIdx < parameters.nstreams;
						streamIdx++) {
					rad::checkFrameworkErrors(
							cudaMemcpy(fv_cpu[streamIdx], d_fv_gpu[streamIdx],
									dim_cpu.space_mem, cudaMemcpyDeviceToHost));
					reloadFlag = reloadFlag
							|| checkOutputErrors(parameters.verbose, dim_cpu,
									streamIdx, fv_cpu[streamIdx], fv_cpu_GOLD);
				}
				if (reloadFlag) {
					readInput(dim_cpu, parameters.input_distances, &rv_cpu,
							parameters.input_charges, &qv_cpu,
							parameters.fault_injection);
					readGold(dim_cpu, parameters.output_gold, fv_cpu_GOLD);

					gpu_memory_unset(parameters.nstreams, parameters.gpu_check,
							d_box_gpu, d_rv_gpu, d_qv_gpu, d_fv_gpu,
							d_fv_gold_gpu);
					gpu_memory_setup(parameters.nstreams, parameters.gpu_check,
							dim_cpu, d_box_gpu, box_cpu, d_rv_gpu, rv_cpu,
							d_qv_gpu, qv_cpu, d_fv_gpu, d_fv_gold_gpu,
							fv_cpu_GOLD);
				}
			}
			if (parameters.verbose)
				printf("Gold check time: %f\n", rad::mysecond() - timestamp);
		}

		//================= PERF
		// iterate for each neighbor of a box (number_nn)
		double flop = number_nn;
		// The last for iterate NUMBER_PAR_PER_BOX times
		flop *= NUMBER_PAR_PER_BOX;
		// the last for uses 46 operations plus 2 exp() functions
		flop *= 46;
		flop *= parameters.nstreams;
		double flops = (double) flop / kernel_time;
		double outputpersec = (double) dim_cpu.space_elem * 4
				* parameters.nstreams / kernel_time;
		if (parameters.verbose)
			printf("BOXES:%d BLOCK:%d OUTPUT/S:%.2f FLOPS:%.2f (GFLOPS:%.2f)\n",
					dim_cpu.boxes1d_arg, NUMBER_THREADS, outputpersec, flops,
					flops / 1000000000);
		if (parameters.verbose)
			printf("Kernel time:%f\n", kernel_time);
		//=====================

		printf(".");
		fflush(stdout);

		double iteration_time = rad::mysecond() - globaltimer;
		if (parameters.verbose)
			printf("Iteration time: %.4fs (%3.1f%% Device)\n", iteration_time,
					(kernel_time / iteration_time) * 100.0);
		if (parameters.verbose)
			printf("===================================\n");

		fflush(stdout);
	}
	gpu_memory_unset(parameters.nstreams, parameters.gpu_check, d_box_gpu,
			d_rv_gpu, d_qv_gpu, d_fv_gpu, d_fv_gold_gpu);
	//=====================================================================
	//	SYSTEM MEMORY DEALLOCATION
	//=====================================================================
	if (!parameters.generate && fv_cpu_GOLD)
		free(fv_cpu_GOLD);

	//if (fv_cpu) free(fv_cpu);
	for (int streamIdx = 0; streamIdx < parameters.nstreams; streamIdx++) {
		free(fv_cpu[streamIdx]);
	}
	if (rv_cpu)
		free(rv_cpu);

	if (qv_cpu)
		free(qv_cpu);

	if (box_cpu)
		free(box_cpu);

	printf("\n");
}

//=============================================================================
//	MAIN FUNCTION
//=============================================================================

int main(int argc, char *argv[]) {

	//=====================================================================
	//	CPU/MCPU VARIABLES
	//=====================================================================
	Parameters parameters(argc, argv);
	Log log;

//	char test_info[200];
//	char test_name[200];
//	snprintf(test_info, 200,
//			"type:%s-precision streams:%d boxes:%d block_size:%d",
//			test_precision_description, parameters.nstreams, dim_cpu.boxes1d_arg,
//			NUMBER_THREADS);
//	snprintf(test_name, 200, "cuda_%s_lava", test_precision_description);
//	printf(
//			"\n=================================\n%s\n%s\n=================================\n\n",
//			test_name, test_info);

	// timer
#ifdef LOGS
	if (!generate) {
		start_log_file(test_name, test_info);
		set_max_errors_iter(MAX_LOGGED_ERRORS_PER_STREAM * nstreams + 32);
	}

#ifdef BUILDPROFILER

	std::string log_file_name(get_log_file_name());
	if(generate) {
		log_file_name = "/tmp/generate.log";
	}
//	rad::Profiler profiler_thread = new rad::JTX2Inst(log_file_name);
	std::shared_ptr<rad::Profiler> profiler_thread = std::make_shared<rad::OBJTYPE>(0, log_file_name);

//START PROFILER THREAD
	profiler_thread->start_profile();
#endif
#endif

	setup_execution<float>(parameters, log);

#ifdef LOGS
#ifdef BUILDPROFILER
	profiler_thread->end_profile();
#endif
	if (!generate) end_log_file();
#endif

	return 0;
}
