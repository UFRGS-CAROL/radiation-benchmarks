/*
 * LavaExecute.cpp
 *
 *  Created on: 22/05/2019
 *      Author: fernando
 */

#include "LavaExecute.h"
#include "types.h"
#include "DataManagement.h"

#include <cuda_fp16.h>
#include <vector>
#include <iostream>

LavaExecute::LavaExecute(Parameters& setup_parameters, Log& log) :
		setup_parameters(setup_parameters), log(log) {
	// TODO Auto-generated constructor stub

}

LavaExecute::~LavaExecute() {
	// TODO Auto-generated destructor stub
}

template<typename full, typename incomplete>
inline void LavaExecute::generic_execute() {
	DataManagement<full> lava_data(this->setup_parameters);

	//=====================================================================
	//	CPU/MCPU VARIABLES
	//=====================================================================

	// timer
	double timestamp;

	// counters
	int l, m, n;

	// system memory
	par_str<full> par_cpu;
	dim_str dim_cpu;
//	box_str *box_cpu;
//	FOUR_VECTOR<full> *rv_cpu;
	full *qv_cpu;
//	FOUR_VECTOR<full> *fv_cpu_GOLD;
//	int nh;
	int number_nn = 0;

	//=====================================================================
	//	INPUTS
	//=====================================================================
	par_cpu.alpha = 0.5;
	//=====================================================================
	//	DIMENSIONS
	//=====================================================================

	// total number of boxes
	dim_cpu.boxes1d_arg = this->setup_parameters.boxes;
	dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg
			* dim_cpu.boxes1d_arg;

	// how many particles space has in each direction
	dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
	dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR<full>);
	dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(full);

	// box array
	dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

	//=====================================================================
	//	SYSTEM MEMORY
	//=====================================================================

	// prepare host memory to receive kernel output
	// output (forces)
	std::vector<FOUR_VECTOR<full>> fv_cpu[this->setup_parameters.nstreams];
	for (int streamIdx = 0; streamIdx < this->setup_parameters.nstreams; streamIdx++) {
		fv_cpu[streamIdx] = std::vector<FOUR_VECTOR<full>>(dim_cpu.space_mem);
	}

	std::vector<FOUR_VECTOR<full>> fv_cpu_GOLD = std::vector<FOUR_VECTOR<full>>(dim_cpu.space_mem);


	//=====================================================================
	//	BOX
	//=====================================================================

	// allocate boxes
	std::vector<box_str>  box_cpu = std::vector<box_str>(dim_cpu.box_mem);


	// initialize number of home boxes
	int nh = 0;

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
				for (l = -1; l < 2; l++) {
					// neighbor boxes in y direction
					for (m = -1; m < 2; m++) {
						// neighbor boxes in x direction
						for (n = -1; n < 2; n++) {

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

						}	 // neighbor boxes in x direction
					}		 // neighbor boxes in y direction
				}			 // neighbor boxes in z direction

				// increment home box
				nh = nh + 1;

			}				 // home boxes in x direction
		}					 // home boxes in y direction
	}						 // home boxes in z direction

	//=====================================================================
	//	PARAMETERS, DISTANCE, CHARGE AND FORCE
	//=====================================================================
	lava_data.readInput();
	lava_data.readGold();

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
	//	VECTORS
	//=====================================================================
	std::vector<box_str> d_box_gpu[nstreams];
	std::vector<FOUR_VECTOR<full>> d_rv_gpu[nstreams];
	std::vector<full> d_qv_gpu[nstreams];
	std::vector<FOUR_VECTOR<full>> d_fv_gpu[nstreams];
	std::vector<FOUR_VECTOR<full>> d_fv_gold_gpu;
//
//	//=====================================================================
//	//	GPU MEMORY SETUP
//	//=====================================================================
//	gpu_memory_setup(nstreams, gpu_check, dim_cpu, d_box_gpu, box_cpu, d_rv_gpu,
//			rv_cpu, d_qv_gpu, qv_cpu, d_fv_gpu, d_fv_gold_gpu, fv_cpu_GOLD);
//
//	////////////// GOLD CHECK Kernel /////////////////
//	// dim3 gck_blockSize = dim3(	GOLDCHK_BLOCK_SIZE,
//	// 	GOLDCHK_BLOCK_SIZE);
//	// dim3 gck_gridSize = dim3(	k / (GOLDCHK_BLOCK_SIZE * GOLDCHK_TILE_SIZE),
//	// 	k / (GOLDCHK_BLOCK_SIZE * GOLDCHK_TILE_SIZE));
//	// //////////////////////////////////////////////////
//
//	//LOOP START
//	int loop;
//	for (loop = 0; loop < iterations; loop++) {
//
//		if (verbose)
//			printf("======== Iteration #%06u ========\n", loop);
//
//		double globaltimer = mysecond();
//		timestamp = mysecond();
//
//		// for(i=0; i<dim_cpu.space_elem; i=i+1) {
//		// 	// set to 0, because kernels keeps adding to initial value
//		// 	fv_cpu[i].v = tested_type_host(0.0);
//		// 	fv_cpu[i].x = tested_type_host(0.0);
//		// 	fv_cpu[i].y = tested_type_host(0.0);
//		// 	fv_cpu[i].z = tested_type_host(0.0);
//		// }
//
//		//=====================================================================
//		//	GPU SETUP
//		//=====================================================================
//		for (streamIdx = 0; streamIdx < nstreams; streamIdx++) {
//			memset(fv_cpu[streamIdx], 0x00, dim_cpu.space_elem);
//			checkFrameworkErrors(
//					cudaMemset(d_fv_gpu[streamIdx], 0x00, dim_cpu.space_mem));
//		}
//
//		if (verbose)
//			printf("Setup prepare time: %.4fs\n", mysecond() - timestamp);
//
//		//=====================================================================
//		//	KERNEL
//		//=====================================================================
//
//		double kernel_time = mysecond();
//#ifdef LOGS
//		if (!generate) start_iteration();
//#endif
//		// launch kernel - all boxes
//		for (streamIdx = 0; streamIdx < nstreams; streamIdx++) {
//			kernel_gpu_cuda<<<blocks, threads, 0, streams[streamIdx]>>>(par_cpu,
//					dim_cpu, d_box_gpu[streamIdx], d_rv_gpu[streamIdx],
//					d_qv_gpu[streamIdx], d_fv_gpu[streamIdx]);
//			checkFrameworkErrors (cudaPeekAtLastError());}
//			//printf("All kernels were commited.\n");
//		for (streamIdx = 0; streamIdx < nstreams; streamIdx++) {
//			checkFrameworkErrors(cudaStreamSynchronize(streams[streamIdx]));
//			checkFrameworkErrors (cudaPeekAtLastError());}
//#ifdef LOGS
//			if (!generate) end_iteration();
//#endif
//		kernel_time = mysecond() - kernel_time;
//
//		//=====================================================================
//		//	COMPARE OUTPUTS / WRITE GOLD
//		//=====================================================================
//		if (generate) {
//			checkFrameworkErrors(
//					cudaMemcpy(fv_cpu_GOLD, d_fv_gpu[0], dim_cpu.space_mem,
//							cudaMemcpyDeviceToHost));
//			writeGold(dim_cpu, output_gold, &fv_cpu_GOLD);
//		} else {
//			timestamp = mysecond();
//			bool checkOnHost = false;
//			// if (test_gpu_check) {
//			// 	assert (d_GOLD != NULL);
//
//			// 	// Send to device
//			// 	unsigned long long int gck_errors = 0;
//			// 	checkOnHost |= checkFrameworkErrorsNoFail( cudaMemcpyToSymbol(gck_device_errors, &gck_errors, sizeof(unsigned long long int)) );
//			// 	// GOLD is already on device.
//
//			// 	/////////////////// Run kernel
//			// 	GoldChkKernel<<<gck_gridSize, gck_blockSize>>>(d_GOLD, d_C, k);
//			// 	checkOnHost |= checkFrameworkErrorsNoFail( cudaPeekAtLastError() );
//			// 	checkOnHost |= checkFrameworkErrorsNoFail( cudaDeviceSynchronize() );
//			// 	///////////////////
//
//			// 	// Receive from device
//			// 	checkOnHost |= checkFrameworkErrorsNoFail( cudaMemcpyFromSymbol(&gck_errors, gck_device_errors, sizeof(unsigned long long int)) );
//			// 	if (gck_errors != 0) {
//			// 		printf("$(%u)", (unsigned int)gck_errors);
//			// 		checkOnHost = true;
//			// 	}
//			// } else {
//			checkOnHost = true;
//			// }
//			if (checkOnHost) {
//				bool reloadFlag = false;
//#pragma omp parallel for shared(reloadFlag)
//				for (int streamIdx = 0; streamIdx < nstreams; streamIdx++) {
//					checkFrameworkErrors(
//							cudaMemcpy(fv_cpu[streamIdx], d_fv_gpu[streamIdx],
//									dim_cpu.space_mem, cudaMemcpyDeviceToHost));
//					reloadFlag = reloadFlag
//							|| checkOutputErrors(verbose, dim_cpu, streamIdx,
//									fv_cpu[streamIdx], fv_cpu_GOLD);
//				}
//				if (reloadFlag) {
//					readInput(dim_cpu, input_distances, &rv_cpu, input_charges,
//							&qv_cpu, fault_injection);
//					readGold(dim_cpu, output_gold, fv_cpu_GOLD);
//
//					gpu_memory_unset(nstreams, gpu_check, d_box_gpu, d_rv_gpu,
//							d_qv_gpu, d_fv_gpu, d_fv_gold_gpu);
//					gpu_memory_setup(nstreams, gpu_check, dim_cpu, d_box_gpu,
//							box_cpu, d_rv_gpu, rv_cpu, d_qv_gpu, qv_cpu,
//							d_fv_gpu, d_fv_gold_gpu, fv_cpu_GOLD);
//				}
//			}
//			if (verbose)
//				printf("Gold check time: %f\n", mysecond() - timestamp);
//		}
//
//		//================= PERF
//		// iterate for each neighbor of a box (number_nn)
//		double flop = number_nn;
//		// The last for iterate NUMBER_PAR_PER_BOX times
//		flop *= NUMBER_PAR_PER_BOX;
//		// the last for uses 46 operations plus 2 exp() functions
//		flop *= 46;
//		flop *= nstreams;
//		double flops = (double) flop / kernel_time;
//		double outputpersec = (double) dim_cpu.space_elem * 4 * nstreams
//				/ kernel_time;
//		if (verbose)
//			printf("BOXES:%d BLOCK:%d OUTPUT/S:%.2f FLOPS:%.2f (GFLOPS:%.2f)\n",
//					dim_cpu.boxes1d_arg, NUMBER_THREADS, outputpersec, flops,
//					flops / 1000000000);
//		if (verbose)
//			printf("Kernel time:%f\n", kernel_time);
//		//=====================
//
//		printf(".");
//		fflush(stdout);
//
//		double iteration_time = mysecond() - globaltimer;
//		if (verbose)
//			printf("Iteration time: %.4fs (%3.1f%% Device)\n", iteration_time,
//					(kernel_time / iteration_time) * 100.0);
//		if (verbose)
//			printf("===================================\n");
//
//		fflush(stdout);
//	}
//
//	gpu_memory_unset(nstreams, gpu_check, d_box_gpu, d_rv_gpu, d_qv_gpu,
//			d_fv_gpu, d_fv_gold_gpu);
//
//	//=====================================================================
//	//	SYSTEM MEMORY DEALLOCATION
//	//=====================================================================
//
//	if (!generate && fv_cpu_GOLD)
//		free(fv_cpu_GOLD);
//
//	//if (fv_cpu) free(fv_cpu);
//	for (int streamIdx = 0; streamIdx < nstreams; streamIdx++) {
//		free(fv_cpu[streamIdx]);
//	}
//
//	if (rv_cpu)
//		free(rv_cpu);
//	if (qv_cpu)
//		free(qv_cpu);
//	if (box_cpu)
//		free(box_cpu);
//	printf("\n");
//
//#ifdef LOGS
//	if (!generate) end_log_file();
//#endif

}

void LavaExecute::execute() {
	switch (this->setup_parameters.redundancy) {
		case NONE:
		case DMR:
			switch (this->setup_parameters.precision) {
			case HALF:

				generic_execute<half, half>();
				break;

			case SINGLE:
				generic_execute<float, float>();
				break;

			case DOUBLE:
				generic_execute<double, double>();
				break;

			}
			break;

		case DMRMIXED:
			switch (this->setup_parameters.precision) {
			case SINGLE:
				generic_execute<float, half>();
				break;

			case DOUBLE:
				generic_execute<double, float>();
				break;

			}
			break;

		}
}
