/*
 * LavaExecute.cpp
 *
 *  Created on: 22/05/2019
 *      Author: fernando
 */

#include "LavaExecute.h"
#include "types.h"
#include "DataManagement.h"
#include "kernels.h"

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
	dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR<full> );
	dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(full);

	// box array
	dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

	//=====================================================================
	//	SYSTEM MEMORY
	//=====================================================================

	// prepare host memory to receive kernel output
	// output (forces)
	std::vector<FOUR_VECTOR<full>> fv_cpu[this->setup_parameters.nstreams];
	for (int streamIdx = 0; streamIdx < this->setup_parameters.nstreams;
			streamIdx++) {
		fv_cpu[streamIdx] = std::vector<FOUR_VECTOR<full>>(dim_cpu.space_mem);
	}

	std::vector<FOUR_VECTOR<full>> fv_cpu_GOLD = std::vector<FOUR_VECTOR<full>>(
			dim_cpu.space_mem);

	//=====================================================================
	//	BOX
	//=====================================================================

	// allocate boxes
	std::vector<box_str> box_cpu = std::vector<box_str>(dim_cpu.box_mem);

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

	DataManagement<full>& ld = lava_data;

	//LOOP START
	for (int loop = 0; loop < this->setup_parameters.iterations; loop++) {

		if (this->setup_parameters.verbose)
			std::cout << "======== Iteration  #" << loop << "  ========\n";

		double globaltimer = Log::mysecond();
		timestamp = Log::mysecond();

		// for(i=0; i<dim_cpu.space_elem; i=i+1) {
		// 	// set to 0, because kernels keeps adding to initial value
		// 	fv_cpu[i].v = tested_type_host(0.0);
		// 	fv_cpu[i].x = tested_type_host(0.0);
		// 	fv_cpu[i].y = tested_type_host(0.0);
		// 	fv_cpu[i].z = tested_type_host(0.0);
		// }

		//=====================================================================
		//	GPU SETUP
		//=====================================================================
		for (int streamIdx = 0; streamIdx < this->setup_parameters.nstreams; streamIdx++) {
			memset(fv_cpu[streamIdx].data(), 0x00, sizeof(FOUR_VECTOR<full>) * fv_cpu[streamIdx].size());

			ld.d_fv_gpu[streamIdx].clear();
		}

		if (this->setup_parameters.verbose)
			std::cout << "Setup prepare time: " << Log::mysecond() - timestamp
					<< std::endl;

		//=====================================================================
		//	KERNEL
		//=====================================================================

		double kernel_time = Log::mysecond();
		this->log.start_iteration_app();

		// launch kernel - all boxes
		for (int streamIdx = 0; streamIdx < this->setup_parameters.nstreams; streamIdx++) {
			kernel_gpu_cuda<<<blocks, threads, 0, ld.streams[streamIdx]>>>(par_cpu,
					dim_cpu, ld.d_box_gpu[streamIdx].data, ld.d_rv_gpu[streamIdx].data,
					ld.d_qv_gpu[streamIdx].data, ld.d_fv_gpu[streamIdx].data);
			checkFrameworkErrors(cudaPeekAtLastError());
		}

		for (int streamIdx = 0; streamIdx < this->setup_parameters.nstreams; streamIdx++) {
			checkFrameworkErrors(cudaStreamSynchronize(ld.streams[streamIdx]));
			checkFrameworkErrors(cudaPeekAtLastError());
		}

		this->log.end_iteration_app();

		kernel_time = Log::mysecond() - kernel_time;

		//=====================================================================
		//	COMPARE OUTPUTS / WRITE GOLD
		//=====================================================================
		if (this->setup_parameters.generate) {
			fv_cpu_GOLD = ld.d_fv_gpu[0].to_vector();
			ld.writeGold();
		} else {
			timestamp = Log::mysecond();
			bool checkOnHost = false;
			checkOnHost = true;
			if (checkOnHost) {
				bool reloadFlag = false;
#pragma omp parallel for shared(reloadFlag)
				for (int streamIdx = 0; streamIdx < this->setup_parameters.nstreams; streamIdx++) {
						fv_cpu[streamIdx] = ld.d_fv_gpu[streamIdx].to_vector();

					reloadFlag = reloadFlag
							|| ld.checkOutputErrors();
				}
				if (reloadFlag) {
					ld.readInput();
					ld.readGold();
				}
			}
			if (this->setup_parameters.verbose)
				std::cout << "Gold check time: " << Log::mysecond() - timestamp << std::endl;
		}

		//================= PERF
		// iterate for each neighbor of a box (number_nn)
		double flop = number_nn;
		// The last for iterate NUMBER_PAR_PER_BOX times
		flop *= NUMBER_PAR_PER_BOX;
		// the last for uses 46 operations plus 2 exp() functions
		flop *= 46;
		flop *= this->setup_parameters.nstreams;
		double flops = (double) flop / kernel_time;
		double outputpersec = (double) dim_cpu.space_elem * 4 * this->setup_parameters.nstreams
				/ kernel_time;
		if (this->setup_parameters.verbose)
			printf("BOXES:%d BLOCK:%d OUTPUT/S:%.2f FLOPS:%.2f (GFLOPS:%.2f)\n",
					dim_cpu.boxes1d_arg, NUMBER_THREADS, outputpersec, flops,
					flops / 1000000000);
		if (this->setup_parameters.verbose)
			printf("Kernel time:%f\n", kernel_time);
		//=====================

		printf(".");
		fflush(stdout);

		double iteration_time = Log::mysecond() - globaltimer;
		if (this->setup_parameters.verbose)
			printf("Iteration time: %.4fs (%3.1f%% Device)\n", iteration_time,
					(kernel_time / iteration_time) * 100.0);
		if (this->setup_parameters.verbose)
			printf("===================================\n");

		fflush(stdout);
	}


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
