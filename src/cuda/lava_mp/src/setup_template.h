/*
 * setup_template.h
 *
 *  Created on: Oct 3, 2019
 *      Author: carol
 */

#ifndef SETUP_TEMPLATE_H_
#define SETUP_TEMPLATE_H_

#include <random>
#include <omp.h>
#include <cuda_fp16.h>

#include "include/cuda_utils.h"
#include "Parameters.h"
#include "Log.h"
#include "types.h"
#include "common.h"
#include "File.h"
#include "KernelCaller.h"

template<typename real_t>
void generateInput(dim_str dim_cpu, std::string& input_distances,
		std::vector<FOUR_VECTOR<real_t>>& rv_cpu, std::string& input_charges,
		std::vector<real_t>& qv_cpu) {

	if (File<real_t>::exists(input_distances) && File<real_t>::exists(input_charges)){
		return;
	}

	// random generator seed set to random value - time in this case
	std::cout << ("Generating input...\n");

	// get a number in the range 0.1 - 1.0
	std::random_device rd; //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<real_t> dis(0.1, 1.0);

	rv_cpu.resize(dim_cpu.space_elem);
	qv_cpu.resize(dim_cpu.space_elem);
	for (auto& rv_cpu_i : rv_cpu) {
		rv_cpu_i.v = real_t(dis(gen));
		rv_cpu_i.x = real_t(dis(gen));
		rv_cpu_i.y = real_t(dis(gen));
		rv_cpu_i.z = real_t(dis(gen));
	}

	std::cout << "TESTE " << rv_cpu[35].x << std::endl;

	if (File<FOUR_VECTOR<real_t>>::write_to_file(input_distances, rv_cpu)) {
		error("error writing rv_cpu from file\n");
	}

	for (auto& qv_cpu_i : qv_cpu) {
		// get a number in the range 0.1 - 1.0
		qv_cpu_i = real_t(dis(gen));
	}

	if (File<real_t>::write_to_file(input_charges, qv_cpu)) {
		error("error writing qv_cpu from file\n");
	}

}

template<typename real_t>
void readInput(dim_str dim_cpu, std::string& input_distances,
		std::vector<FOUR_VECTOR<real_t>>& rv_cpu, std::string& input_charges,
		std::vector<real_t>& qv_cpu, int fault_injection) {

	rv_cpu.resize(dim_cpu.space_elem);
	qv_cpu.resize(dim_cpu.space_elem);

	if (File<FOUR_VECTOR<real_t>>::read_from_file(input_distances, rv_cpu)) {
		error("error reading rv_cpu from file\n");
	}

	if (File<real_t>::read_from_file(input_charges, qv_cpu)) {
		error("error reading qv_cpu from file\n");
	}

	// =============== Fault injection
	if (fault_injection) {
		qv_cpu[2] = 0.732637263; // must be in range 0.1 - 1.0
		std::cout << "!!> Fault injection: qv_cpu[2]= " << qv_cpu[2]
				<< std::endl;
	}
	// ========================
}

template<typename real_t>
void readGold(dim_str dim_cpu, std::string& output_gold,
		std::vector<FOUR_VECTOR<real_t>>& fv_cpu_GOLD) {
	if (File<FOUR_VECTOR<real_t>>::read_from_file(output_gold, fv_cpu_GOLD)) {
		error("error reading fv_cpu_GOLD from file\n");
	}
}

template<typename real_t>
void writeGold(dim_str dim_cpu, std::string& output_gold,
		std::vector<FOUR_VECTOR<real_t>>& fv_cpu) {

	int number_zeros = 0;
	for (auto& fv_cpu_i : fv_cpu) {
		if (fv_cpu_i.v == real_t(0.0))
			number_zeros++;
		if (fv_cpu_i.x == real_t(0.0))
			number_zeros++;
		if (fv_cpu_i.y == real_t(0.0))
			number_zeros++;
		if (fv_cpu_i.z == real_t(0.0))
			number_zeros++;
	}

	if (File<FOUR_VECTOR<real_t>>::write_to_file(output_gold, fv_cpu)) {
		error("error writing fv_cpu from file\n");
	}

	std::cout << "Number of zeros " << number_zeros << std::endl;
}

template<typename real_t>
void gpu_memory_setup(const Parameters& parameters,
		VectorOfDeviceVector<box_str>& d_box_gpu, std::vector<box_str>& box_cpu,
		VectorOfDeviceVector<FOUR_VECTOR<real_t>>& d_rv_gpu,
		std::vector<FOUR_VECTOR<real_t>>& rv_cpu,
		VectorOfDeviceVector<real_t>& d_qv_gpu, std::vector<real_t>& qv_cpu,
		VectorOfDeviceVector<FOUR_VECTOR<real_t>>& d_fv_gpu,
		std::vector<std::vector<FOUR_VECTOR<real_t>>>& fv_cpu,
rad::DeviceVector<FOUR_VECTOR<real_t>>& d_fv_gold_gpu, std::vector<FOUR_VECTOR<real_t>>& fv_cpu_GOLD) {

	for (int stream_idx = 0; stream_idx < parameters.nstreams; stream_idx++) {
		d_box_gpu[stream_idx] = box_cpu;
		d_rv_gpu[stream_idx] = rv_cpu;
		d_qv_gpu[stream_idx] = qv_cpu;
		d_fv_gpu[stream_idx] = fv_cpu[stream_idx];
	}

	if (parameters.gpu_check) {
		d_fv_gold_gpu = fv_cpu_GOLD;
	}
}

template<typename real_t>
void gpu_memory_unset(const Parameters& parameters,
		VectorOfDeviceVector<box_str>& d_box_gpu,
		VectorOfDeviceVector<FOUR_VECTOR<real_t>>& d_rv_gpu,
		VectorOfDeviceVector<real_t>& d_qv_gpu,
		VectorOfDeviceVector<FOUR_VECTOR<real_t>>& d_fv_gpu,
		rad::DeviceVector<FOUR_VECTOR<real_t>>& d_fv_gold_gpu) {

	//=====================================================================
	//	GPU MEMORY DEALLOCATION
	//=====================================================================
	for (int stream_idx = 0; stream_idx < parameters.nstreams; stream_idx++) {
		d_rv_gpu[stream_idx].resize(0);
		d_qv_gpu[stream_idx].resize(0);
		d_fv_gpu[stream_idx].resize(0);
		d_box_gpu[stream_idx].resize(0);
	}
	if (parameters.gpu_check) {
		d_fv_gold_gpu.resize(0);
	}
}

template<const uint32_t COUNT, typename half_t, typename real_t>
void setup_execution(Parameters& parameters, Log& log,
		KernelCaller<COUNT, half_t, real_t>& kernel_caller) {
	//=====================================================================
	//	CPU/MCPU VARIABLES
	//=====================================================================
	// timer
	double timestamp;

	// system memory
	par_str<real_t> par_cpu;
	dim_str dim_cpu;
	std::vector<box_str> box_cpu;
	std::vector<FOUR_VECTOR<real_t>> rv_cpu;
	std::vector<real_t> qv_cpu;
	std::vector<FOUR_VECTOR<real_t>> fv_cpu_GOLD;
	int nh;
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
	dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR<real_t> );
	dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(real_t);
	// box array
	dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);
	//=====================================================================
	//	SYSTEM MEMORY
	//=====================================================================
	// prepare host memory to receive kernel output
	// output (forces)
	std::vector<std::vector<FOUR_VECTOR<real_t>>>fv_cpu(parameters.nstreams);
	kernel_caller.set_half_t_vectors(parameters.nstreams, dim_cpu.space_elem);

	for (auto& fv_cpu_i : fv_cpu) {
		fv_cpu_i.resize(dim_cpu.space_elem);
	}

	fv_cpu_GOLD.resize(dim_cpu.space_elem);
	//=====================================================================
	//	BOX
	//=====================================================================
	// allocate boxes
	box_cpu.resize(dim_cpu.number_boxes);

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
		generateInput(dim_cpu, parameters.input_distances, rv_cpu,
				parameters.input_charges, qv_cpu);
		readInput(dim_cpu, parameters.input_distances, rv_cpu,
				parameters.input_charges, qv_cpu, parameters.fault_injection);
	} else {
		readInput(dim_cpu, parameters.input_distances, rv_cpu,
				parameters.input_charges, qv_cpu, parameters.fault_injection);
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
	std::vector<CudaStream> streams(parameters.nstreams);

	//=====================================================================
	//	VECTORS
	//=====================================================================
	VectorOfDeviceVector<box_str> d_box_gpu(parameters.nstreams);
	VectorOfDeviceVector<FOUR_VECTOR<real_t>> d_rv_gpu(parameters.nstreams);
	VectorOfDeviceVector<real_t> d_qv_gpu(parameters.nstreams);
	VectorOfDeviceVector<FOUR_VECTOR<real_t>> d_fv_gpu(parameters.nstreams);

	rad::DeviceVector<FOUR_VECTOR<real_t>> d_fv_gold_gpu;
	//=====================================================================
	//	GPU MEMORY SETUP
	//=====================================================================
	gpu_memory_setup(parameters, d_box_gpu, box_cpu, d_rv_gpu, rv_cpu, d_qv_gpu,
			qv_cpu, d_fv_gpu, fv_cpu, d_fv_gold_gpu, fv_cpu_GOLD);

	//LOOP START
	for (int loop = 0; loop < parameters.iterations; loop++) {

		if (parameters.verbose)
			std::cout << "======== Iteration #" << loop << "========\n";

		double globaltimer = rad::mysecond();
		timestamp = rad::mysecond();

		//=====================================================================
		//	GPU SETUP
		//=====================================================================
		for (uint32_t stream_idx = 0; stream_idx < parameters.nstreams;
				stream_idx++) {
			auto& it = fv_cpu[stream_idx];
			std::fill(it.begin(), it.end(), FOUR_VECTOR<real_t>());
			d_fv_gpu[stream_idx].clear();
		}

		kernel_caller.clear_half_t();

		if (parameters.verbose)
			std::cout << "Setup prepare time: " << rad::mysecond() - timestamp
					<< "s\n";

		//=====================================================================
		//	KERNEL
		//=====================================================================

		double kernel_time = rad::mysecond();
		log.start_iteration();

		// launch kernel - all boxes
		for (uint32_t stream_idx = 0; stream_idx < parameters.nstreams;
				stream_idx++) {

			kernel_caller.kernel_call(blocks, threads, streams[stream_idx],
					par_cpu, dim_cpu, d_box_gpu[stream_idx].data(),
					d_rv_gpu[stream_idx].data(), d_qv_gpu[stream_idx].data(),
					d_fv_gpu[stream_idx].data(), stream_idx);

			rad::checkFrameworkErrors (cudaPeekAtLastError());;
		}

		for (auto& st : streams) {
			st.sync();
			rad::checkFrameworkErrors (cudaPeekAtLastError());;
		}

		log.end_iteration();
		kernel_time = rad::mysecond() - kernel_time;

		auto cpy_time = rad::mysecond();
		for (uint32_t stream_idx = 0; stream_idx < parameters.nstreams;
				stream_idx++) {
			fv_cpu[stream_idx] = d_fv_gpu[stream_idx].to_vector();
		}
		cpy_time = rad::mysecond() - cpy_time;

		//=====================================================================
		//	COMPARE OUTPUTS / WRITE GOLD
		//=====================================================================
		if (parameters.generate) {
//			fv_cpu_GOLD = d_fv_gpu[0].to_vector();
			writeGold(dim_cpu, parameters.output_gold, fv_cpu[0]);
		} else {
			timestamp = rad::mysecond();

			bool reloadFlag = false;
#pragma omp parallel for shared(reloadFlag, fv_cpu, fv_cpu_GOLD, log)
			for (uint32_t stream_idx = 0; stream_idx < parameters.nstreams;
					stream_idx++) {
//				fv_cpu[stream_idx] = d_fv_gpu[stream_idx].to_vector();
				auto error = kernel_caller.check_output_errors(
						parameters.verbose, stream_idx, fv_cpu[stream_idx],
						fv_cpu_GOLD, log);

#pragma omp atomic
				reloadFlag = reloadFlag || error;
			}

			if (reloadFlag) {
				readInput(dim_cpu, parameters.input_distances, rv_cpu,
						parameters.input_charges, qv_cpu,
						parameters.fault_injection);
				readGold(dim_cpu, parameters.output_gold, fv_cpu_GOLD);

				gpu_memory_unset(parameters, d_box_gpu, d_rv_gpu, d_qv_gpu,
						d_fv_gpu, d_fv_gold_gpu);
				gpu_memory_setup(parameters, d_box_gpu, box_cpu, d_rv_gpu,
						rv_cpu, d_qv_gpu, qv_cpu, d_fv_gpu, fv_cpu,
						d_fv_gold_gpu, fv_cpu_GOLD);
			}

			if (parameters.verbose)
				std::cout << "Gold check time: " << rad::mysecond() - timestamp
						<< std::endl;
		}

		//================= PERF
		// iterate for each neighbor of a box (number_nn)
		double flop = number_nn;
		// The last for iterate NUMBER_PAR_PER_BOX times
		flop *= NUMBER_PAR_PER_BOX;
		// the last for uses 46 operations plus 2 exp() functions
		flop *= 46;
		flop *= parameters.nstreams;
		double flops = flop / kernel_time;
		double outputpersec = dim_cpu.space_elem * 4 * parameters.nstreams
				/ kernel_time;
		double iteration_time = rad::mysecond() - globaltimer;

		if (parameters.verbose) {
			std::cout << "BOXES: " << dim_cpu.boxes1d_arg;
			std::cout << " BLOCK: " << NUMBER_THREADS;
			std::cout << " OUTPUT/S:" << outputpersec;
			std::cout << " FLOPS:" << flops;
			std::cout << " (GFLOPS:" << flops / 1.0e9 << ") ";
			std::cout << "Kernel time:" << kernel_time << std::endl;
			std::cout << "Copy time:" << cpy_time << std::endl;
			std::cout << "Iteration time: " << iteration_time << "s ("
					<< (kernel_time / iteration_time) * 100.0 << "% of Device)"
					<< std::endl;

			std::cout << "===================================" << std::endl;
		} else {
			std::cout << ".";
		}

	}

	if (parameters.generate) {
		kernel_caller.sync_half_t();
		std::cout << "Max element threshold "
				<< kernel_caller.get_max_threshold(fv_cpu) << std::endl;
	}

}

#endif /* SETUP_TEMPLATE_H_ */
