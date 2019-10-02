//============================================================================
//	UPDATE
//============================================================================

//	14 APR 2011 Lukasz G. Szafaryn
//  2014-2018 Caio Lunardi
//  2018 Fernando Fernandes dos Santos

#include <iostream>
#include <cuda_fp16.h>
#include <random>

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

#include "cuda_utils.h"
#include "Parameters.h"
#include "Log.h"
#include "types.h"
#include "common.h"
#include "nondmr_kernels.h"
#include "dmr_kernels.h"
#include "File.h"

template<typename tested_type>
void generateInput(dim_str dim_cpu, std::string& input_distances,
		std::vector<FOUR_VECTOR<tested_type>>& rv_cpu,
		std::string& input_charges, std::vector<tested_type>& qv_cpu) {
	// random generator seed set to random value - time in this case
	std::cout << ("Generating input...\n");

	// get a number in the range 0.1 - 1.0
	std::random_device rd; //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<tested_type> dis(0.1, 1.0);

	rv_cpu.resize(dim_cpu.space_elem);
	qv_cpu.resize(dim_cpu.space_elem);
	for (auto& rv_cpu_i : rv_cpu) {
		rv_cpu_i.v = tested_type(dis(gen));
		rv_cpu_i.x = tested_type(dis(gen));
		rv_cpu_i.y = tested_type(dis(gen));
		rv_cpu_i.z = tested_type(dis(gen));
	}

	std::cout << "TESTE " << rv_cpu[35].x << std::endl;

	if (File<FOUR_VECTOR<tested_type>>::write_to_file(input_distances,
			rv_cpu)) {
		error("error writing rv_cpu from file\n");
	}

	for (auto& qv_cpu_i : qv_cpu) {
		// get a number in the range 0.1 - 1.0
		qv_cpu_i = tested_type(dis(gen));
	}

	if (File<tested_type>::write_to_file(input_charges, qv_cpu)) {
		error("error writing qv_cpu from file\n");
	}

}

template<typename tested_type>
void readInput(dim_str dim_cpu, std::string& input_distances,
		std::vector<FOUR_VECTOR<tested_type>>& rv_cpu,
		std::string& input_charges, std::vector<tested_type>& qv_cpu,
		int fault_injection) {

	rv_cpu.resize(dim_cpu.space_elem);
	qv_cpu.resize(dim_cpu.space_elem);

	if (File<FOUR_VECTOR<tested_type>>::read_from_file(input_distances,
			rv_cpu)) {
		error("error reading rv_cpu from file\n");
	}

	if (File<tested_type>::read_from_file(input_charges, qv_cpu)) {
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

template<typename tested_type>
void readGold(dim_str dim_cpu, std::string& output_gold,
		std::vector<FOUR_VECTOR<tested_type>>& fv_cpu_GOLD) {
	if (File<FOUR_VECTOR<tested_type>>::read_from_file(output_gold,
			fv_cpu_GOLD)) {
		error("error reading fv_cpu_GOLD from file\n");
	}
}

template<typename tested_type>
void writeGold(dim_str dim_cpu, std::string& output_gold,
		std::vector<FOUR_VECTOR<tested_type>>& fv_cpu) {

	int number_zeros = 0;
	for (auto& fv_cpu_i : fv_cpu) {
		if (fv_cpu_i.v == tested_type(0.0))
			number_zeros++;
		if (fv_cpu_i.x == tested_type(0.0))
			number_zeros++;
		if (fv_cpu_i.y == tested_type(0.0))
			number_zeros++;
		if (fv_cpu_i.z == tested_type(0.0))
			number_zeros++;
	}

	if (File<FOUR_VECTOR<tested_type>>::write_to_file(output_gold, fv_cpu)) {
		error("error writing fv_cpu from file\n");
	}

	std::cout << "Number of zeros " << number_zeros << std::endl;
}

template<typename tested_type>
void gpu_memory_setup(const Parameters& parameters,
		VectorOfDeviceVector<box_str>& d_box_gpu, std::vector<box_str>& box_cpu,
		VectorOfDeviceVector<FOUR_VECTOR<tested_type>>& d_rv_gpu,
		std::vector<FOUR_VECTOR<tested_type>>& rv_cpu,
		VectorOfDeviceVector<tested_type>& d_qv_gpu,
		std::vector<tested_type>& qv_cpu,
		VectorOfDeviceVector<FOUR_VECTOR<tested_type>>& d_fv_gpu,
		std::vector<std::vector<FOUR_VECTOR<tested_type>>>& fv_cpu,
rad::DeviceVector<FOUR_VECTOR<tested_type>>& d_fv_gold_gpu, std::vector<FOUR_VECTOR<tested_type>>& fv_cpu_GOLD) {

	for (int streamIdx = 0; streamIdx < parameters.nstreams; streamIdx++) {
		d_box_gpu[streamIdx] = box_cpu;
		d_rv_gpu[streamIdx] = rv_cpu;
		d_qv_gpu[streamIdx] = qv_cpu;
		d_fv_gpu[streamIdx] = fv_cpu[streamIdx];
	}

	if (parameters.gpu_check) {
		d_fv_gold_gpu = fv_cpu_GOLD;
	}
}

template<typename tested_type>
void gpu_memory_unset(const Parameters& parameters,
		VectorOfDeviceVector<box_str>& d_box_gpu,
		VectorOfDeviceVector<FOUR_VECTOR<tested_type>>& d_rv_gpu,
		VectorOfDeviceVector<tested_type>& d_qv_gpu,
		VectorOfDeviceVector<FOUR_VECTOR<tested_type>>& d_fv_gpu,
		rad::DeviceVector<FOUR_VECTOR<tested_type>>& d_fv_gold_gpu) {

	//=====================================================================
	//	GPU MEMORY DEALLOCATION
	//=====================================================================
	for (int streamIdx = 0; streamIdx < parameters.nstreams; streamIdx++) {
		d_rv_gpu[streamIdx].resize(0);
		d_qv_gpu[streamIdx].resize(0);
		d_fv_gpu[streamIdx].resize(0);
		d_box_gpu[streamIdx].resize(0);
	}
	if (parameters.gpu_check) {
		d_fv_gold_gpu.resize(0);
	}
}

// Returns true if no errors are found. False if otherwise.
// Set votedOutput pointer to retrieve the voted matrix
template<typename tested_type>
bool checkOutputErrors(int verbose, int streamIdx,
		std::vector<FOUR_VECTOR<tested_type>>& fv_cpu,
		std::vector<FOUR_VECTOR<tested_type>>& fv_cpu_GOLD, Log& log) {
	int host_errors = 0;

#pragma omp parallel for shared(host_errors)
	for (int i = 0; i < fv_cpu_GOLD.size(); i = i + 1) {
		auto valGold = fv_cpu_GOLD[i];
		auto valOutput = fv_cpu[i];
		if (valGold != valOutput) {
#pragma omp critical
			{
				char error_detail[500];
				host_errors++;

				snprintf(error_detail, 500,
						"stream: %d, p: [%d], v_r: %1.20e, v_e: %1.20e, x_r: %1.20e, "
								"x_e: %1.20e, y_r: %1.20e, y_e: %1.20e, z_r: %1.20e, z_e: %1.20e",
						streamIdx, i, (double) valOutput.v, (double) valGold.v,
						(double) valOutput.x, (double) valGold.x,
						(double) valOutput.y, (double) valGold.y,
						(double) valOutput.z, (double) valGold.z);
				if (verbose && (host_errors < 10))
					std::cout << error_detail << std::endl;

				log.log_error_detail(std::string(error_detail));
			}
		}
	}

	// printf("numErrors:%d", host_errors);

	log.update_errors(host_errors);

	if (host_errors != 0)
		printf("#");

	return (host_errors == 0);
}

template<typename tested_type>
void setup_execution(Parameters& parameters, Log& log) {
	//=====================================================================
	//	CPU/MCPU VARIABLES
	//=====================================================================
	// timer
	double timestamp;

	// system memory
	par_str<tested_type> par_cpu;
	dim_str dim_cpu;
	std::vector<box_str> box_cpu;
	std::vector<FOUR_VECTOR<tested_type>> rv_cpu;
	std::vector<tested_type> qv_cpu;
	std::vector<FOUR_VECTOR<tested_type>> fv_cpu_GOLD;
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
	dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR<tested_type> );
	dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(tested_type);
	// box array
	dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);
	//=====================================================================
	//	SYSTEM MEMORY
	//=====================================================================
	// prepare host memory to receive kernel output
	// output (forces)
	std::vector<std::vector<FOUR_VECTOR<tested_type>>> fv_cpu(parameters.nstreams);

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
	VectorOfDeviceVector<FOUR_VECTOR<tested_type>> d_rv_gpu(
			parameters.nstreams);
	VectorOfDeviceVector<tested_type> d_qv_gpu(parameters.nstreams);
	VectorOfDeviceVector<FOUR_VECTOR<tested_type>> d_fv_gpu(
			parameters.nstreams);
	rad::DeviceVector<FOUR_VECTOR<tested_type>> d_fv_gold_gpu;
	//=====================================================================
	//	GPU MEMORY SETUP
	//=====================================================================
	gpu_memory_setup(parameters, d_box_gpu, box_cpu, d_rv_gpu, rv_cpu, d_qv_gpu,
			qv_cpu, d_fv_gpu, fv_cpu, d_fv_gold_gpu, fv_cpu_GOLD);

	//LOOP START
	for (int loop = 0; loop < parameters.iterations; loop++) {

		if (parameters.verbose)
			printf("======== Iteration #%06u ========\n", loop);

		double globaltimer = rad::mysecond();
		timestamp = rad::mysecond();

		//=====================================================================
		//	GPU SETUP
		//=====================================================================
		for (int streamIdx = 0; streamIdx < parameters.nstreams; streamIdx++) {
			auto& it = fv_cpu[streamIdx];
			std::fill(it.begin(), it.end(), FOUR_VECTOR<tested_type>());
			d_fv_gpu[streamIdx].clear();
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
			kernel_gpu_cuda<<<blocks, threads, 0, streams[streamIdx].stream>>>(
					par_cpu, dim_cpu, d_box_gpu[streamIdx].data(),
					d_rv_gpu[streamIdx].data(), d_qv_gpu[streamIdx].data(),
					d_fv_gpu[streamIdx].data());
			rad::checkFrameworkErrors(cudaPeekAtLastError());
		}

		for (auto& st : streams) {
			st.sync();
			rad::checkFrameworkErrors(cudaPeekAtLastError());
		}

		log.end_iteration();
		kernel_time = rad::mysecond() - kernel_time;

		//=====================================================================
		//	COMPARE OUTPUTS / WRITE GOLD
		//=====================================================================
		if (parameters.generate) {
			fv_cpu_GOLD = d_fv_gpu[0].to_vector();
			writeGold(dim_cpu, parameters.output_gold, fv_cpu_GOLD);
		} else {
			timestamp = rad::mysecond();

			bool reloadFlag = false;
#pragma omp parallel for shared(reloadFlag, fv_cpu, fv_cpu_GOLD, log)
			for (int streamIdx = 0; streamIdx < parameters.nstreams;
					streamIdx++) {
				fv_cpu[streamIdx] = d_fv_gpu[streamIdx].to_vector();
				reloadFlag = reloadFlag
						|| checkOutputErrors(parameters.verbose, streamIdx,
								fv_cpu[streamIdx], fv_cpu_GOLD, log);
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
				std::cout << "Gold check time: " << rad::mysecond() - timestamp;
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

			std::cout << "Iteration time: " << iteration_time << "s ("
					<< (kernel_time / iteration_time) * 100.0 << "% of Device)"
					<< std::endl;

			std::cout << "===================================" << std::endl;
		} else {
			std::cout << ".";
		}

	}

}

//=============================================================================
//	MAIN FUNCTION
//=============================================================================

int main(int argc, char *argv[]) {
	std::cout << std::boolalpha;
	//=====================================================================
	//	CPU/MCPU VARIABLES
	//=====================================================================
	Parameters parameters(argc, argv);
	Log log;

	std::cout << parameters << std::endl;
	auto test_precision_description = "float";

	std::string test_info = std::string("type:") + test_precision_description
			+ "-precision streams:" + std::to_string(parameters.nstreams)
			+ " boxes:" + std::to_string(parameters.boxes) + " block_size:"
			+ std::to_string(NUMBER_THREADS);

	std::string test_name = std::string("cuda_") + test_precision_description
			+ "_lava";
	std::cout << "=================================" << std::endl;
	std::cout << test_precision_description << " " << test_name << std::endl;
	std::cout << "=================================" << std::endl;

	// timer
	if (!parameters.generate) {
		log = Log(test_name, test_info);
	}

	log.set_max_errors_iter(
	MAX_LOGGED_ERRORS_PER_STREAM * parameters.nstreams + 32);

#ifdef BUILDPROFILER

	std::string log_file_name = log.get_log_file_name();
	if(parameters.generate) {
		log_file_name = "/tmp/generate.log";
	}

	std::shared_ptr<rad::Profiler> profiler_thread = std::make_shared<rad::OBJTYPE>(0, log_file_name);

//START PROFILER THREAD
	profiler_thread->start_profile();
#endif

	setup_execution<float>(parameters, log);

#ifdef BUILDPROFILER
	profiler_thread->end_profile();
#endif

	return 0;
}
