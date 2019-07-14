/*
 * persistent_lib.h
 *
 *  Created on: 14/06/2019
 *      Author: fernando
 */

#ifndef PERSISTENT_LIB_H_
#define PERSISTENT_LIB_H_

#include <iostream>
#include "cuda_utils.h"

namespace rad {

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#ifdef DEBUGPLIB
#define DEBUG 1
#endif

#define UINTCAST(x) ((unsigned int*)(x))

typedef unsigned int uint32;
typedef unsigned char byte;

volatile __device__ byte running;
volatile __device__ byte allow_threads_process;
volatile __device__ uint32 gpu_mutex;

#define DEFAULT_TIME_TO_WAIT 0.01

struct HostPersistentControler {
	cudaStream_t st;
	uint32 block_number;
	double best_time_to_wait;
	bool do_not_check_more;

	HostPersistentControler(dim3 grid_dim) {
		this->block_number = grid_dim.x * grid_dim.y * grid_dim.z;
		checkFrameworkErrors(
				cudaStreamCreateWithFlags(&this->st, cudaStreamNonBlocking));
		//make the kernel iterate in the loop
		this->set_running(1);
		//do not allow threads to process
		this->set_allow_threads_to_process(0);
		//reset the mutex
		this->set_mutex(0);
		this->best_time_to_wait = DEFAULT_TIME_TO_WAIT;
		this->do_not_check_more = false;
	}

	virtual ~HostPersistentControler() {
		this->end_kernel();
		checkFrameworkErrors(cudaStreamDestroy(this->st));
		checkFrameworkErrors(cudaDeviceSynchronize());

	}

	void sync_stream() {
		checkFrameworkErrors(cudaStreamSynchronize(this->st));
	}

	void start_kernel() {
		//do not allow threads to process
		this->set_allow_threads_to_process(0);
		//make the kernel iterate in the loop
		this->set_running(1);
	}

	void end_kernel() {
		//make the kernel exit
		this->set_running(0);
		//sync the device
		checkFrameworkErrors(cudaDeviceSynchronize());
	}

	void start_processing() {
		//set the mutex to 0
		this->set_mutex(0);
		//allow threads to process
		this->set_allow_threads_to_process(1);
	}

	void wait_gpu() {
		while (true) {
			uint32 counter = 0;

			checkFrameworkErrors(
					cudaMemcpyFromSymbolAsync(&counter, gpu_mutex,
							sizeof(uint32), 0, cudaMemcpyDeviceToHost,
							this->st));
			checkFrameworkErrors(cudaStreamSynchronize(this->st));
#ifdef DEBUG
			std::cout << "FINISHED " << counter << " " << this->block_number << " " <<
					best_time_to_wait << std::endl;
#endif
			//it saves 1 second
			if (this->block_number <= counter) {
				this->do_not_check_more = true;
				break;
			}
			rad::sleep(this->best_time_to_wait);

			if(this->do_not_check_more == false)
				this->best_time_to_wait *= 2;
		}
		checkFrameworkErrors(cudaGetLastError());
	}

	void end_processing() {
		//do not allow threads to process
		this->set_allow_threads_to_process(0);
	}

	void process_data_on_kernel() {
		this->start_processing();
#ifdef DEBUG
		std::cout << "PROCESS STARTED, now waiting" << std::endl;
#endif
		this->wait_gpu();
		this->end_processing();
#ifdef DEBUG
		std::cout << "PROCESS FINISHED" << std::endl;
#endif
	}

private:
	void set_allow_threads_to_process(byte value) {
		checkFrameworkErrors(
				cudaMemcpyToSymbolAsync(allow_threads_process, &value, sizeof(byte), 0,
						cudaMemcpyHostToDevice, this->st));
		checkFrameworkErrors(cudaGetLastError());
		checkFrameworkErrors(cudaStreamSynchronize(this->st));
#ifdef DEBUG
		std::cout << "ALLOW_THREADS_PROCESS set to " << bool(value)
				<< std::endl;
#endif
	}

	void set_running(byte value) {
		checkFrameworkErrors(
				cudaMemcpyToSymbolAsync(running, &value, sizeof(byte), 0,
						cudaMemcpyHostToDevice, this->st));
		checkFrameworkErrors(cudaGetLastError());
		checkFrameworkErrors(cudaStreamSynchronize(this->st));

#ifdef DEBUG
		std::cout << "RUNNING set to " << bool(value) << std::endl;
#endif
	}

	void set_mutex(uint32 value) {
		checkFrameworkErrors(
				cudaMemcpyToSymbolAsync(gpu_mutex, &value, sizeof(uint32), 0,
						cudaMemcpyHostToDevice, this->st));
		checkFrameworkErrors(cudaGetLastError());
		checkFrameworkErrors(cudaStreamSynchronize(this->st));
#ifdef DEBUG
		std::cout << "GPU_MUTEX set to " << value << std::endl;
#endif
	}
};

struct PersistentKernel {
	uint32 blocks_to_synch;
	uint32 tid_in_block;
	bool it_has_processed;

	__device__ PersistentKernel() {
		//thread ID in a block
		this->tid_in_block = threadIdx.x + threadIdx.y + threadIdx.z;
		this->blocks_to_synch = gridDim.x * gridDim.y * gridDim.z;
		this->it_has_processed = false;
	}

	__device__ void wait_for_work() {
//		__syncthreads();
//
//		// only thread 0 is used for synchronization
//		if (this->tid_in_block == 0) {
//			while (running == 1) {
//				if (gpu_mutex == 0) {
//					break;
//				}
//			}
//		}
//		__syncthreads();
		if(gpu_mutex == 0)
			this->it_has_processed = false;
	}

	__device__ void iteration_finished() {
		__syncthreads();
		// only thread 0 is used for synchronization
		if (this->tid_in_block == 0) {
			atomicAdd(UINTCAST(&gpu_mutex), 1);
			//only when all blocks add 1 to g_mutex will
			//g_mutex equal to blocks_to_synch
			while (gpu_mutex < this->blocks_to_synch) {
				if (running == 0) {
					break;
				}
			}
		}
		this->it_has_processed = true;
		__syncthreads();
	}

	__device__ bool is_able_to_process() {
		return (allow_threads_process == 1 && this->it_has_processed == false);
	}

	__device__ bool keep_working() {
		return (running == 1);
	}

};
}

#endif /* PERSISTENT_LIB_H_ */
