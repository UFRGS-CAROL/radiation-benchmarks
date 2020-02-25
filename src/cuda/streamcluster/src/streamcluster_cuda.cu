/***********************************************
 streamcluster_cuda.cu
 : parallelized code of streamcluster

 - original code from PARSEC Benchmark Suite
 - parallelization with CUDA API has been applied by

 Shawn Sang-Ha Lee - sl4ge@virginia.edu
 University of Virginia
 Department of Electrical and Computer Engineering
 Department of Computer Science

 ***********************************************/

#include <vector>
#include <iostream>
#include <fstream>

#include "device_vector.h"

#include "cuda_utils.h"
#include "streamcluster.h"

#define THREADS_PER_BLOCK 1024 //512#define MAXBLOCKS 65536
//=======================================
// Euclidean Distance
//=======================================
__device__ float d_dist(int p1, int p2, int num, int dim, float *coord_d) {
	float retval = 0.0;
	for (int i = 0; i < dim; i++) {
		float tmp = coord_d[(i * num) + p1] - coord_d[(i * num) + p2];
		retval += tmp * tmp;
	}
	return retval;
}

//=======================================
// Kernel - Compute Cost
//=======================================
__global__ void kernel_compute_cost(int num, int dim, long x, Point *p, int K,
		int stride, float *coord_d, float *work_mem_d, int *center_table_d,
		bool *switch_membership_d) {
	// block ID and global thread ID
	const int bid = blockIdx.x + gridDim.x * blockIdx.y;
	const int tid = blockDim.x * bid + threadIdx.x;

	if (tid < num) {
		float *lower = &work_mem_d[tid * stride];

		// cost between this point and point[x]: euclidean distance multiplied by weight
		float x_cost = d_dist(tid, x, num, dim, coord_d) * p[tid].weight;

		// if computed cost is less then original (it saves), mark it as to reassign
		if (x_cost < p[tid].cost) {
			switch_membership_d[tid] = 1;
			lower[K] += x_cost - p[tid].cost;
		}
		// if computed cost is larger, save the difference
		else {
			lower[center_table_d[p[tid].assign]] += p[tid].cost - x_cost;
		}
	}
}

//=======================================
// pgain Entry - CUDA SETUP + CUDA CALL
//=======================================
float pgain(long x, Points *points, float z, long int *numcenters, int kmax,
		bool *is_center, int *center_table, bool *switch_membership,
		bool isCoordChanged, double *serial_t, double *cpu_to_gpu_t,
		double *gpu_to_cpu_t, double *alloc_t, double *kernel_t,
		double *free_t) {
	static int iter = 0;		// counter for total# of iteration

	int stride = *numcenters + 1;			// size of each work_mem segment
	int K = *numcenters;				// number of centers
	int num = points->num;				// number of points
	int dim = points->dim;				// number of dimension
	int nThread = num;			// number of threads == number of data points

	//=========================================
	// ALLOCATE HOST MEMORY + DATA PREPARATION
	//=========================================
	static std::vector<float> work_mem_h(stride * (nThread + 1));

	// Only on the first iteration
	static std::vector<float> coord_h(num * dim);

	// build center-index table
	int count = 0;
	for (int i = 0; i < num; i++) {
		if (is_center[i]) {
			center_table[i] = count++;
		}
	}

	// Extract 'coord'
	// Only if first iteration OR coord has changed
	if (isCoordChanged || iter == 0) {
		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < num; j++) {
				coord_h[(num * i) + j] = points->p[j].coord[i];
			}
		}
	}

	//=======================================
	// ALLOCATE GPU MEMORY
	//=======================================
	rad::DeviceVector<float> work_mem_d(stride * (nThread + 1));

	// Only on the first iteration
	// device memory
	static rad::DeviceVector<int> center_table_d(num);
	static rad::DeviceVector<bool> switch_membership_d(num);
	static rad::DeviceVector<Point> p(num);
	static rad::DeviceVector<float> coord_d(num * dim);

	//=======================================
	// CPU-TO-GPU MEMORY COPY
	//=======================================
	// Only if first iteration OR coord has changed
	if (isCoordChanged || iter == 0) {
		coord_d = coord_h;
	}

	center_table_d.fill_n(center_table, num);

	p.fill_n(points->p, num);

	switch_membership_d.clear();
	work_mem_d.clear();

	//=======================================
	// KERNEL: CALCULATE COST
	//=======================================
	// Determine the number of thread blocks in the x- and y-dimension
	int num_blocks = (int) ((float) (num + THREADS_PER_BLOCK - 1)
			/ (float) THREADS_PER_BLOCK);
	int num_blocks_y = (int) ((float) (num_blocks + MAXBLOCKS - 1)
			/ (float) MAXBLOCKS);
	int num_blocks_x = (int) ((float) (num_blocks + num_blocks_y - 1)
			/ (float) num_blocks_y);
	dim3 grid_size(num_blocks_x, num_blocks_y, 1);

	kernel_compute_cost<<<grid_size, THREADS_PER_BLOCK>>>(num,// in:	# of data
			dim,					// in:	dimension of point coordinates
			x,						// in:	point to open a center at
			p.data(),						// in:	data point array
			K,						// in:	number of centers
			stride,					// in:  size of each work_mem segment
			coord_d.data(),				// in:	array of point coordinates
			work_mem_d.data(),				// out:	cost and lower field array
			center_table_d.data(),			// in:	center index table
			switch_membership_d.data()		// out:  changes in membership
			);
	rad::checkFrameworkErrors(cudaDeviceSynchronize());
	rad::checkFrameworkErrors(cudaGetLastError());

	//=======================================
	// GPU-TO-CPU MEMORY COPY
	//=======================================
	work_mem_d.to_vector(work_mem_h);
	switch_membership_d.get_n(switch_membership, num);

	//=======================================
	// CPU (SERIAL) WORK
	//=======================================
	int number_of_centers_to_close = 0;
	float gl_cost_of_opening_x = z;
	float *gl_lower = &work_mem_h[stride * nThread];
	// compute the number of centers to close if we are to open i
	for (int i = 0; i < num; i++) {
		if (is_center[i]) {
			float low = z;
			for (int j = 0; j < num; j++) {
				low += work_mem_h[j * stride + center_table[i]];
			}

			gl_lower[center_table[i]] = low;

			if (low > 0) {
				++number_of_centers_to_close;
				work_mem_h[i * stride + K] -= low;
			}
		}
		gl_cost_of_opening_x += work_mem_h[i * stride + K];
	}

	//if opening a center at x saves cost (i.e. cost is negative) do so; otherwise, do nothing
	if (gl_cost_of_opening_x < 0) {
		for (int i = 0; i < num; i++) {
			bool close_center = gl_lower[center_table[points->p[i].assign]] > 0;
			if (switch_membership[i] || close_center) {
				points->p[i].cost = dist(points->p[i], points->p[x], dim)
						* points->p[i].weight;
				points->p[i].assign = x;
			}
		}

		for (int i = 0; i < num; i++) {
			if (is_center[i] && gl_lower[center_table[i]] > 0) {
				is_center[i] = false;
			}
		}

		if (x >= 0 && x < num) {
			is_center[x] = true;
		}
		*numcenters = *numcenters + 1 - number_of_centers_to_close;
	} else {
		gl_cost_of_opening_x = 0;
	}

	iter++;
	return -gl_cost_of_opening_x;
}
