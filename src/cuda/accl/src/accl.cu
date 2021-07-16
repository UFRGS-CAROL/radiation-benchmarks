/*
 * -- NUPAR: A Benchmark Suite for Modern GPU Architectures
 *    NUPAR - 2 December 2014
 *    Fanny Nina-Paravecino
 *    Northeastern University
 *    NUCAR Research Laboratory
 *
 * Redistribution  and  use in  source and binary forms, with or without
 * modification, are  permitted provided  that the following  conditions
 * are met:
 *
 * 1. Redistributions  of  source  code  must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce  the above copyright
 * notice, this list of conditions,  and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. All  advertising  materials  mentioning  features  or  use of this
 * software must display the following acknowledgement:
 * This  product  includes  software  developed  at  the Northeastern U.
 *
 * 4. The name of the  University,  the name of the  Laboratory,  or the
 * names  of  its  contributors  may  not  be used to endorse or promote
 * products  derived   from   this  software  without  specific  written
 * permission.
 *
 * -- Disclaimer:
 *
 * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * ---------------------------------------------------------------------
 */
#include <cstdio>
#include <cmath>

#include "cuda_runtime_api.h"

#include "generic_log.h"
#include "device_vector.h"
#include "cuda_utils.h"

#include "multi_compiler_analysis.h"

//#define THREADSX 16
//#define THREADSY 16
//#define THREADS 512

__global__ void findSpansKernel(int *out, int *components, const int *in, const int rows,
		const int cols) {
	uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint colsSpans = ((cols + 2 - 1) / 2) * 2;
	int current;
	uint colsComponents = colsSpans / 2;
	bool flagFirst = true;
	int indexOut = 0;
	int indexComp = 0;
	uint comp = i * colsComponents;
	if (i < rows) {
		for (int j = 0; j < cols; j++) {
			if (flagFirst && in[i * cols + j] > 0) {
				current = in[i * cols + j];
				out[i * colsSpans + indexOut] = j;
				indexOut++;
				flagFirst = false;
			}
			if (!flagFirst && in[i * cols + j] != current) {
				out[i * colsSpans + indexOut] = j - 1;
				indexOut++;
				flagFirst = true;
				/*add the respective label*/
				components[i * colsComponents + indexComp] = comp;
				indexComp++;
				comp++;
			}
		}
		if (!flagFirst) {
			out[i * colsSpans + indexOut] = cols - 1;
			/*add the respective label*/
			components[i * colsComponents + indexComp] = comp;
		}
	}
}

//__global__ void relabelKernel(int *components, int previousLabel, int newLabel,
//		const int colsComponents) {
//	uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
//	uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
//
//	if (components[i * colsComponents + j] == previousLabel) {
//		components[i * colsComponents + j] = newLabel;
//	}
//}

//__global__ void relabel2Kernel(int *components, int previousLabel, int newLabel,
//		const int colsComponents, const int idx, const int frameRows) {
//	uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
//	uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
//	i = i * colsComponents + j;
//	i = i + (colsComponents * frameRows * idx);
//	if (components[i] == previousLabel) {
//		components[i] = newLabel;
//	}
//
//}
__global__ void relabelUnrollKernel(int *components, int previousLabel, int newLabel,
		const int colsComponents, const int idx, const int frameRows, const int factor) {
	uint id_i_child = (blockIdx.x * blockDim.x) + threadIdx.x;
	id_i_child = id_i_child + (frameRows * idx);
	uint id_j_child = (blockIdx.y * blockDim.y) + threadIdx.y;
	id_j_child = (colsComponents / factor) * id_j_child;
	uint i = id_i_child;
	for (int j = id_j_child; j < (colsComponents / factor); j++) {
		if (components[i * colsComponents + j] == previousLabel) {
			components[i * colsComponents + j] = newLabel;
		}
	}
}

__global__ void mergeSpansKernel(int *components, int *spans, const int rows, const int cols,
		const int frameRows) {
	uint idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint colsSpans = ((cols + 2 - 1) / 2) * 2;
	uint colsComponents = colsSpans / 2;
	/*Merge Spans*/
	int startX, endX, newStartX, newEndX;
	int label = -1;
	/*threads and blocks need to relabel the components labels*/
	int threads = 16;
	const int factor = 2;

	/*--------For 256, 512--------*/
	dim3 threadsPerBlockUnrollRelabel(threads * threads);
	dim3 numBlocksUnrollRelabel((frameRows * factor) / (threads * threads));
	/*-----------------*/

	for (uint i = idx * frameRows; i < ((idx * frameRows) + frameRows) - 1; i++) //compute until penultimate row, since we need the below row to compare
			{
		for (int j = 0; j < colsSpans - 1 && spans[i * colsSpans + j] >= 0; j = j + 2) //verify if there is a Span available
				{
			startX = spans[i * colsSpans + j];
			endX = spans[i * colsSpans + j + 1];
			uint newI = i + 1; //line below
			for (int k = 0; k < colsSpans - 1 && spans[newI * colsSpans + k] >= 0; k = k + 2) //verify if there is a New Span available
					{
				newStartX = spans[newI * colsSpans + k];
				newEndX = spans[newI * colsSpans + k + 1];
				if (startX <= newEndX && endX >= newStartX) //Merge components
						{
					label = components[i * (colsSpans / 2) + (j / 2)]; //choose the startSpan label
					relabelUnrollKernel<<<numBlocksUnrollRelabel, threadsPerBlockUnrollRelabel>>>(
							components, components[newI * (colsSpans / 2) + (k / 2)], label,
							colsComponents, idx, frameRows, factor);

					cudaDeviceSynchronize();
					cudaGetLastError();
//					cudaError_t err = cudaGetLastError();
//					if (err != cudaSuccess)
//						printf("\tError:%s \n", (char) err);
				}
				__syncthreads();
			}
		}
	}
}

/***
 * 	rad::DeviceVector<int> devIn = in;
 rad::DeviceVector<int> devComponents = components;
 rad::DeviceVector<int> devOut = out;
 */

double acclCuda(rad::DeviceVector<int> &devOut, rad::DeviceVector<int> &devComponents,
		const rad::DeviceVector<int> &devIn, uint nFrames, uint nFramsPerStream, const int rows,
		const int cols, int logs_active, rad::Log &log, std::vector<cudaStream_t> &streams) {
	auto acc = rad::mysecond();
	const int colsSpans = ((cols + 2 - 1) / 2) * 2; /*ceil(cols/2)*2*/
	const int colsComponents = colsSpans / 2;

	/*compute sizes of matrices*/
//	const int sizeIn = rows * cols;
//	const int sizeComponents = colsComponents * rows;
//	const int sizeOut = colsSpans * rows;
	/*Block and Grid size*/
	int blockSize;
	int minGridSize;
	int gridSize;

	/*Frame Info*/
	const int frameRows = rows / nFrames;

	/*Streams Information*/
	uint nStreams = nFrames / nFramsPerStream;
	int rowsOccupancyMax = frameRows * nFramsPerStream;
	rad::checkFrameworkErrors(
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, findSpansKernel, 0,
					rowsOccupancyMax));
	// printf("Best Kernel Size\n");
	// printf("-----------------\n");
	// printf("\t Minimum gridSize to acchieve high occupancy: %d\n", minGridSize);
	// printf("\t Block Size: %d\n", blockSize);
	// printf("\t Rows Max Occupancy: %d\n", rowsOccupancyMax);

//    cudaEvent_t start, stop;
//    float time;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);

	/* Choose which GPU to run on, change this on a multi-GPU system.*/
	rad::checkFrameworkErrors(cudaSetDevice(0));

	/* Allocate GPU buffers for three vectors (two input, one output)*/
//	rad::checkFrameworkErrors(cudaMalloc((void** )&devOut, sizeOut * sizeof(int)));
//	rad::checkFrameworkErrors(cudaMalloc((void** )&devComponents, sizeComponents * sizeof(int)));
//	rad::checkFrameworkErrors(cudaMalloc((void** )&devIn, sizeIn * sizeof(int)));
	/* Copy input vectors from host memory to GPU buffers*/
//	rad::checkFrameworkErrors(cudaMemcpy(devIn, in, sizeIn * sizeof(int), cudaMemcpyHostToDevice));
//	rad::checkFrameworkErrors(cudaMemcpy(devComponents, components, sizeComponents * sizeof(int),		cudaMemcpyHostToDevice));
//	rad::checkFrameworkErrors(cudaMemcpy(devOut, out, sizeOut * sizeof(int),		cudaMemcpyHostToDevice));
//	rad::DeviceVector<int> devIn = in;
//	rad::DeviceVector<int> devComponents = components;
//	rad::DeviceVector<int> devOut = out;
	/*launch streams*/

	/*variables for streaming*/
	const int frameSpansSize = rows / nStreams * colsSpans;
	const int frameCompSize = rows / nStreams * colsComponents;

	/* Round up according to array size */
	blockSize = 256;
	gridSize = (rows / nStreams) / blockSize;
	//gridSize = rows/blockSize;

	/* Launch a kernel on the GPU with one thread for each element*/
	// printf("Number of frames processed: %d\n", nFrames);
	// printf("Number of streams created: %d\n", nStreams);
//    cudaEventRecord(start, 0); /*measure time*
	if (logs_active)
		log.start_iteration();
	for (int i = 0; i < nStreams; ++i) {
		findSpansKernel<<<gridSize, blockSize, 0, streams[i]>>>(devOut.data() + i * frameSpansSize,
				devComponents.data() + i * frameCompSize, devIn.data() + i * frameSpansSize, rows,
				cols);

		/*Merge Spans*/
		mergeSpansKernel<<<1, nFramsPerStream, 0, streams[i]>>>(
				devComponents.data() + i * frameCompSize, devOut.data() + i * frameSpansSize, rows,
				cols, frameRows);
	}
	rad::checkFrameworkErrors(cudaDeviceSynchronize());
	//rad::checkFrameworkErrors(cudaGetLastError());

	if (logs_active)
		log.end_iteration();
	/* Copy device to host*/
//	rad::checkFrameworkErrors(
//			cudaMemcpy(components, devComponents, sizeComponents * sizeof(int),
//					cudaMemcpyDeviceToHost));
//	rad::checkFrameworkErrors(
//			cudaMemcpy(out, devOut, sizeOut * sizeof(int),
//					cudaMemcpyDeviceToHost));
//	devComponents.to_vector(components);
//	devOut.to_vector(out);
//    cudaEventRecord(stop, 0);
//    cudaEventSynchronize(stop);
//
//    cudaEventElapsedTime(&time, start, stop);
	//printf ("Time kernel execution: %f ms\n", time);

	/* Analysis of occupancy*/
//    int maxActiveBlocks;
//    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks,
//                                                  findSpansKernel, blockSize, 0);
//    int device;
//    cudaDeviceProp props;
//    cudaGetDevice(&device);
//    cudaGetDeviceProperties(&props, device);
//	float occupancy = (maxActiveBlocks * blockSize / props.warpSize)
//			/ (float) (props.maxThreadsPerMultiProcessor / props.warpSize);
	// printf("Occupancy Results\n");
	// printf("-----------------\n");
	// printf("\t Block Size: %d\n", blockSize);
	// printf("\t Grid Size: %d\n", gridSize);
	// printf("\t Theoretical occupancy: %f\n", occupancy);
	/*Free*/
//	cudaFree(devOut);
//	cudaFree(devIn);
//	cudaFree(devComponents);
	return rad::mysecond();
}

std::string get_multi_compiler_header() {
	return rad::get_multi_compiler_header();
}
