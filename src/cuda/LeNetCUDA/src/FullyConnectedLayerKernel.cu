/*
 * FullyConnectedLayerKernel.cu
 *
 *  Created on: Jun 6, 2017
 *      Author: carol
 */

#include "FullyConnectedLayerKernel.h"
#include "cudaUtil.h"

__global__ void forward_gpu_kernel(float *t){

}


void forward_gpu(FullyConnectedLayer l){
	cudaError_t ret = cudaDeviceSynchronize();
	CUDA_CHECK_RETURN(ret);
}
