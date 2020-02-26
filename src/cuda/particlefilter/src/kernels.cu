/*
 * kernels.cu
 *
 *  Created on: 25/02/2020
 *      Author: fernando
 */

#include "common.h"

/********************************
 * CALC LIKELIHOOD SUM
 * DETERMINES THE LIKELIHOOD SUM BASED ON THE FORMULA: SUM( (IK[IND] - 100)^2 - (IK[IND] - 228)^2)/ 100
 * param 1 I 3D matrix
 * param 2 current ind array
 * param 3 length of ind array
 * returns a float_t representing the sum
 ********************************/
__device__ float_t calcLikelihoodSum(unsigned char * I, int * ind, int numOnes,
		int index) {
	float_t likelihoodSum = 0.0;
	int x;
	for (x = 0; x < numOnes; x++)
		likelihoodSum += (pow((float_t) (I[ind[index * numOnes + x]] - 100), 2)
				- pow((float_t) (I[ind[index * numOnes + x]] - 228), 2)) / 50.0;
	return likelihoodSum;
}

/****************************
 CDF CALCULATE
 CALCULATES CDF
 param1 CDF
 param2 weights
 param3 Nparticles
 *****************************/
__device__ void cdfCalc(float_t * CDF, float_t * weights, int Nparticles) {
	int x;
	CDF[0] = weights[0];
	for (x = 1; x < Nparticles; x++) {
		CDF[x] = weights[x] + CDF[x - 1];
	}
}

/*****************************
 * RANDU
 * GENERATES A UNIFORM DISTRIBUTION
 * returns a float_t representing a randomily generated number from a uniform distribution with range [0, 1)
 ******************************/
__device__ float_t d_randu(int * seed, int index) {

	int M = INT_MAX;
	int A = 1103515245;
	int C = 12345;
	int num = A * seed[index] + C;
	seed[index] = num % M;

	return fabs(seed[index] / ((float_t) M));
}

__device__ float_t d_randn(int * seed, int index) {
	//Box-Muller algortihm
	float_t pi = 3.14159265358979323846;
	float_t u = d_randu(seed, index);
	float_t v = d_randu(seed, index);
	float_t cosine = cos(2 * pi * v);
	float_t rt = -2 * log(u);
	return sqrt(rt) * cosine;
}

/****************************
 UPDATE WEIGHTS
 UPDATES WEIGHTS
 param1 weights
 param2 likelihood
 param3 Nparcitles
 ****************************/
__device__ float_t updateWeights(float_t * weights, float_t * likelihood,
		int Nparticles) {
	int x;
	float_t sum = 0;
	for (x = 0; x < Nparticles; x++) {
		weights[x] = weights[x] * exp(likelihood[x]);
		sum += weights[x];
	}
	return sum;
}

__device__ int findIndexBin(float_t * CDF, int beginIndex, int endIndex,
		float_t value) {
	if (endIndex < beginIndex)
		return -1;
	int middleIndex;
	while (endIndex > beginIndex) {
		middleIndex = beginIndex + ((endIndex - beginIndex) / 2);
		if (CDF[middleIndex] >= value) {
			if (middleIndex == 0)
				return middleIndex;
			else if (CDF[middleIndex - 1] < value)
				return middleIndex;
			else if (CDF[middleIndex - 1] == value) {
				while (CDF[middleIndex] == value && middleIndex >= 0)
					middleIndex--;
				middleIndex++;
				return middleIndex;
			}
		}
		if (CDF[middleIndex] > value)
			endIndex = middleIndex - 1;
		else
			beginIndex = middleIndex + 1;
	}
	return -1;
}

/** added this function. was missing in original float_t version.
 * Takes in a float_t and returns an integer that approximates to that float_t
 * @return if the mantissa < .5 => return value < input value; else return value > input value
 */
__device__ float_t dev_round_float_t(float_t value) {
	int newValue = (int) (value);
	if (value - newValue < .5f)
		return newValue;
	else
		return newValue++;
}

/*****************************
 * CUDA Find Index Kernel Function to replace FindIndex
 * param1: arrayX
 * param2: arrayY
 * param3: CDF
 * param4: u
 * param5: xj
 * param6: yj
 * param7: weights
 * param8: Nparticles
 *****************************/
__global__ void find_index_kernel(float_t * arrayX, float_t * arrayY,
		float_t * CDF, float_t * u, float_t * xj, float_t * yj,
		float_t * weights, int Nparticles) {
	int block_id = blockIdx.x;
	int i = blockDim.x * block_id + threadIdx.x;

	if (i < Nparticles) {

		int index = -1;
		int x;

		for (x = 0; x < Nparticles; x++) {
			if (CDF[x] >= u[i]) {
				index = x;
				break;
			}
		}
		if (index == -1) {
			index = Nparticles - 1;
		}

		xj[i] = arrayX[index];
		yj[i] = arrayY[index];

		//weights[i] = 1 / ((float_t) (Nparticles)); //moved this code to the beginning of likelihood kernel

	}
	__syncthreads();
}

__global__ void normalize_weights_kernel(float_t * weights, int Nparticles,
		float_t* partial_sums, float_t * CDF, float_t * u, int * seed) {
	int block_id = blockIdx.x;
	int i = blockDim.x * block_id + threadIdx.x;
	__shared__ float_t u1, sumWeights;

	if (0 == threadIdx.x)
		sumWeights = partial_sums[0];

	__syncthreads();

	if (i < Nparticles) {
		weights[i] = weights[i] / sumWeights;
	}

	__syncthreads();

	if (i == 0) {
		cdfCalc(CDF, weights, Nparticles);
		u[0] = (1 / ((float_t) (Nparticles))) * d_randu(seed, i); // do this to allow all threads in all blocks to use the same u1
	}

	__syncthreads();

	if (0 == threadIdx.x)
		u1 = u[0];

	__syncthreads();

	if (i < Nparticles) {
		u[i] = u1 + i / ((float_t) (Nparticles));
	}
}

__global__ void sum_kernel(float_t* partial_sums, int Nparticles) {
	int block_id = blockIdx.x;
	int i = blockDim.x * block_id + threadIdx.x;

	if (i == 0) {
		int x;
		float_t sum = 0.0;
		int num_blocks = ceil(
				(float_t) Nparticles / (float_t) threads_per_block);
		for (x = 0; x < num_blocks; x++) {
			sum += partial_sums[x];
		}
		partial_sums[0] = sum;
	}
}

/*****************************
 * CUDA Likelihood Kernel Function to replace FindIndex
 * param1: arrayX
 * param2: arrayY
 * param2.5: CDF
 * param3: ind
 * param4: objxy
 * param5: likelihood
 * param6: I
 * param6.5: u
 * param6.75: weights
 * param7: Nparticles
 * param8: countOnes
 * param9: max_size
 * param10: k
 * param11: IszY
 * param12: Nfr
 *****************************/
__global__ void likelihood_kernel(float_t * arrayX, float_t * arrayY,
		float_t * xj, float_t * yj, float_t * CDF, int * ind, int * objxy,
		float_t * likelihood, unsigned char * I, float_t * u, float_t * weights,
		int Nparticles, int countOnes, int max_size, int k, int IszY, int Nfr,
		int *seed, float_t* partial_sums) {
	int block_id = blockIdx.x;
	int i = blockDim.x * block_id + threadIdx.x;
	int y;

	int indX, indY;
	__shared__ float_t buffer[512];
	if (i < Nparticles) {
		arrayX[i] = xj[i];
		arrayY[i] = yj[i];

		weights[i] = 1 / ((float_t) (Nparticles)); //Donnie - moved this line from end of find_index_kernel to prevent all weights from being reset before calculating position on final iteration.

		arrayX[i] = arrayX[i] + 1.0 + 5.0 * d_randn(seed, i);
		arrayY[i] = arrayY[i] - 2.0 + 2.0 * d_randn(seed, i);

	}

	__syncthreads();

	if (i < Nparticles) {
		for (y = 0; y < countOnes; y++) {
			//added dev_round_float_t() to be consistent with roundDouble
			indX = dev_round_float_t(arrayX[i]) + objxy[y * 2 + 1];
			indY = dev_round_float_t(arrayY[i]) + objxy[y * 2];

			ind[i * countOnes + y] = abs(indX * IszY * Nfr + indY * Nfr + k);
			if (ind[i * countOnes + y] >= max_size)
				ind[i * countOnes + y] = 0;
		}
		likelihood[i] = calcLikelihoodSum(I, ind, countOnes, i);

		likelihood[i] = likelihood[i] / countOnes;

		weights[i] = weights[i] * exp(likelihood[i]); //Donnie Newell - added the missing exponential function call

	}

	buffer[threadIdx.x] = 0.0;

	__syncthreads();

	if (i < Nparticles) {

		buffer[threadIdx.x] = weights[i];
	}

	__syncthreads();

	//this doesn't account for the last block that isn't full
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			buffer[threadIdx.x] += buffer[threadIdx.x + s];
		}

		__syncthreads();

	}
	if (threadIdx.x == 0) {
		partial_sums[blockIdx.x] = buffer[0];
	}

	__syncthreads();

}
