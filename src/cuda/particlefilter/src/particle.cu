#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include <sys/time.h>

#include "common.h"

/*****************************
 *GET_TIME
 *returns a long int representing the time
 *****************************/
long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}
// Returns the number of seconds elapsed between the two specified times

float_t elapsed_time(long long start_time, long long end_time) {
	return (float_t) (end_time - start_time) / (1000 * 1000);
}

/*****************************
 * CHECK_ERROR
 * Checks for CUDA errors and prints them to the screen to help with
 * debugging of CUDA related programming
 *****************************/
void check_error(cudaError e) {
	if (e != cudaSuccess) {
		printf("\nCUDA error: %s\n", cudaGetErrorString(e));
		exit(1);
	}
}

void cuda_print_float_t_array(float_t *array_GPU, size_t size) {
	//allocate temporary array for printing
	float_t* mem = (float_t*) malloc(sizeof(float_t) * size);

	//transfer data from device
	cudaMemcpy(mem, array_GPU, sizeof(float_t) * size, cudaMemcpyDeviceToHost);

	printf("PRINTING ARRAY VALUES\n");
	//print values in memory
	for (size_t i = 0; i < size; ++i) {
		printf("[%d]:%0.6f\n", i, mem[i]);
	}
	printf("FINISHED PRINTING ARRAY VALUES\n");

	//clean up memory
	free(mem);
	mem = NULL;
}

/**
 * Generates a uniformly distributed random number using the provided seed and GCC's settings for the Linear Congruential Generator (LCG)
 * @see http://en.wikipedia.org/wiki/Linear_congruential_generator
 * @note This function is thread-safe
 * @param seed The seed array
 * @param index The specific index of the seed to be advanced
 * @return a uniformly distributed number [0, 1)
 */

float_t randu(int * seed, int index) {
	int num = A * seed[index] + C;
	seed[index] = num % M;
	return fabs(seed[index] / ((float_t) M));
}

/**
 * Generates a normally distributed random number using the Box-Muller transformation
 * @note This function is thread-safe
 * @param seed The seed array
 * @param index The specific index of the seed to be advanced
 * @return a float_t representing random number generated using the Box-Muller algorithm
 * @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
 */
float_t randn(int * seed, int index) {
	/*Box-Muller algorithm*/
	float_t u = randu(seed, index);
	float_t v = randu(seed, index);
	float_t cosine = cos(2 * PI * v);
	float_t rt = -2 * log(u);
	return sqrt(rt) * cosine;
}

float_t test_randn(int * seed, int index) {
	//Box-Muller algortihm
	float_t pi = 3.14159265358979323846;
	float_t u = randu(seed, index);
	float_t v = randu(seed, index);
	float_t cosine = cos(2 * pi * v);
	float_t rt = -2 * log(u);
	return sqrt(rt) * cosine;
}

/** 
 * Takes in a float_t and returns an integer that approximates to that float_t
 * @return if the mantissa < .5 => return value < input value; else return value > input value
 */
float_t roundDouble(float_t value) {
	int newValue = (int) (value);
	if (value - newValue < .5)
		return newValue;
	else
		return newValue++;
}

/**
 * Set values of the 3D array to a newValue if that value is equal to the testValue
 * @param testValue The value to be replaced
 * @param newValue The value to replace testValue with
 * @param array3D The image vector
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 */
void setIf(int testValue, int newValue, unsigned char * array3D, int * dimX,
		int * dimY, int * dimZ) {
	int x, y, z;
	for (x = 0; x < *dimX; x++) {
		for (y = 0; y < *dimY; y++) {
			for (z = 0; z < *dimZ; z++) {
				if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue)
					array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
			}
		}
	}
}

/**
 * Sets values of 3D matrix using randomly generated numbers from a normal distribution
 * @param array3D The video to be modified
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param seed The seed array
 */
void addNoise(unsigned char * array3D, int * dimX, int * dimY, int * dimZ,
		int * seed) {
	int x, y, z;
	for (x = 0; x < *dimX; x++) {
		for (y = 0; y < *dimY; y++) {
			for (z = 0; z < *dimZ; z++) {
				array3D[x * *dimY * *dimZ + y * *dimZ + z] = array3D[x * *dimY
						* *dimZ + y * *dimZ + z]
						+ (unsigned char) (5 * randn(seed, 0));
			}
		}
	}
}

/**
 * Fills a radius x radius matrix representing the disk
 * @param disk The pointer to the disk to be made
 * @param radius  The radius of the disk to be made
 */
void strelDisk(int * disk, int radius) {
	int diameter = radius * 2 - 1;
	int x, y;
	for (x = 0; x < diameter; x++) {
		for (y = 0; y < diameter; y++) {
			float_t distance = sqrt(
					pow((float_t) (x - radius + 1), 2)
							+ pow((float_t) (y - radius + 1), 2));
			if (distance < radius)
				disk[x * diameter + y] = 1;
		}
	}
}

/**
 * Dilates the provided video
 * @param matrix The video to be dilated
 * @param posX The x location of the pixel to be dilated
 * @param posY The y location of the pixel to be dilated
 * @param poxZ The z location of the pixel to be dilated
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param error The error radius
 */
void dilate_matrix(unsigned char * matrix, int posX, int posY, int posZ,
		int dimX, int dimY, int dimZ, int error) {
	int startX = posX - error;
	while (startX < 0)
		startX++;
	int startY = posY - error;
	while (startY < 0)
		startY++;
	int endX = posX + error;
	while (endX > dimX)
		endX--;
	int endY = posY + error;
	while (endY > dimY)
		endY--;
	int x, y;
	for (x = startX; x < endX; x++) {
		for (y = startY; y < endY; y++) {
			float_t distance = sqrt(
					pow((float_t) (x - posX), 2)
							+ pow((float_t) (y - posY), 2));
			if (distance < error)
				matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
		}
	}
}

/**
 * Dilates the target matrix using the radius as a guide
 * @param matrix The reference matrix
 * @param dimX The x dimension of the video
 * @param dimY The y dimension of the video
 * @param dimZ The z dimension of the video
 * @param error The error radius to be dilated
 * @param newMatrix The target matrix
 */
void imdilate_disk(unsigned char * matrix, int dimX, int dimY, int dimZ,
		int error, unsigned char * newMatrix) {
	int x, y, z;
	for (z = 0; z < dimZ; z++) {
		for (x = 0; x < dimX; x++) {
			for (y = 0; y < dimY; y++) {
				if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
					dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
				}
			}
		}
	}
}

/**
 * Fills a 2D array describing the offsets of the disk object
 * @param se The disk object
 * @param numOnes The number of ones in the disk
 * @param neighbors The array that will contain the offsets
 * @param radius The radius used for dilation
 */
void getneighbors(int * se, int numOnes, int * neighbors, int radius) {
	int x, y;
	int neighY = 0;
	int center = radius - 1;
	int diameter = radius * 2 - 1;
	for (x = 0; x < diameter; x++) {
		for (y = 0; y < diameter; y++) {
			if (se[x * diameter + y]) {
				neighbors[neighY * 2] = (int) (y - center);
				neighbors[neighY * 2 + 1] = (int) (x - center);
				neighY++;
			}
		}
	}
}

/**
 * The synthetic video sequence we will work with here is composed of a
 * single moving object, circular in shape (fixed radius)
 * The motion here is a linear motion
 * the foreground intensity and the background intensity is known
 * the image is corrupted with zero mean Gaussian noise
 * @param I The video itself
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames of the video
 * @param seed The seed array used for number generation
 */
void videoSequence(unsigned char * I, int IszX, int IszY, int Nfr, int * seed) {
	int k;
	int max_size = IszX * IszY * Nfr;
	/*get object centers*/
	int x0 = (int) roundDouble(IszY / 2.0);
	int y0 = (int) roundDouble(IszX / 2.0);
	I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

	/*move point*/
	int xk, yk, pos;
	for (k = 1; k < Nfr; k++) {
		xk = abs(x0 + (k - 1));
		yk = abs(y0 - 2 * (k - 1));
		pos = yk * IszY * Nfr + xk * Nfr + k;
		if (pos >= max_size)
			pos = 0;
		I[pos] = 1;
	}

	/*dilate matrix*/
	unsigned char * newMatrix = (unsigned char *) malloc(
			sizeof(unsigned char) * IszX * IszY * Nfr);
	imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
	int x, y;
	for (x = 0; x < IszX; x++) {
		for (y = 0; y < IszY; y++) {
			for (k = 0; k < Nfr; k++) {
				I[x * IszY * Nfr + y * Nfr + k] = newMatrix[x * IszY * Nfr
						+ y * Nfr + k];
			}
		}
	}
	free(newMatrix);

	/*define background, add noise*/
	setIf(0, 100, I, &IszX, &IszY, &Nfr);
	setIf(1, 228, I, &IszX, &IszY, &Nfr);
	/*add noise*/
	addNoise(I, &IszX, &IszY, &Nfr, seed);

}

/**
 * Finds the first element in the CDF that is greater than or equal to the provided value and returns that index
 * @note This function uses sequential search
 * @param CDF The CDF
 * @param lengthCDF The length of CDF
 * @param value The value to be found
 * @return The index of value in the CDF; if value is never found, returns the last index
 */
int findIndex(float_t * CDF, int lengthCDF, float_t value) {
	int index = -1;
	int x;
	for (x = 0; x < lengthCDF; x++) {
		if (CDF[x] >= value) {
			index = x;
			break;
		}
	}
	if (index == -1) {
		return lengthCDF - 1;
	}
	return index;
}

extern __global__ void likelihood_kernel(float_t * arrayX, float_t * arrayY,
		float_t * xj, float_t * yj, float_t * CDF, int * ind, int * objxy,
		float_t * likelihood, unsigned char * I, float_t * u, float_t * weights,
		int Nparticles, int countOnes, int max_size, int k, int IszY, int Nfr,
		int *seed, float_t* partial_sums);

extern __global__ void sum_kernel(float_t* partial_sums, int Nparticles);
extern __global__ void normalize_weights_kernel(float_t * weights,
		int Nparticles, float_t* partial_sums, float_t * CDF, float_t * u,
		int * seed);
extern __global__ void find_index_kernel(float_t * arrayX, float_t * arrayY,
		float_t * CDF, float_t * u, float_t * xj, float_t * yj,
		float_t * weights, int Nparticles);

/**
 * The implementation of the particle filter using OpenMP for many frames
 * @see http://openmp.org/wp/
 * @note This function is designed to work with a video of several frames. In addition, it references a provided MATLAB function which takes the video, the objxy matrix and the x and y arrays as arguments and returns the likelihoods
 * @param I The video to be run
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames
 * @param seed The seed array used for random number generation
 * @param Nparticles The number of particles to be used
 */
void particleFilter(unsigned char * I, int IszX, int IszY, int Nfr, int * seed,
		int Nparticles) {
	int max_size = IszX * IszY * Nfr;
	//original particle centroid
	float_t xe = roundDouble(IszY / 2.0);
	float_t ye = roundDouble(IszX / 2.0);

	//expected object locations, compared to center
	int radius = 5;
	int diameter = radius * 2 - 1;
	int * disk = (int*) malloc(diameter * diameter * sizeof(int));
	strelDisk(disk, radius);
	int countOnes = 0;
	int x, y;
	for (x = 0; x < diameter; x++) {
		for (y = 0; y < diameter; y++) {
			if (disk[x * diameter + y] == 1)
				countOnes++;
		}
	}
	int * objxy = (int *) malloc(countOnes * 2 * sizeof(int));
	getneighbors(disk, countOnes, objxy, radius);
	//initial weights are all equal (1/Nparticles)
	float_t * weights = (float_t *) malloc(sizeof(float_t) * Nparticles);
	for (x = 0; x < Nparticles; x++) {
		weights[x] = 1 / ((float_t) (Nparticles));
	}

	//initial likelihood to 0.0
	float_t * likelihood = (float_t *) malloc(sizeof(float_t) * Nparticles);
	float_t * arrayX = (float_t *) malloc(sizeof(float_t) * Nparticles);
	float_t * arrayY = (float_t *) malloc(sizeof(float_t) * Nparticles);
	float_t * xj = (float_t *) malloc(sizeof(float_t) * Nparticles);
	float_t * yj = (float_t *) malloc(sizeof(float_t) * Nparticles);
	float_t * CDF = (float_t *) malloc(sizeof(float_t) * Nparticles);

	//GPU copies of arrays
	float_t * arrayX_GPU;
	float_t * arrayY_GPU;
	float_t * xj_GPU;
	float_t * yj_GPU;
	float_t * CDF_GPU;
	float_t * likelihood_GPU;
	unsigned char * I_GPU;
	float_t * weights_GPU;
	int * objxy_GPU;

	int * ind = (int*) malloc(sizeof(int) * countOnes * Nparticles);
	int * ind_GPU;
	float_t * u = (float_t *) malloc(sizeof(float_t) * Nparticles);
	float_t * u_GPU;
	int * seed_GPU;
	float_t* partial_sums;

	//CUDA memory allocation
	check_error(
			cudaMalloc((void **) &arrayX_GPU, sizeof(float_t) * Nparticles));
	check_error(
			cudaMalloc((void **) &arrayY_GPU, sizeof(float_t) * Nparticles));
	check_error(cudaMalloc((void **) &xj_GPU, sizeof(float_t) * Nparticles));
	check_error(cudaMalloc((void **) &yj_GPU, sizeof(float_t) * Nparticles));
	check_error(cudaMalloc((void **) &CDF_GPU, sizeof(float_t) * Nparticles));
	check_error(cudaMalloc((void **) &u_GPU, sizeof(float_t) * Nparticles));
	check_error(
			cudaMalloc((void **) &likelihood_GPU,
					sizeof(float_t) * Nparticles));
	//set likelihood to zero
	check_error(
			cudaMemset((void *) likelihood_GPU, 0,
					sizeof(float_t) * Nparticles));
	check_error(
			cudaMalloc((void **) &weights_GPU, sizeof(float_t) * Nparticles));
	check_error(
			cudaMalloc((void **) &I_GPU,
					sizeof(unsigned char) * IszX * IszY * Nfr));
	check_error(cudaMalloc((void **) &objxy_GPU, sizeof(int) * 2 * countOnes));
	check_error(
			cudaMalloc((void **) &ind_GPU,
					sizeof(int) * countOnes * Nparticles));
	check_error(cudaMalloc((void **) &seed_GPU, sizeof(int) * Nparticles));
	check_error(
			cudaMalloc((void **) &partial_sums, sizeof(float_t) * Nparticles));

	//Donnie - this loop is different because in this kernel, arrayX and arrayY
	//  are set equal to xj before every iteration, so effectively, arrayX and
	//  arrayY will be set to xe and ye before the first iteration.
	for (x = 0; x < Nparticles; x++) {

		xj[x] = xe;
		yj[x] = ye;

	}

	int k;
	int indX, indY;
	//start send
	long long send_start = get_time();
	check_error(
			cudaMemcpy(I_GPU, I, sizeof(unsigned char) * IszX * IszY * Nfr,
					cudaMemcpyHostToDevice));
	check_error(
			cudaMemcpy(objxy_GPU, objxy, sizeof(int) * 2 * countOnes,
					cudaMemcpyHostToDevice));
	check_error(
			cudaMemcpy(weights_GPU, weights, sizeof(float_t) * Nparticles,
					cudaMemcpyHostToDevice));
	check_error(
			cudaMemcpy(xj_GPU, xj, sizeof(float_t) * Nparticles,
					cudaMemcpyHostToDevice));
	check_error(
			cudaMemcpy(yj_GPU, yj, sizeof(float_t) * Nparticles,
					cudaMemcpyHostToDevice));
	check_error(
			cudaMemcpy(seed_GPU, seed, sizeof(int) * Nparticles,
					cudaMemcpyHostToDevice));
	long long send_end = get_time();
	printf("TIME TO SEND TO GPU: %f\n", elapsed_time(send_start, send_end));
	int num_blocks = ceil((float_t) Nparticles / (float_t) threads_per_block);

	for (k = 1; k < Nfr; k++) {

		likelihood_kernel<<<num_blocks, threads_per_block>>>(arrayX_GPU,
				arrayY_GPU, xj_GPU, yj_GPU, CDF_GPU, ind_GPU, objxy_GPU,
				likelihood_GPU, I_GPU, u_GPU, weights_GPU, Nparticles,
				countOnes, max_size, k, IszY, Nfr, seed_GPU, partial_sums);

		sum_kernel<<<num_blocks, threads_per_block>>>(partial_sums, Nparticles);

		normalize_weights_kernel<<<num_blocks, threads_per_block>>>(weights_GPU,
				Nparticles, partial_sums, CDF_GPU, u_GPU, seed_GPU);

		find_index_kernel<<<num_blocks, threads_per_block>>>(arrayX_GPU,
				arrayY_GPU, CDF_GPU, u_GPU, xj_GPU, yj_GPU, weights_GPU,
				Nparticles);

	}    //end loop

	//block till kernels are finished
	cudaThreadSynchronize();
	long long back_time = get_time();

	cudaFree(xj_GPU);
	cudaFree(yj_GPU);
	cudaFree(CDF_GPU);
	cudaFree(u_GPU);
	cudaFree(likelihood_GPU);
	cudaFree(I_GPU);
	cudaFree(objxy_GPU);
	cudaFree(ind_GPU);
	cudaFree(seed_GPU);
	cudaFree(partial_sums);

	long long free_time = get_time();
	check_error(
			cudaMemcpy(arrayX, arrayX_GPU, sizeof(float_t) * Nparticles,
					cudaMemcpyDeviceToHost));
	long long arrayX_time = get_time();
	check_error(
			cudaMemcpy(arrayY, arrayY_GPU, sizeof(float_t) * Nparticles,
					cudaMemcpyDeviceToHost));
	long long arrayY_time = get_time();
	check_error(
			cudaMemcpy(weights, weights_GPU, sizeof(float_t) * Nparticles,
					cudaMemcpyDeviceToHost));
	long long back_end_time = get_time();
	printf("GPU Execution: %lf\n", elapsed_time(send_end, back_time));
	printf("FREE TIME: %lf\n", elapsed_time(back_time, free_time));
	printf("TIME TO SEND BACK: %lf\n", elapsed_time(back_time, back_end_time));
	printf("SEND ARRAY X BACK: %lf\n", elapsed_time(free_time, arrayX_time));
	printf("SEND ARRAY Y BACK: %lf\n", elapsed_time(arrayX_time, arrayY_time));
	printf("SEND WEIGHTS BACK: %lf\n",
			elapsed_time(arrayY_time, back_end_time));

	xe = 0;
	ye = 0;
	// estimate the object location by expected values
	for (x = 0; x < Nparticles; x++) {
		xe += arrayX[x] * weights[x];
		ye += arrayY[x] * weights[x];
	}
	printf("XE: %lf\n", xe);
	printf("YE: %lf\n", ye);
	float_t distance = sqrt(
			pow((float_t) (xe - (int) roundDouble(IszY / 2.0)), 2)
					+ pow((float_t) (ye - (int) roundDouble(IszX / 2.0)), 2));
	printf("%lf\n", distance);

	//CUDA freeing of memory
	cudaFree(weights_GPU);
	cudaFree(arrayY_GPU);
	cudaFree(arrayX_GPU);

	//free regular memory
	free(likelihood);
	free(arrayX);
	free(arrayY);
	free(xj);
	free(yj);
	free(CDF);
	free(ind);
	free(u);
}

