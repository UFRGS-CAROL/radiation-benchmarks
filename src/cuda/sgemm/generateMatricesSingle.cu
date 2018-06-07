#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <unistd.h>
#include <string>
#include <sys/time.h>
#include <float.h>

#include <random>
//#include <cublas.h>
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

// helper functions
#include "helper_string.h"
#include "helper_cuda.h"

#define DEFAULT_INPUT_SIZE 8192

#define MAX_SVALUE_NO_TENSOR ((float)sqrt(FLT_MAX / DEFAULT_INPUT_SIZE))
#define MAX_SVALUE_TENSOR 65503.0
int k = 0;
int size;
float *A, *B, *GOLD;

bool host_check = false;
bool generator_debug = false;

char *gold_matrix_path, *a_matrix_path, *b_matrix_path;

void usage() {
	printf(
			"Usage: generateMatricesSingle -size=N [-generator_debug] [-host_check] [-input_a=<path>] [-input_b=<path>] [-gold=<path>] [-use_tensors=<0 or 1>]\n");
}

void generateInputMatrices(unsigned char use_tersor_cores) {
	float *h_A, *h_B;
	FILE *f_A, *f_B;
	float MAX_SVALUE = MAX_SVALUE_NO_TENSOR;
	if (use_tersor_cores == 1) {
		MAX_SVALUE = MAX_SVALUE_TENSOR;
	}

	h_A = (float*) malloc(
			sizeof(float) * DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE);
	h_B = (float*) malloc(
			sizeof(float) * DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE);

	printf("Max value: %f Min: %f\n", MAX_SVALUE, -MAX_SVALUE);

	std::random_device rd; //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<float> dis(-MAX_SVALUE, MAX_SVALUE);
//	srand(time(NULL));

	if (!generator_debug) {
		for (int i = 0; i < DEFAULT_INPUT_SIZE; i++) {
			for (int j = 0; j < DEFAULT_INPUT_SIZE; j++) {
//				h_A[i * DEFAULT_INPUT_SIZE + j] = (rand()
//						/ ((float) (RAND_MAX) + 1) * (-4.06e16 - 4.4e16))
//						+ 4.1e16;
				h_A[i * DEFAULT_INPUT_SIZE + j] = dis(gen);

				h_B[i * DEFAULT_INPUT_SIZE + j] = dis(gen);
			}
		}
	} else {
		for (int i = 0; i < DEFAULT_INPUT_SIZE; i++) {
			for (int j = 0; j < DEFAULT_INPUT_SIZE; j++) {
				h_A[i * DEFAULT_INPUT_SIZE + j] = float(2.0);
				h_B[i * DEFAULT_INPUT_SIZE + j] = float(2.0);
			}
		}
	}

	int numZeros;
	int numNans;
	int numInfs;
// printf("Write\n");
	f_A = fopen(a_matrix_path, "wb");
	f_B = fopen(b_matrix_path, "wb");

	float val;

	numZeros = 0;
	numNans = 0;
	numInfs = 0;
	for (int i = 0; i < DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE; i++) {
		val = h_A[i];
		if (val == 0)
			numZeros++;
		if (isnan(val))
			numNans++;
		if (isinf(val))
			numInfs++;
	}
	printf("Number of zeros/NaNs/INFs on matrix A: %d/%d/%d\n", numZeros,
			numNans, numInfs);

	numZeros = 0;
	numNans = 0;
	numInfs = 0;
	for (int i = 0; i < DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE; i++) {
		val = h_B[i];
		if (val == 0)
			numZeros++;
		if (isnan(val))
			numNans++;
		if (isinf(val))
			numInfs++;
	}
	printf("Number of zeros/NaNs/INFs on matrix B: %d/%d/%d\n", numZeros,
			numNans, numInfs);

	for (int i = 0; i < DEFAULT_INPUT_SIZE; i++) {
		fwrite(&(h_A[i * DEFAULT_INPUT_SIZE]),
				sizeof(float) * DEFAULT_INPUT_SIZE, 1, f_A);
	}

	printf("Element 32 of matrix A: %f\n", (float) h_A[32]);

	printf("Element 50 of matrix B: %f\n", (float) h_B[50]);

	for (int i = 0; i < DEFAULT_INPUT_SIZE; i++) {
		fwrite(&(h_B[i * DEFAULT_INPUT_SIZE]),
				sizeof(float) * DEFAULT_INPUT_SIZE, 1, f_B);
	}
	printf("Done\n");

	fclose(f_A);
	fclose(f_B);

	free(h_A);
	free(h_B);

	return;
}

void ReadMatrixFromFile() {

	int i;
	FILE *f_A, *f_B;

	f_A = fopen(a_matrix_path, "rb");
	f_B = fopen(b_matrix_path, "rb");
	if (!(f_A && f_B)) {
		printf("Error opening matrices A, B.\n");
		printf("exit on line: %d", __LINE__);
		exit(-1);
	}
	size_t ret_value[2];
	for (i = 0; i < k; i++) {
		ret_value[0] = fread(&A[k * i], sizeof(float) * k, 1, f_A);
		ret_value[1] = fread(&B[k * i], sizeof(float) * k, 1, f_B);
		if (ret_value[0] != 1 || ret_value[1] != 1) {
			printf("Bad input/gold formatting: %lu ; %lu .\n", ret_value[0],
					ret_value[1]);
		}
	}
	printf("Done reading matrices\n");

	fclose(f_A);
	fclose(f_B);
}

void GetDevice() {

	cudaDeviceProp prop;
	cudaError_t teste;
	int count = 0;
	teste = cudaGetDeviceCount(&count);
	printf("Get Device Test: %s\n", cudaGetErrorString(teste));
	for (int i = 0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf("Name: %s\n", prop.name);
	}
	int *ndevice;
	int dev = 0;
	ndevice = &dev;
	cudaGetDevice(ndevice);

	cudaSetDevice(0);
	cudaGetDeviceProperties(&prop, 0);
	printf("\ndevice: %d %s\n", *ndevice, prop.name);

}

double mysecond() {
	struct timeval tp;
	struct timezone tzp;
	int i = gettimeofday(&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

float* openmpMul(float* a, float* b, size_t size) {
	double time = mysecond();

	float* bT = (float*) malloc(sizeof(float) * size * size);
	float* c = (float*) calloc(size * size, sizeof(float));

	if (c == NULL || bT == NULL) {
		printf("could not alloc hostGold matrix.");
		return NULL;
	}

#pragma omp parallel for
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			bT[j * size + i] = b[i * size + j];

#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				c[i * size + j] += a[j * size + k] * bT[i * size + k];
			}
		}
	}

	printf("host mmul time: %.2f seconds\n", mysecond() - time);

	return c;
}

void generateGoldMatrixHalf(unsigned char use_tensor_cores) {
	////////////////////////////////////////////////////
	/////////////CUBLAS GEMM VARS///////////////////////
	const float alpha = 1.0;
	const float beta = 1.0;
	cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_T;
	////////////////////////////////////////////////////
	// Alloc blas handle
	cublasHandle_t blas_handle;

	checkCudaErrors(cublasCreate(&blas_handle));

	printf("Tensor cores %d, is handle defined? %d\n", use_tensor_cores,
			(blas_handle && true));
	if (use_tensor_cores == 0) {
		cublasSetMathMode(blas_handle, CUBLAS_DEFAULT_MATH);
	} else if (use_tensor_cores == 1) {
		cublasSetMathMode(blas_handle, CUBLAS_TENSOR_OP_MATH);
	}

	////////////////////////////////////////////////////
	//////////DEVICE VARS///////////////////////////////
	float *d_A;
	float *d_B;
	float *d_C;
	////////////////////////////////////////////////////

	A = (float*) malloc(size * sizeof(float));
	B = (float*) malloc(size * sizeof(float));
	GOLD = (float*) malloc(size * sizeof(float));

	ReadMatrixFromFile();
	if (k <= 16) {
		printf("\nMatrix A: \n");
		for (int i = 0; i < k * k; i++) {
			printf(" %.2e", (float) A[i]);
			if ((i + 1) % k == 0)
				printf("\n");
		}
		printf("\nMatrix B: \n");
		for (int i = 0; i < k * k; i++) {
			printf(" %.2e", (float) B[i]);
			if ((i + 1) % k == 0)
				printf("\n");
		}
	}

	checkCudaErrors(cudaMalloc((void** ) &d_A, size * sizeof(float)));

	checkCudaErrors(cudaMalloc((void** ) &d_B, size * sizeof(float)));

	checkCudaErrors(cudaMalloc((void** ) &d_C, size * sizeof(float)));

	checkCudaErrors(cudaMemset(d_C, 0, size * sizeof(float))); // ZERA C

	checkCudaErrors(
			cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice)); // PUSH A

	checkCudaErrors(
			cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice)); // PUSH B

	printf("cudaSGEMM... k=%d\n", k);
	double time = mysecond();

	cublasSgemm(blas_handle, transa, transb, k, k, k, &alpha, d_A, k, d_B, k,
			&beta, d_C, k);

	checkCudaErrors(cudaPeekAtLastError());

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaPeekAtLastError());

	time = mysecond() - time;

	/////////// PERF
	double flops = 2.0 * (double) k * k * k;
	double gflops = flops / time;
	double outputpersec = (double) k * k / time;
	printf("kernel time: %lf\n", time);
	printf("SIZE:%d OUTPUT/S:%f FLOPS:%f (GFLOPS:%.2f)\n", k, outputpersec,
			gflops, gflops / 1000000000);
	///////////

	checkCudaErrors(
			cudaMemcpy(GOLD, d_C, size * sizeof(float),
					cudaMemcpyDeviceToHost));

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	printf("Analysing output on host...\n");

	int i, j;
	FILE *f_GOLD;

	f_GOLD = fopen(gold_matrix_path, "wb");

	float val;

	int numZeros = 0;
	int numNans = 0;
	int numInfs = 0;
	float maxAbsVal = 0.0;
#pragma omp parallel for
	for (int i = 0; i < k * k; i++) {
		val = GOLD[i];
		if (fabs(val) > maxAbsVal) {
#pragma omp critical
			maxAbsVal = max(fabs(val), maxAbsVal);
		}
		if (val == 0) {
#pragma omp atomic
			numZeros++;
			if (numZeros < 5)
				printf("Zero in position (%d,%d)\n", (int) floor(i / k),
						(int) (i - floor(i / k) * k));
		}
		if (isnan(val)) {
#pragma omp atomic
			numNans++;
			if (numNans < 5)
				printf("NaN in position (%d,%d)\n", (int) floor(i / k),
						(int) (i - floor(i / k) * k));
		}
		if (isinf(val)) {
#pragma omp atomic
			numInfs++;
			if (numInfs < 5)
				printf("INF in position (%d,%d)\n", (int) floor(i / k),
						(int) (i - floor(i / k) * k));
		}
	}
	printf("Number of zeros/NaNs/INFs on gold: %d/%d/%d\n", numZeros, numNans,
			numInfs);
	printf("Maximum absolute value on gold: %f\n", maxAbsVal);

	if (k <= 16) {
		for (int i = 0; i < k * k; i++) {
			printf(" %.2e", (float) GOLD[i]);
			if ((i + 1) % k == 0)
				printf("\n");
		}
	}

	if (host_check) {
		printf("Calculating mMul using OpenMP on Host...\n");
		float *hostGold = openmpMul(A, B, k);
		if (k <= 16) {
			printf("Host CPU Gold:\n");
			for (int i = 0; i < k * k; i++) {
				printf(" %.2e", (float) hostGold[i]);
				if ((i + 1) % k == 0)
					printf("\n");
			}
		}
		printf("Comparing GPU result with Host result...\n");
		float maxDiff = 0.0;
		float maxAbsDiff = 0.0;
		for (i = 0; i < k; i++) {
			for (j = 0; j < k; j++) {
				register float diff = fabs(
						(hostGold[i * k + j] - GOLD[i * k + j])
								/ hostGold[i * k + j]);
				register float absDiff = hostGold[i * k + j] - GOLD[i * k + j];
				if (diff > maxDiff) {
					maxDiff = max(diff, maxDiff);
					printf(
							"New diff! (%d,%d) hostGold!=gpuGold %e != %e (diff: %e)\n",
							i, j, hostGold[i * k + j], GOLD[i * k + j], diff);
				}
				if (absDiff > maxAbsDiff) {
					maxAbsDiff = max(absDiff, maxAbsDiff);
				}
				// if (diff > 0.1) {
				// 	printf("Fail! (%d,%d) hostGold!=gpuGold %f != %f (diff: %e)\n", i, j, (float)hostGold[i*k+j], (float)GOLD[i*k+j], diff);
				// 	fflush(stdout);
				// 	exit(-1);
				// }
			}
		}
		printf(
				"CPU and GPU match by a relative error of up to %e element difference.\nMaximum element absolute difference: %e (relatively to float representation: %e)\nWriting to file...\n",
				maxDiff, maxAbsDiff, maxAbsDiff / FLT_MAX);
	}

	//printf("-------------------------\n%.10f\n%.10f\n%.10f\n", GOLD[0], GOLD[1], GOLD[2]);

	for (i = 0; i < k; i++) {
		fwrite(&(GOLD[i * k]), sizeof(float) * k, 1, f_GOLD);
	}

	fclose(f_GOLD);
	cublasDestroy(blas_handle);

	return;
}

int main(int argc, char** argv) {
//====================================
//================== Read parameters
	unsigned char use_tensor_cores = 0;

	if (argc < 2) {
		usage();
		exit(-1);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "size")) {
		k = getCmdLineArgumentInt(argc, (const char **) argv, "size");

		if ((k <= 0) || (k % 16 != 0)) {
			printf("Invalid input size given on the command-line: %d\n", k);
			printf("exit on line: %d", __LINE__);
			exit(EXIT_FAILURE);
		}
	} else {
		usage();
		printf("exit on line: %d", __LINE__);
		exit(EXIT_FAILURE);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "input_a")) {
		getCmdLineArgumentString(argc, (const char **) argv, "input_a",
				&a_matrix_path);
	} else {
		a_matrix_path = new char[100];
		snprintf(a_matrix_path, 100, "sgemm_a_%i.matrix",
				(signed int) DEFAULT_INPUT_SIZE);
		printf("Using default input_a path: %s\n", a_matrix_path);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "input_b")) {
		getCmdLineArgumentString(argc, (const char **) argv, "input_b",
				&b_matrix_path);
	} else {
		b_matrix_path = new char[100];
		snprintf(b_matrix_path, 100, "sgemm_b_%i.matrix",
				(signed int) DEFAULT_INPUT_SIZE);
		printf("Using default input_a path: %s\n", b_matrix_path);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "gold")) {
		getCmdLineArgumentString(argc, (const char **) argv, "gold",
				&gold_matrix_path);
	} else {
		gold_matrix_path = new char[100];
		snprintf(gold_matrix_path, 100, "sgemm_gold_%i.matrix", (signed int) k);
		printf("Using default gold path: %s\n", gold_matrix_path);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "host_check")) {
		host_check = true;
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "generator_debug")) {
		generator_debug = true;
	}

	//flag for tensor cores
	if (checkCmdLineFlag(argc, (const char **) argv, "use_tensors")) {
		use_tensor_cores = getCmdLineArgumentInt(argc, (const char **) argv,
				"use_tensors");
	}
//====================================

	GetDevice();

	size = k * k;

	printf("Each input matrix size: %.4fGB\n",
			(float) sizeof(float) * DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE
					/ (1024 * 1024 * 1024));

	FILE *test_file;
	test_file = fopen(a_matrix_path, "rb");
	if (!test_file) {
		printf("Generating input matrices...\n");
		generateInputMatrices(use_tensor_cores);
	} else {
		printf("Input matrices already exist...\n");
	}

	generateGoldMatrixHalf(use_tensor_cores);

	return 0;
}
