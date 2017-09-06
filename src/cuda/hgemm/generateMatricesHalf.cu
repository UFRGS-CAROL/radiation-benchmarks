#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <unistd.h>
#include <string>
#include <sys/time.h>

#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include "half.hpp"

// helper functions
#include "helper_string.h"
#include "helper_cuda.h"

#define DEFAULT_INPUT_SIZE 8192
#define MAX_HALF 65503
#define MAX_HVALUE (float)(sqrt(MAX_HALF / DEFAULT_INPUT_SIZE))

#define BLOCK_SIZE 32

int k=0;
int sizea, sizeb, sizec;
half *A, *B, *GOLD;

char *gold_matrix_path, *a_matrix_path, *b_matrix_path;

void usage() {
    printf("Usage: generateMatrices -size=N [-input_a=<path>] [-input_b=<path>] [-gold=<path>]\n");
}

void generateInputMatrices()
{
	half *h_A, *h_B;
	FILE *f_A, *f_B;

    h_A = (half*)malloc(sizeof(half) * DEFAULT_INPUT_SIZE*DEFAULT_INPUT_SIZE);
    h_B = (half*)malloc(sizeof(half) * DEFAULT_INPUT_SIZE*DEFAULT_INPUT_SIZE);
    printf("Each input matrix size: %.4fGB\n", (float)sizeof(half) * DEFAULT_INPUT_SIZE*DEFAULT_INPUT_SIZE / (1024*1024*1024));
    printf("Max value: %f Min: %f\n", MAX_HVALUE, -MAX_HVALUE);

	srand(time(NULL));

    half_float::half tempValue;

    for (int i=0; i<DEFAULT_INPUT_SIZE; i++) {
        for (int j=0; j<DEFAULT_INPUT_SIZE; j++) {
            do {
                tempValue = half_float::half((((float)rand() / RAND_MAX)) * (MAX_HVALUE * 2.0) - MAX_HVALUE);
            } while (isnan((float)tempValue) || isinf((float)tempValue) || (float)tempValue==0.0);
            h_A[i * DEFAULT_INPUT_SIZE + j] = *((half*)&tempValue);

            do {
                tempValue = half_float::half((((float)rand() / RAND_MAX)) * (MAX_HVALUE * 2.0) - MAX_HVALUE);
            } while (isnan((float)tempValue) || isinf((float)tempValue) || (float)tempValue==0.0);
            h_B[i * DEFAULT_INPUT_SIZE + j] = *((half*)&tempValue);
        }
    }

	int numZeros;
    int numNans;
    int numInfs;
// printf("Write\n");
	f_A = fopen(a_matrix_path, "wb");
	f_B = fopen(b_matrix_path, "wb");

    half_float::half val;

	numZeros = 0;
    numNans = 0;
    numInfs = 0;
	for (int i = 0; i<DEFAULT_INPUT_SIZE*DEFAULT_INPUT_SIZE; i++) {
        val=(float)*((half_float::half*)&(h_A[i]));
		if (val == 0) numZeros++;
        if (isnan(val)) numNans++;
        if (isinf(val)) numInfs++;
	}
	printf("Number of zeros/NaNs/INFs on matrix A: %d/%d/%d\n", numZeros, numNans, numInfs);

	numZeros = 0;
    numNans = 0;
    numInfs = 0;
	for (int i = 0; i<DEFAULT_INPUT_SIZE*DEFAULT_INPUT_SIZE; i++) {
        val=(float)*((half_float::half*)&(h_B[i]));
		if (val == 0) numZeros++;
        if (isnan(val)) numNans++;
        if (isinf(val)) numInfs++;
	}
	printf("Number of zeros/NaNs/INFs on matrix B: %d/%d/%d\n", numZeros, numNans, numInfs);

	for(int i=0; i<DEFAULT_INPUT_SIZE; i++)
	{
		fwrite(&(h_A[i * DEFAULT_INPUT_SIZE]), sizeof(half) * DEFAULT_INPUT_SIZE, 1, f_A);
	}

	printf("Element 32 of matrix A: %f (raw half: %hx)\n", (float)*((half_float::half*)&(h_A[32])), h_A[32].x);

	printf("Element 50 of matrix B: %f (raw half: %hx)\n", (float)*((half_float::half*)&(h_B[50])), h_B[50].x);


	for(int i=0; i<DEFAULT_INPUT_SIZE; i++)
	{
		fwrite(&(h_B[i * DEFAULT_INPUT_SIZE]), sizeof(half) * DEFAULT_INPUT_SIZE, 1, f_B);
	}
	printf("Done\n");

	fclose(f_A);
	fclose(f_B);

	free(h_A);
	free(h_B);

	return;
}

void ReadMatrixFromFile(){

	int i;
	FILE *f_A, *f_B;

	f_A = fopen(a_matrix_path,"rb");
	f_B = fopen(b_matrix_path,"rb");
	if (!(f_A&&f_B))
	{
		printf("Error opening matrices A, B.\n");
		printf("exit on line: %d", __LINE__); exit(-1);
	}
    size_t ret_value[2];
	for(i=0; i<k; i++)
	{
		ret_value[0] = fread (&A[ k * i ], sizeof(half)*k, 1, f_A);
		ret_value[1] = fread (&B[ k * i ], sizeof(half)*k, 1, f_B);
        if (ret_value[0] != 1 || ret_value[1] != 1) {
            printf("Bad input/gold formatting: %lu ; %lu .\n", ret_value[0], ret_value[1]);
        }
	}
printf("Done reading matrices\n");

	fclose(f_A);
	fclose(f_B);
}

void GetDevice(){

    cudaDeviceProp prop;
    cudaError_t teste;
    int count=0;
    teste = cudaGetDeviceCount(&count);
	printf("Get Device Test: %s\n", cudaGetErrorString(teste));
    for (int i=0; i< count; i++) {
        cudaGetDeviceProperties( &prop, i );
        printf( "Name: %s\n", prop.name );
    }
    int *ndevice; int dev = 0;
    ndevice = &dev;
    cudaGetDevice(ndevice);

    cudaSetDevice(0);
        cudaGetDeviceProperties( &prop, 0 );
	printf("\ndevice: %d %s\n", *ndevice, prop.name);

}

double mysecond()
{
   struct timeval tp;
   struct timezone tzp;
   int i = gettimeofday(&tp,&tzp);
   return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void generateGoldMatrix()
{
	////////////////////////////////////////////////////
	/////////////CUBLAS GEMM VARS///////////////////////
    half_float::half oneValue(1.0);
	const half alpha = *((half*)&oneValue);
	const half beta = *((half*)&oneValue);
	cublasOperation_t transa = CUBLAS_OP_T;
	cublasOperation_t transb = CUBLAS_OP_T;
	////////////////////////////////////////////////////

	////////////////////////////////////////////////////
	//////////DEVICE VARS///////////////////////////////

	half *d_A;
	half *d_B;
	half *d_C;
	////////////////////////////////////////////////////

	A = ( half* ) malloc( sizea * sizeof( half ) );
	B = ( half* ) malloc( sizeb * sizeof( half ) );
	GOLD = ( half* ) malloc( sizec * sizeof( half ) );

	ReadMatrixFromFile();
  if (k <= 16) {
    printf("\nMatrix A: \n");
    for (int i = 0; i<k*k; i++) {
      printf(" %.2e", (float)*((half_float::half*)&(A[i])));
      if ((i+1)%k == 0) printf("\n");
    }
    printf("\nMatrix B: \n");
    for (int i = 0; i<k*k; i++) {
      printf(" %.2e", (float)*((half_float::half*)&(B[i])));
      if ((i+1)%k == 0) printf("\n");
    }
  }

	checkCudaErrors( cudaMalloc( ( void** ) &d_A, sizea * sizeof( half ) ));

	checkCudaErrors( cudaMalloc( ( void** ) &d_B, sizeb * sizeof( half ) ));

	checkCudaErrors( cudaMalloc( ( void** ) &d_C, sizec * sizeof( half ) ));


	checkCudaErrors( cudaMemset( d_C, 0, sizec * sizeof( half )) ); // ZERA C

	checkCudaErrors( cudaMemcpy( d_A, A, sizea * sizeof( half ), cudaMemcpyHostToDevice ) ); // PUSH A

	checkCudaErrors( cudaMemcpy( d_B, B, sizeb * sizeof( half ), cudaMemcpyHostToDevice ) ); // PUSH B

	printf("cublasHgemm... k=%d transa=%hx transb=%hx\n", k, transa, transb);
	double time = mysecond();

	cublasHandle_t cublasHandle;
	checkCudaErrors( cublasCreate(&cublasHandle) );

	checkCudaErrors( cublasHgemm(cublasHandle, transa, transb,
			   k, k, k,
			   &alpha,
			   d_A, k,
			   d_B, k,
			   &beta,
			   d_C, k ) );
	checkCudaErrors( cudaDeviceSynchronize() );

	cublasDestroy(cublasHandle);

	time=mysecond()-time;

	/////////// PERF
    double flops = 2.0*(double)k*k*k;
    double gflops = flops / time;
    double outputpersec = (double)k*k/time;
    printf("kernel time: %lf\n",time);
    printf("SIZE:%d OUTPUT/S:%f FLOPS:%f (GFLOPS:%.2f)\n",k, outputpersec, gflops, gflops/1000000000);
	///////////

	checkCudaErrors( cudaMemcpy(GOLD, d_C, sizec * sizeof( half ), cudaMemcpyDeviceToHost) );

	cudaFree( d_A );
	cudaFree( d_B );
	cudaFree( d_C );

	int i;
	FILE *f_GOLD;

	f_GOLD = fopen(gold_matrix_path, "wb");

    half_float::half val;

	int numZeros = 0;
    int numNans = 0;
    int numInfs = 0;
	for (int i = 0; i<k*k; i++) {
        val=(float)*((half_float::half*)&(GOLD[i]));
		if (val == 0) numZeros++;
        if (isnan(val)) numNans++;
        if (isinf(val)) numInfs++;
	}
	printf("Number of zeros/NaNs/INFs on gold: %d/%d/%d\n", numZeros, numNans, numInfs);

  if (numZeros > 0 && k <= 16) {
    for (int i = 0; i<k*k; i++) {
      printf(" %.2e",  (float)*((half_float::half*)&(GOLD[i])));
      if ((i+1)%k == 0) printf("\n");
    }
  }

	//printf("-------------------------\n%.10f\n%.10f\n%.10f\n", GOLD[0], GOLD[1], GOLD[2]);

	for(i=0; i<k; i++)
	{
		fwrite( &GOLD[i * k], sizeof(half)*k, 1, f_GOLD );
	}

	fclose(f_GOLD);

	return;
}

int main (int argc, char** argv)
{
//====================================
//================== Read parameters
	if (argc<2) {
		usage();
		exit (-1);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "size"))
    {
        k = getCmdLineArgumentInt(argc, (const char **)argv, "size");

        if ((k <= 0)||(k % 16 != 0))
        {
            printf("Invalid input size given on the command-line: %d\n", k);
            printf("exit on line: %d", __LINE__); exit(EXIT_FAILURE);
        }
    }
	else
	{
		usage();
		printf("exit on line: %d", __LINE__); exit(EXIT_FAILURE);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "input_a"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "input_a", &a_matrix_path);
    }
    else
    {
        a_matrix_path = new char[100];
        snprintf(a_matrix_path, 100, "hgemm_a_%i", (signed int)DEFAULT_INPUT_SIZE);
        printf("Using default input_a path: %s\n", a_matrix_path);
    }

	if (checkCmdLineFlag(argc, (const char **)argv, "input_b"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "input_b", &b_matrix_path);
    }
    else
    {
        b_matrix_path = new char[100];
        snprintf(b_matrix_path, 100, "hgemm_b_%i", (signed int)DEFAULT_INPUT_SIZE);
        printf("Using default input_a path: %s\n", b_matrix_path);
    }

	if (checkCmdLineFlag(argc, (const char **)argv, "gold"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "gold", &gold_matrix_path);
    }
    else
    {
        gold_matrix_path = new char[100];
        snprintf(gold_matrix_path, 100, "hgemm_gold_%i", (signed int)k);
        printf("Using default gold path: %s\n", gold_matrix_path);
    }
//====================================

	GetDevice();

	sizea = k * k;
	sizeb = k * k;
	sizec = k * k;


	FILE *test_file;
	test_file=fopen(a_matrix_path, "rb");
	if (!test_file)
	{
		printf("Generating input matrices...\n");
		generateInputMatrices();
	}
	else
	{	printf("Input matrices already exist...\n");	}

	generateGoldMatrix();

	return 0;
}
