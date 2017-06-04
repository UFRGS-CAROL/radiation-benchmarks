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

// helper functions
#include "helper_string.h"
#include "helper_cuda.h"

#define DEFAULT_INPUT_SIZE 8192
#define MAX_HALF 65504

#define ELEMENTS_PER_THREAD 128

int k=0;
int lda, ldb, ldc;
int sizea, sizeb, sizec;	
__half *A, *B, *GOLD;

char *gold_matrix_path, *a_matrix_path, *b_matrix_path;

void usage() {
    printf("Usage: generateMatrices -size=N [-input_a=<path>] [-input_b=<path>] [-gold=<path>]\n");
}

__global__ void floatToHalfKernel(__half *out, float in) {
	*out = __float2half(in);
}

__global__ void halfToFloatKernel(__half in, float *out) {
	*out = __half2float(in);
}

__half float2half(float in) {
	__half half;
	__half *d_half;

	cudaMalloc(&d_half, sizeof(__half));
	floatToHalfKernel<<<1, 1>>>(d_half, in);

	cudaDeviceSynchronize();

	cudaMemcpy(&half, d_half, sizeof(__half), cudaMemcpyDeviceToHost);

	return half;
}

float half2float(__half in) {
	float h_float;
	float *d_float;

	cudaMalloc(&d_float, sizeof(float));
	halfToFloatKernel<<<1, 1>>>(in, d_float);

	cudaDeviceSynchronize();

	cudaMemcpy(&h_float, d_float, sizeof(float), cudaMemcpyDeviceToHost);

	return h_float;
}

__global__ void generateKernel(__half *out, unsigned int matrixSize, unsigned int seed) {
	curandState_t state;
	curand_init(seed, threadIdx.x, 0, &state);

	for (register int i=0; i<ELEMENTS_PER_THREAD; i++) {
		float temp = curand_normal(&state) * MAX_HALF;
		out[ELEMENTS_PER_THREAD*threadIdx.x + i] = __float2half(temp);
	}
}

void generateInputMatrices()
{
	__half *h_A, *h_B;
	__half *dev_A, *dev_B;
	FILE *f_A, *f_B;

	h_A = (__half*)malloc(DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE * sizeof(__half));
	h_B = (__half*)malloc(DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE * sizeof(__half));

	if (!h_B || !h_A) {
		printf("Cant alloc host memory for input generation.\n");
		printf("exit on line: %d", __LINE__); exit(-1);
	}

//================== Set block and grid size for GoldChk kernel
//====================================

	/* CUDA's random number library uses curandState_t to keep track of the seed value
		we will store a random state for every thread  */

	// printf("Alloc\n");
	// curandState_t* state;

	/* allocate space on the GPU for the random states */

	// printf("Size: %ldMB", blocksize * blocksize * sizeof(curandState_t) / (1024*1024));

	// checkCudaErrors( cudaMalloc((void**) &state, sizeof(curandState_t)) );
	checkCudaErrors( cudaMalloc((void**) &dev_A, DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE * sizeof(__half)) );
	checkCudaErrors( cudaMemset(dev_A, 0, DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE * sizeof(__half)) );

// printf("InitRand\n");

	// curandInitKernel<<<1, 1>>>(state, time(NULL));
	// checkCudaErrors( cudaDeviceSynchronize() );

// printf("Generate\n");
	generateKernel<<<1, DEFAULT_INPUT_SIZE/ELEMENTS_PER_THREAD>>>(dev_A, DEFAULT_INPUT_SIZE, time(NULL));
	checkCudaErrors( cudaDeviceSynchronize() );

// printf("Copy\n");
	checkCudaErrors( cudaMemcpy(h_A, dev_A, DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE * sizeof(__half), cudaMemcpyDeviceToHost) );

	cudaFree(dev_A);
	// cudaFree(state);

// printf("Alloc\n");
	// checkCudaErrors( cudaMalloc((void**) &state, sizeof(curandState_t)) );
	checkCudaErrors( cudaMalloc((void**) &dev_B, DEFAULT_INPUT_SIZE  * DEFAULT_INPUT_SIZE* sizeof(__half)) );
	checkCudaErrors( cudaMemset(dev_B, 0, DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE * sizeof(__half)) );

	// printf("Init Rand\n");

	// curandInitKernel<<<1, 1>>>(state, time(NULL));
	// cudaDeviceSynchronize();

// printf("Generate\n");
	generateKernel<<<1, DEFAULT_INPUT_SIZE/ELEMENTS_PER_THREAD>>>(dev_B, DEFAULT_INPUT_SIZE, time(NULL));
	cudaDeviceSynchronize();

// printf("Copy\n");
	checkCudaErrors( cudaMemcpy(h_B, dev_B, DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE * sizeof(__half), cudaMemcpyDeviceToHost) );

	cudaFree(dev_B);
	// cudaFree(state);


// printf("Write\n");
	f_A = fopen(a_matrix_path, "wb");
	f_B = fopen(b_matrix_path, "wb");

	for(int i=0; i<DEFAULT_INPUT_SIZE; i++)
	{
		fwrite(&(h_A[i * DEFAULT_INPUT_SIZE]), sizeof(__half) * DEFAULT_INPUT_SIZE, 1, f_A);
	}

	printf("Element 32 of matrix A: %f (raw __half: %hx)\n", half2float(h_A[32]), h_A[32].x);

	printf("Element 50 of matrix B: %f (raw __half: %hx)\n", half2float(h_B[50]), h_B[50].x);


	for(int i=0; i<DEFAULT_INPUT_SIZE; i++)
	{
		fwrite(&(h_B[i * DEFAULT_INPUT_SIZE]), sizeof(__half) * DEFAULT_INPUT_SIZE, 1, f_B);
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
	for(i=0; i<k; i++)
	{
		fread (&A[ lda * i ], sizeof(__half)*k, 1, f_A);
		fread (&B[ lda * i ], sizeof(__half)*k, 1, f_B);
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
	const __half alpha = float2half(1.0f);
	const __half beta = float2half(1.0f);
	cublasOperation_t transa = CUBLAS_OP_T;
	cublasOperation_t transb = CUBLAS_OP_T;
	////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////
	//////////DEVICE VARS///////////////////////////////	

	__half *d_A;
	__half *d_B;
	__half *d_C;
	////////////////////////////////////////////////////

	A = ( __half* ) malloc( sizea * sizeof( __half ) );
	B = ( __half* ) malloc( sizeb * sizeof( __half ) );
	GOLD = ( __half* ) malloc( sizec * sizeof( __half ) );
	
	ReadMatrixFromFile();

	checkCudaErrors( cudaMalloc( ( void** ) &d_A, sizea * sizeof( __half ) ));

	checkCudaErrors( cudaMalloc( ( void** ) &d_B, sizea * sizeof( __half ) ));

	checkCudaErrors( cudaMalloc( ( void** ) &d_C, sizea * sizeof( __half ) ));


	checkCudaErrors( cudaMemset( d_C, 0, sizeb * sizeof( __half )) ); // ZERA C

	checkCudaErrors( cudaMemcpy( d_A, A, sizeb * sizeof( __half ), cudaMemcpyHostToDevice ) ); // PUSH A

	checkCudaErrors( cudaMemcpy( d_B, B, sizeb * sizeof( __half ), cudaMemcpyHostToDevice ) ); // PUSH B

	printf("cublasHgemm... k=%d transa=%c transb=%c lda=%d ldb=%d ldc=%d\n", k, transa, transb, lda, ldb, ldc);
	double time = mysecond();

	cublasHandle_t cublasHandle;
	checkCudaErrors( cublasCreate(&cublasHandle) );

	checkCudaErrors( cublasHgemm(cublasHandle, transa, transb,
			   k, k, k,
			   &alpha,
			   d_A, lda,
			   d_B, ldb,
			   &beta,
			   d_C, ldc ) );
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

	checkCudaErrors( cudaMemcpy(GOLD, d_C, sizec * sizeof( __half ), cudaMemcpyDeviceToHost) );

	cudaFree( d_A );
	cudaFree( d_B );
	cudaFree( d_C );

	int i;
	FILE *f_GOLD;

	f_GOLD = fopen(gold_matrix_path, "wb");

	int numZeros = 0;
	for (int i = 0; i<k*k; i++) {
		if (GOLD[i].x == 0) {
			numZeros++;
		}
	}
	printf("Number of zeros: %d\n", numZeros);

	//printf("-------------------------\n%.10f\n%.10f\n%.10f\n", GOLD[0], GOLD[1], GOLD[2]);

	for(i=0; i<k; i++)
	{
		fwrite( &GOLD[i * lda], sizeof(__half)*k, 1, f_GOLD );
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

	lda = max( 1, k + 16 );
	sizea = lda * k;
	ldb = max( 1, k + 16 );
	sizeb = ldb * k;
	ldc = max( 1, k + 16 );
	sizec = ldc * k;
	
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
