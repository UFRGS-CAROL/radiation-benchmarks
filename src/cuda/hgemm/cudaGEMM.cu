#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <string>
#include <omp.h>

#ifdef LOGS
#include "log_helper.h"
#endif
// The timestamp is updated on every log_helper function call.

#include <cublas_v2.h>

#include <cuda_fp16.h>

// helper functions
#include "helper_string.h"
#include "helper_cuda.h"

#undef min
#define min( x, y ) ( (x) < (y) ? (x) : (y) )
#undef max
#define max( x, y ) ( (x) > (y) ? (x) : (y) )

#define BLOCK_SIZE 32

#define DEFAULT_INPUT_SIZE 8192

__half int2hfloat(int x);

int verbose = 0;
int fault_injection = 0;

int k=0; // k x k matrix size
int iterations=100000000; // global loop iteracion

//================== Input paths
char *gold_matrix_path, *a_matrix_path, *b_matrix_path;

FILE* f_A;
FILE* f_B;
FILE* f_GOLD;
//====================================

//================== Host and device matrix ptr's
__half *A;
__half *B;
__half *C;
__half *GOLD;

__half *d_A;
__half *d_B;
__half *d_C;
//====================================

//================== cublas GEMM parameters
const __half alpha = int2hfloat(1);
const __half beta = int2hfloat(1);
cublasOperation_t transa = CUBLAS_OP_T;
cublasOperation_t transb = CUBLAS_OP_T;
int sizea, sizeb, sizec;
int lda, ldb, ldc;

//========= FP16 Host conversion
__half int2hfloat(int x)
{
  unsigned sign = x < 0;
  unsigned absx = ((unsigned)x ^ -sign) + sign; // safe abs(x)
  unsigned tmp = absx, manbits = 0;
  int exp = 0, truncated = 0;

  // calculate the number of bits needed for the mantissa
  while (tmp)
  {
    tmp >>= 1;
    manbits++;
  }

  // half-precision floats have 11 bits in the mantissa.
  // truncate the excess or insert the lacking 0s until there are 11.
  if (manbits)
  {
    exp = 10; // exp bias because 1.0 is at bit position 10
    while (manbits > 11)
    {
      truncated |= absx & 1;
      absx >>= 1;
      manbits--;
      exp++;
    }
    while (manbits < 11)
    {
      absx <<= 1;
      manbits++;
      exp--;
    }
  }

  if (exp + truncated > 15)
  {
    // absx was too big, force it to +/- infinity
    exp = 31; // special infinity value
    absx = 0;
  }
  else if (manbits)
  {
    // normal case, absx > 0
    exp += 15; // bias the exponent
  }

  __half ret;
  ret.x = (sign << 15) | ((unsigned)exp << 10) | (absx & ((1u<<10)-1));

  return ret;
}

void GetDevice(){
//================== Retrieve and set the default CUDA device
    cudaDeviceProp prop;
    cudaError_t teste;
    int count=0;
    teste = cudaGetDeviceCount(&count);
	printf("\nGet Device Test: %s\n", cudaGetErrorString(teste));
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

void allocCudaMemory()
{
//================== CUDA error handlers
	cudaError_t malloc;
	const char *erro;
//====================================
	malloc = cudaMalloc( ( void** ) &d_A, sizea * sizeof( __half ) );
	erro = cudaGetErrorString(malloc);
	if(strcmp(erro, "no error") != 0) {
#ifdef LOGS
		log_error_detail("error a"); end_log_file();
#endif
		exit(EXIT_FAILURE);
	} //mem allocate failure

	malloc = cudaMalloc( ( void** ) &d_B, sizea * sizeof( __half ) );
	erro = cudaGetErrorString(malloc);
	if(strcmp(erro, "no error") != 0) {
#ifdef LOGS
		log_error_detail("error b"); end_log_file();
#endif
		exit(EXIT_FAILURE);
	} //mem allocate failure

	malloc = cudaMalloc( ( void** ) &d_C, sizea * sizeof( __half ) );
	erro = cudaGetErrorString(malloc);
	if(strcmp(erro, "no error") != 0) {
#ifdef LOGS
		log_error_detail("error c"); end_log_file();
#endif
		exit(EXIT_FAILURE);} //mem allocate failure
}

void copyCudaMemory()
{
//================== CUDA error handlers
	cudaError_t mcpy;
	const char *erro;
//====================================
	mcpy = cudaMemset(d_C, 0, sizea * sizeof (__half));
	erro = cudaGetErrorString(mcpy);
	if(strcmp(erro, "no error") != 0) {
#ifdef LOGS
		log_error_detail("error gpu load c"); end_log_file();
#endif
		exit(EXIT_FAILURE);} //mem allocate failure

	mcpy = cudaMemcpy( d_A, A, sizeb * sizeof( __half ), cudaMemcpyHostToDevice ); // PUSH A
	erro = cudaGetErrorString(mcpy);
	if(strcmp(erro, "no error") != 0) {
#ifdef LOGS
		log_error_detail("error gpu load b"); end_log_file();
#endif
		exit(EXIT_FAILURE);} //mem allocate failure

	mcpy = cudaMemcpy( d_B, B, sizeb * sizeof( __half ), cudaMemcpyHostToDevice ); // PUSH B
	erro = cudaGetErrorString(mcpy);
	if(strcmp(erro, "no error") != 0) {
#ifdef LOGS
		log_error_detail("error gpu load b"); end_log_file();
#endif
		exit(EXIT_FAILURE);} //mem allocate failure
}

void ReadMatrixFromFile(){
//================== Read inputs to HOST memory
	int i;
	double time = mysecond();
	f_A = fopen(a_matrix_path,"rb");
	f_B = fopen(b_matrix_path,"rb");
	f_GOLD = fopen(gold_matrix_path,"rb");
	if (!(f_A&&f_B&&f_GOLD))
	{
		printf ("Cant open matrices.\n");
#ifdef LOGS
		log_error_detail("Cant open matrices"); end_log_file();
#endif
		exit(-3);
	}
	for(i=0; i<k; i++)
	{
		fread (&A[ lda * i ], sizeof(__half)*k, 1, f_A);
		fread (&B[ lda * i ], sizeof(__half)*k, 1, f_B);
		fread (&GOLD[ lda * i ], sizeof(__half)*k, 1, f_GOLD);
	}
	if (verbose) printf("Done reading matrices in %.2fs\n", mysecond() - time);

	fclose(f_A);
	fclose(f_B);
	fclose(f_GOLD);

	if (fault_injection)
	{
		A[3] = int2hfloat(6.0);
		printf("!! Injected 6.0 on position A[3]\n");
	}
}

// __device__ int kerrors;

// __global__ void GoldChkKernel (__half *gk, __half *ck, int n)//, int *kerrors)
// {
// //================== HW Accelerated output validation
// 	int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
// 	int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y;
// 	//if ((fabs((gk[ty*n+tx]-ck[ty*n+tx])/gk[ty*n+tx]) > 0.0000000001)||(fabs((gk[ty*n+tx]-ck[ty*n+tx])/ck[ty*n+tx]) > 0.0000000001))
// 	if (gk[ty*n + tx].x != ck[ty*n + tx].x)
// 		atomicAdd(&kerrors, 1);

// }

void usage() {
    printf("Usage: cudaGemm -size=N [-input_a=<path>] [-input_b=<path>] [-gold=<path>] [-iterations=N] [-verbose] [-no-warmup]\n");
}

int main( int argc, char* argv[] )
{
//================== CUDA error handlers
	// cudaError_t mcpy;
	// const char *erro;
//====================================

//================== Test vars
	int i, j, loop2;
	// int kernel_errors=0;
	// int zero = 0;
	// double time;
	double kernel_time, global_time;
	int device_warmup = 1;
//====================================

//================== Read test parameters
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
            exit(EXIT_FAILURE);
        }
    }
	else
	{
		usage();
		exit(EXIT_FAILURE);
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

	if (checkCmdLineFlag(argc, (const char **)argv, "iterations"))
    {
        iterations = getCmdLineArgumentInt(argc, (const char **)argv, "iterations");
    }

	if (checkCmdLineFlag(argc, (const char **)argv, "verbose"))
    {
        verbose = 1;
    }

	if (checkCmdLineFlag(argc, (const char **)argv, "debug"))
    {
		fault_injection = 1;
        printf("!! Will be injected an input error\n");
    }

	if (checkCmdLineFlag(argc, (const char **)argv, "no-warmup"))
    {
		device_warmup = 0;
        printf("!! The first iteration may not reflect real timing information\n");
    }
//====================================

//================== Set block and grid size for GoldChk kernel
	int gridsize = k/BLOCK_SIZE < 1 ? 1 : k/BLOCK_SIZE;
	int blocksize = k/BLOCK_SIZE < 1 ? k : BLOCK_SIZE;
	dim3 dimBlock(blocksize,blocksize);
	dim3 dimGrid(gridsize,gridsize);
//====================================

//================== Init logs
#ifdef LOGS
	char test_info[90];
	snprintf(test_info, 90, "size:%d type:half-precision", k);
	start_log_file("cudaGEMM", test_info);
#endif
//====================================

//================== cublas GEMM parameters
	lda = max( 1, k + 16 );
	sizea = lda * k;
	ldb = max( 1, k + 16 );
	sizeb = ldb * k;
	ldc = max( 1, k + 16 );
	sizec = ldc * k;
//====================================

//================== Alloc HOST memory
	A = ( __half* ) malloc( sizea * sizeof( __half ) );
	B = ( __half* ) malloc( sizeb * sizeof( __half ) );
	C = ( __half* ) malloc( sizeb * sizeof( __half ) );

	GOLD = ( __half* ) malloc( sizec * sizeof( __half ) );

	if (!(A && B && C && GOLD)) {
		printf("Failed on host malloc.\n");
		exit(-3);
	}
//====================================

//================== Init test environment
	// kernel_errors=0;
	GetDevice();
	ReadMatrixFromFile();
	cublasHandle_t cublasHandle;
	checkCudaErrors( cublasCreate(&cublasHandle) );
	printf( "cublasHGEMM\n" );
	fflush(stdout);
//====================================

//================== Init DEVICE memory
	allocCudaMemory();
	copyCudaMemory();
//====================================


	for(loop2=0; loop2<iterations; loop2++)
	{//================== Global test loop

		if (!loop2 && device_warmup) printf("First iteration: device warmup. Please wait...\n");

		// Timer...
		global_time = mysecond();

		cudaMemset(d_C, 0, sizea * sizeof (__half));

		kernel_time = mysecond();
		#ifdef LOGS
		if (loop2 || !device_warmup)
			start_iteration();
		#endif
		//================== Device computation, GEMM
		checkCudaErrors( cublasHgemm(cublasHandle, transa, transb,
			   k, k, k,
			   &alpha,
			   d_A, lda,
			   d_B, ldb,
			   &beta,
			   d_C, ldc ) );
		checkCudaErrors( cudaDeviceSynchronize() );
		//====================================
		#ifdef LOGS
		if (loop2 || !device_warmup)
			end_iteration();
		#endif
		kernel_time = mysecond() - kernel_time;

		if (loop2 || !device_warmup)
			if (verbose) printf("Device kernel time for iteration %d: %.3fs\n", loop2, kernel_time);

		/*
		// Timer...
		time = mysecond();

		//================== Send GOLD to device, to perform HW output validation
		mcpy = cudaMemcpy(d_A, GOLD, sizea * sizeof( __half ), cudaMemcpyHostToDevice );
		erro = cudaGetErrorString(mcpy);
		if(strcmp(erro, "no error") != 0) {
			printf("error mem load gold\n");
			#ifdef LOGS
			log_error_detail("error mem load gold"); end_log_file();
			#endif
			return 1;
		} //mem allocate failure
		cudaMemcpyToSymbol(kerrors, &zero, sizeof(int));
		//====================================

		//================== Device computation, output validation
		GoldChkKernel<<<dimGrid,dimBlock>>>(d_A, d_C, ldc);
		cudaDeviceSynchronize();
		//====================================

		//================== Retrieve output mismatchs
		kernel_errors=0;
		cudaMemcpyFromSymbol(&kernel_errors, kerrors, sizeof(unsigned int));
		//====================================

		if (loop2 || !device_warmup)
			if (verbose) printf("Device gold check kernel time for iteration %d: %.3fs\n", loop2, mysecond() - time);

		//================== If there are errors, check on host (increased reliability)
		if (loop2 || !device_warmup)
			if (kernel_errors!=0)
			{

				printf(" kernel error: %d\n", kernel_errors);

				mcpy = cudaMemcpy(A, d_C, sizec * sizeof( __half ), cudaMemcpyDeviceToHost);
				erro = cudaGetErrorString(mcpy);
				if(strcmp(erro, "no error") != 0) {
					#ifdef LOGS
					log_error_detail("error mem down c"); end_log_file();
					#endif
					return 1;
				} //mem allocate failure
		*/
		if (memcmp(A, GOLD, sizeof(__half) * k*k)) {
			char error_detail[150];
			int host_errors = 0;

			#pragma omp parallel for
			for(i=0; (i<k); i++)
			{
				for(j=0; (j<k); j++)
				{
					if (A[i + ldc * j].x != GOLD[i + ldc * j].x)
					//if ((fabs((A[i+ldc*j]-GOLD[i+ldc*j])/A[i+ldc*j]) > 0.0000000001)||(fabs((A[i+ldc*j]-GOLD[i+ldc*j])/GOLD[i+ldc*j]) > 0.0000000001))
					#pragma omp critical
					{

						snprintf(error_detail, 150, "p: [%d, %d], r: %hd, e: %hd", i, j, A[i + ldc * j].x, GOLD[i + ldc * j].x);
						//printf("%s\n", error_detail);
						#ifdef LOGS
						log_error_detail(error_detail);
						#endif
						host_errors++;
						//ea++;
						//fprintf(file, "\n p: [%d, %d], r: %1.16e, e: %1.16e, error: %d\n", i, j, A[i + ldc * j], GOLD[i + ldc * j], t_ea);

					}
				}
			}

			#ifdef LOGS
				log_error_count(host_errors);
			#endif
			//================== Release device memory to ensure there is no corrupted data on the inputs of the next iteration
			cudaFree( d_A );
			cudaFree( d_B );
			cudaFree( d_C );
			//====================================
			ReadMatrixFromFile();
			//================== Init DEVICE memory
			allocCudaMemory();
			copyCudaMemory();
			//====================================
		}

		//====================================

		//================== Console hearthbeat
		/*if(kernel_errors > 0 || (loop2 % 10 == 0))
		{
			printf("test number: %d\n", loop2);
			printf(" kernel time: %f\n", kernel_time);
		}
		else
		{*/
			printf(".");
			fflush(stdout);
		//}
		//====================================

		// //================== Send A back to the device
		// mcpy = cudaMemcpy(d_A, A, sizea * sizeof( __half ), cudaMemcpyHostToDevice );
		// erro = cudaGetErrorString(mcpy);
		// if(strcmp(erro, "no error") != 0) {
		// 	printf("error mem load A\n");
		// 	#ifdef LOGS
		// 	log_error_detail("error mem load A"); end_log_file();
		// 	#endif
		// 	return 1;
		// } //mem allocate failure
		// //===================================

		if (loop2 || !device_warmup)
			if (verbose)
			{
				/////////// PERF
				double flops = 2.0*(double)k*k*k;
				double gflops = flops / kernel_time;
				double outputpersec = (double)k*k/kernel_time;
				printf("SIZE:%d OUTPUT/S:%f FLOPS:%f (GFLOPS:%.2f)\n",k, outputpersec, gflops, gflops/1000000000);
				///////////
			}

		if (loop2 || !device_warmup) 
			if (verbose) printf("Iteration #%d time: %.3fs\n\n\n", loop2, mysecond() - global_time);
		fflush(stdout);
	}

	//================== Release device memory
	cudaFree( d_A );
	cudaFree( d_B );
	cudaFree( d_C );
	//====================================

	free( A );
	free( B );
	#ifdef LOGS
	end_log_file();
	#endif

	return 0;
}
