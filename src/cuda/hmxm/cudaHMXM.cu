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
#include "half.hpp"

// helper functions
#include "helper_string.h"
#include "helper_cuda.h"

#undef min
#define min( x, y ) ( (x) < (y) ? (x) : (y) )
#undef max
#define max( x, y ) ( (x) > (y) ? (x) : (y) )

#define BLOCK_SIZE 32

#define DEFAULT_INPUT_SIZE 8192

int verbose = 0;
int fault_injection = 0;

int size=0; // k x k matrix size
int iterations=100000000; // global loop iteracion

//================== Input paths
char *gold_matrix_path, *a_matrix_path, *b_matrix_path;

FILE* f_A;
FILE* f_B;
FILE* f_GOLD;
//====================================

//================== Host and device matrix ptr's
half_float::half *A;
half_float::half *B_T;
half_float::half *C;
half_float::half *GOLD;

half *d_A;
half *d_B_T;
half *d_C;
//====================================

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
	malloc = cudaMalloc( ( void** ) &d_A, size * size * sizeof( half ) );
	erro = cudaGetErrorString(malloc);
	if(strcmp(erro, "no error") != 0) {
#ifdef LOGS
		log_error_detail("error a"); end_log_file();
#endif
		exit(EXIT_FAILURE);
	} //mem allocate failure

	malloc = cudaMalloc( ( void** ) &d_B_T, size * size * sizeof( half ) );
	erro = cudaGetErrorString(malloc);
	if(strcmp(erro, "no error") != 0) {
#ifdef LOGS
		log_error_detail("error b"); end_log_file();
#endif
		exit(EXIT_FAILURE);
	} //mem allocate failure

	malloc = cudaMalloc( ( void** ) &d_C, size * size * sizeof( half ) );
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
	mcpy = cudaMemset(d_C, 0, size * size * sizeof (half));
	erro = cudaGetErrorString(mcpy);
	if(strcmp(erro, "no error") != 0) {
#ifdef LOGS
		log_error_detail("error gpu load c"); end_log_file();
#endif
		exit(EXIT_FAILURE);} //mem allocate failure

	mcpy = cudaMemcpy( d_A, A, size * size * sizeof( half ), cudaMemcpyHostToDevice ); // PUSH A
	erro = cudaGetErrorString(mcpy);
	if(strcmp(erro, "no error") != 0) {
#ifdef LOGS
		log_error_detail("error gpu load b"); end_log_file();
#endif
		exit(EXIT_FAILURE);} //mem allocate failure

	mcpy = cudaMemcpy( d_B_T, B_T, size * size * sizeof( half ), cudaMemcpyHostToDevice ); // PUSH B
	erro = cudaGetErrorString(mcpy);
	if(strcmp(erro, "no error") != 0) {
#ifdef LOGS
		log_error_detail("error gpu load b"); end_log_file();
#endif
		exit(EXIT_FAILURE);} //mem allocate failure
}

void ReadMatrixFromFile(){
//================== Read inputs to HOST memory
	int i, j;
	if (verbose) printf("Reading matrices... ");
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
	half_float::half *B = (half_float::half*)malloc(sizeof(half_float::half) * size * size);
	if (!B) {
		printf ("Cant alloc B matrice.\n");
#ifdef LOGS
		log_error_detail("Cant alloc B matrice"); end_log_file();
#endif
		exit(-3);
	}
    size_t ret_value[3];
    for(i=0; i<size; i++)
    {
      ret_value[0] = fread (&(A[ size * i ]), sizeof(half)*size, 1, f_A);
      ret_value[1] = fread (&(B[ size * i ]), sizeof(half)*size, 1, f_B);
      ret_value[2] = fread (&(GOLD[ size * i ]), sizeof(half)*size, 1, f_GOLD);
      if (ret_value[0] != 1 || ret_value[1] != 1 || ret_value[2] != 1) {
         printf("Bad input/gold formatting: %lu ; %lu ; %lu .\n", ret_value[0], ret_value[1], ret_value[2]);
         #ifdef LOGS
    		log_error_detail("Bad input/gold formatting."); end_log_file();
         #endif
    		exit(-3);
      }
    }
	if (verbose) printf("Done reading matrices in %.2fs\n", mysecond() - time);

	fclose(f_A);
	fclose(f_B);
	fclose(f_GOLD);

	for (i=0; i<size; i++) {
		for (j=0; j<size; j++) {
			B_T[i*size+j] = B[j*size+i];
		}
	}
	
	free(B);

	if (fault_injection)
	{
        half_float::half tempValue(6.5);
		A[3] = *((half_float::half*)&tempValue);
		printf("!! Injected 6.5 on position A[3]\n");
	}
}

bool badass_memcmp(half_float::half *gold, half_float::half *found, unsigned long n){
	double result = 0.0;
	int i;
	unsigned long  chunk = ceil(float(n) / float(omp_get_max_threads()));
	// printf("size %d max threads %d chunk %d\n", n, omp_get_max_threads(), chunk);
	double time = mysecond();
#pragma omp parallel for default(shared) private(i) schedule(static,chunk) reduction(+:result)
   for (i=0; i < n; i++)
     result = result + (gold[i] - found[i]);

    //  printf("comparing took %lf seconds, diff %lf\n", mysecond() - time, result);
	if (fabs(result) > 0.0000000001)
		return true;
	return false;
}

// __device__ int kerrors;
//
// __global__ void GoldChkKernel (half *gk, half *ck, int n)//, int *kerrors)
// {
// //================== HW Accelerated output validation
// 	int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
// 	int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y;
// 	//if ((fabs((gk[ty*n+tx]-ck[ty*n+tx])/gk[ty*n+tx]) > 0.0000000001)||(fabs((gk[ty*n+tx]-ck[ty*n+tx])/ck[ty*n+tx]) > 0.0000000001))
// 	if (gk[ty*n + tx].x != ck[ty*n + tx].x)
// 		atomicAdd(&kerrors, 1);
//
// }

__global__ void MatrixMulKernel_T (half *d_A, half *d_B_T, half *d_C, int n)
{
	int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                                                      
	int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y; 
	int k;
	half2 *d_A2 = (half2*)d_A;
	half2 *d_B_T2 = (half2*)d_B_T;
	half2 *d_C2 = (half2*)d_C;
	
	d_C2[ty*n + tx] = __float2half2_rn(0.0);
	for (k = 0;  k < n; k++)
		d_C2[ty*n + tx] = __hfma2(d_A2[ty*n + k], d_B_T2[ty*n + k], d_C2[ty*n + tx]);

}

void usage() {
    printf("Usage: hmxm -size=N [-input_a=<path>] [-input_b=<path>] [-gold=<path>] [-iterations=N] [-verbose] [-no-warmup]\n");
}

int main( int argc, char* argv[] )
{
//================== CUDA error handlers
	cudaError_t mcpy;
	const char *erro;
//====================================

//================== Test vars
	int i, j, loop2;
	// int kernel_errors=0;
	// int zero = 0;
	double time;
	double kernel_time, global_time;
    double total_kernel_time, min_kernel_time, max_kernel_time;
	int device_warmup = 1;
    // int gpu_check = 1;
//====================================

//================== Read test parameters
	if (argc<2) {
		usage();
		exit (-1);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "size"))
    {
        size = getCmdLineArgumentInt(argc, (const char **)argv, "size");

        if ((size <= 0)||(size % 16 != 0))
        {
            printf("Invalid input size given on the command-line: %d\n", size);
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
        snprintf(gold_matrix_path, 100, "hgemm_gold_%i", (signed int)size);
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

	// if (checkCmdLineFlag(argc, (const char **)argv, "no-gpu-gold-check"))
    // {
	// 	gpu_check = 0;
    // } else {
    //     printf("!! The gold check will happen on the GPU and fall back to CPU in case of errors\n");
    // }
//====================================

//================== Set block and grid size for GoldChk kernel
	int gridsize = size/BLOCK_SIZE < 1 ? 1 : size/BLOCK_SIZE;
	int blocksize = size/BLOCK_SIZE < 1 ? size : BLOCK_SIZE;
	dim3 dimBlock(blocksize,blocksize);
	dim3 dimGrid(gridsize,gridsize);
//====================================

//================== Init logs
#ifdef LOGS
	char test_info[90];
	snprintf(test_info, 90, "size:%d type:half-precision", size);
	start_log_file("cudaHMxM", test_info);
#endif
//====================================

//================== Alloc HOST memory
	A = ( half_float::half* ) malloc( size * size * sizeof( half ) );
	B_T = ( half_float::half* ) malloc( size * size * sizeof( half ) );
	C = ( half_float::half* ) malloc( size * size * sizeof( half ) );

	GOLD = ( half_float::half* ) malloc( size * size * sizeof( half ) );

	if (!(A && B_T && C && GOLD)) {
		printf("Failed on host malloc.\n");
		exit(-3);
	}
//====================================

//================== Init test environment
	// kernel_errors=0;
    total_kernel_time = 0;
    min_kernel_time = UINT_MAX;
    max_kernel_time = 0;
	GetDevice();
	ReadMatrixFromFile();
	cublasHandle_t cublasHandle;
	checkCudaErrors( cublasCreate(&cublasHandle) );
	printf( "cudaHMxM\n" );
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

		cudaMemset(d_C, 0, size * sizeof (half));

		if (verbose) printf(",");

		kernel_time = mysecond();
		#ifdef LOGS
		if (loop2 || !device_warmup)
			start_iteration();
		#endif
		//================== Device computation, HMxM
		MatrixMulKernel_T<<<gridsize, blocksize>>>(d_A, d_B_T, d_C, size);
		checkCudaErrors( cudaDeviceSynchronize() );
		//====================================
		#ifdef LOGS
		if (loop2 || !device_warmup)
			end_iteration();
		#endif
		kernel_time = mysecond() - kernel_time;
      
		if (loop2 || !device_warmup) {
		  total_kernel_time += kernel_time;
		  min_kernel_time = min(min_kernel_time, kernel_time);
		  max_kernel_time = max(max_kernel_time, kernel_time);
		}

		if (loop2 || !device_warmup)
			if (verbose) printf("Device kernel time for iteration %d: %.3fs\n", loop2, kernel_time);

    	if (verbose) printf(",");

        // Timer...
        time = mysecond();

        //if (kernel_errors != 0) {
        if (loop2 || !device_warmup) {
            mcpy = cudaMemcpy(A, d_C, size * size * sizeof( half ), cudaMemcpyDeviceToHost );
            erro = cudaGetErrorString(mcpy);
            if(strcmp(erro, "no error") != 0) {
                printf("error mem load gold to host\n");
                #ifdef LOGS
                    log_error_detail("error mem load gold to host"); end_log_file();
                #endif
                return 1;
            } //mem allocate failure
            //~ if (memcmp(A, GOLD, sizeof(double) * k*k)) {
            if (badass_memcmp(GOLD, A, size * size)){
    			char error_detail[150];
    			int host_errors = 0;

                printf("!");

    			#pragma omp parallel for
    			for(i=0; (i<size); i++)
    			{
    				for(j=0; (j<size); j++)
    				{
    					if (A[i + size * j] != GOLD[i + size * j])
    					//if ((fabs((A[i+size*j]-GOLD[i+size*j])/A[i+size*j]) > 0.0000000001)||(fabs((A[i+size*j]-GOLD[i+size*j])/GOLD[i+size*j]) > 0.0000000001))
    					#pragma omp critical
    					{

    						snprintf(error_detail, 150, "p: [%d, %d], r: %1.16e, e: %1.16e", i, j, A[i + size * j], GOLD[i + size * j]);
    						printf("%s\n", error_detail);
    						#ifdef LOGS
    						log_error_detail(error_detail);
    						#endif
    						host_errors++;
    						//ea++;
    						//fprintf(file, "\n p: [%d, %d], r: %1.16e, e: %1.16e, error: %d\n", i, j, A[i + size * j], GOLD[i + size * j], t_ea);

    					}
    				}
    			}

                // printf("numErrors:%d", host_errors);

    			#ifdef LOGS
    				log_error_count(host_errors);
    			#endif
    			//================== Release device memory to ensure there is no corrupted data on the inputs of the next iteration
    			cudaFree( d_A );
    			cudaFree( d_B_T );
    			cudaFree( d_C );
    			//====================================
    			ReadMatrixFromFile();
    			//================== Init DEVICE memory
    			allocCudaMemory();
    			copyCudaMemory();
    			//====================================
    		}
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

		if (loop2 || !device_warmup)
			if (verbose) printf("Gold check time for iteration %d: %.3fs\n", loop2, mysecond() - time);

		if (loop2 || !device_warmup)
			if (verbose)
			{
				/////////// PERF
				double flops = 2.0*(double)size*size*size;
				double gflops = flops / kernel_time;
				double outputpersec = (double)size*size/kernel_time;
				printf("SIZE:%d OUTPUT/S:%f FLOPS:%f (GFLOPS:%.2f)\n",size, outputpersec, gflops, gflops/1000000000);
				///////////
			}

		if (loop2 || !device_warmup)
			if (verbose) printf("Iteration #%d time: %.3fs\n\n\n", loop2, mysecond() - global_time);
		fflush(stdout);
	}

    double gflops = 2.0*(double)size*size*size / 1000000000; // Bilion FLoating-point OPerationS
    double averageKernelTime = total_kernel_time / (iterations - (device_warmup ? 1 : 0));
    printf("\n-- END --\n"
    "Total kernel time: %.3fs\n"
    "Iterations: %d\n"
    "Average kernel time: %.3fs (best: %.3fs ; worst: %.3fs)\n"
    "Average GFLOPs: %.2f (best: %.2f ; worst: %.2f)\n", 
    total_kernel_time, 
    iterations, 
    averageKernelTime, min_kernel_time, max_kernel_time,
    gflops / averageKernelTime, gflops / min_kernel_time, gflops / max_kernel_time);

	//================== Release device memory
	cudaFree( d_A );
	cudaFree( d_B_T );
	cudaFree( d_C );
	//====================================

	free( A );
	free( B_T );
	#ifdef LOGS
	end_log_file();
	#endif

	return 0;
}
