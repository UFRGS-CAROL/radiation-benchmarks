#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <string>

#include "half.hpp"

#ifdef LOGS
#include "log_helper.h"
#endif

#include "cuda.h"
#include "cublas.h"

#define N_ERRORS_LOG 500
#define BLOCK_SIZE 32

int k=0; // N will be received on runtime
int iteractions=1; // iteractions will be received on runtime

half_float::half *h_A;
half_float::half *h_B_T;
half_float::half *h_C;
half_float::half *h_GOLD;
half_float::half *d_A;
half_float::half *d_B;
half_float::half *d_C;

unsigned int *d_errpos;
unsigned int *errpos;

FILE* file;
FILE* log_file;
FILE* timefile;

using namespace std;

string gold_matrix_path, a_matrix_path, b_matrix_path;

void usage() {
    printf("Usage: hmxm <input_size> <A_MATRIX> <B_MATRIX> <GOLD_MATRIX> <#iteractions>\n");
}

void GetDevice(){

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
	printf("device: %d %s\n\n", *ndevice, prop.name);

}

double mysecond()
{
   struct timeval tp;
   struct timezone tzp;
   int i = gettimeofday(&tp,&tzp);
   return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void ReadMatrixFromFile(){

	double time = mysecond();
	FILE *f_A, *f_B, *f_GOLD;

	f_A = fopen(a_matrix_path.c_str(),"rb");
	f_B = fopen(b_matrix_path.c_str(),"rb");
	f_GOLD = fopen(gold_matrix_path.c_str(),"rb");
	if (!(f_A&&f_B))
	{
		printf("Error opening matrices A, B.\n");
		exit(-1);
	}
    fread(h_A,sizeof(half_float::half)*k*k, 1, f_A);
    fread(h_B_T,sizeof(half_float::half)*k*k, 1, f_B);
	fread(h_GOLD,sizeof(half_float::half)*k*k, 1, f_GOLD);
	
	// Transpose h_B
	for (int i=0; i<k; i++) {
		for (int j=0; j<k; j++) {
			half_float::half tempValue = h_B_T[i*k + j];
			h_B_T[i*k + j] = h_B_T[j*k + i];
			h_B_T[j*k + i] = tempValue;
		}
	}

	printf("Done reading matrices in %f\n", mysecond() - time);

    fclose(f_A);
    fclose(f_B);
    fclose(f_GOLD);
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

__global__ void MatrixMulKernel (half *d_A, half *d_B_T, half *d_C, int n)
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

int main( int argc, char* argv[] )
{

	
	cudaError_t malloc_mem1;
	cudaError_t malloc_a;
	const char *erro_malloc;


	int ea=0; //wrong integers in the current loop

	int i, j, loop2;

	int kernel_errors=0;
	int zero = 0;


	////////////////////////////////////////////////////
	////////////////////GET PARAM///////////////////////
	if (argc!=6) {
		usage();
		exit (-1);
	}

	k = atoi (argv[1]);
	iteractions = atoi (argv[5]);
	if (((k%32)!=0)||(k<0)){
		printf ("Enter a valid input. (k=%i)\n", k);
		exit (-1);
	}

	a_matrix_path = argv[2];
	b_matrix_path = argv[3];
	gold_matrix_path = argv[4];

	//////////BLOCK and GRID size///////////////////////
	int gridsize = k/BLOCK_SIZE < 1 ? 1 : k/BLOCK_SIZE;
	int blocksize = k/BLOCK_SIZE < 1 ? k : BLOCK_SIZE;
	dim3 dimBlock(blocksize,blocksize);
	dim3 dimGrid(gridsize,gridsize);
	////////////////////////////////////////////////////

#ifdef LOGS
	char test_info[100];
	snprintf(test_info, 100, "size:%d",k);
	start_log_file("cudaHMXM", test_info);
#endif

	int size = k*k;

	h_A = ( half_float::half* ) malloc( size * sizeof( half_float::half ) );
	h_B_T = ( half_float::half* ) malloc( size * sizeof( half_float::half ) );
	h_C = ( half_float::half* ) malloc( size * sizeof( half_float::half ) );
	h_GOLD = ( half_float::half* ) malloc( size * sizeof( half_float::half ) );

	ReadMatrixFromFile();

	kernel_errors=0;
	
	GetDevice();

	printf( "CUDA HMXM Not optimized - %ix%i\n", k, k );


	for(loop2=0; loop2<iteractions; loop2++)
	{

	//	file = fopen(file_name, "a");	
	
		// ======> DEVICE MEMORY ALLOC
		malloc_a = cudaMalloc( ( void** ) &d_A, size * sizeof( half_float::half ) );
		erro_malloc = cudaGetErrorString(malloc_a);
		if(strcmp(erro_malloc, "no error") != 0) if(strcmp(erro_malloc, "no error") != 0) {
#ifdef LOGS
			log_error_detail("error a"); end_log_file(); 
#endif
			return 1;} //mem allocate failure

		malloc_a = cudaMalloc( ( void** ) &d_B, size * sizeof( half_float::half ) );
		erro_malloc = cudaGetErrorString(malloc_a);
		if(strcmp(erro_malloc, "no error") != 0) if(strcmp(erro_malloc, "no error") != 0) {
#ifdef LOGS
			log_error_detail("error b"); end_log_file(); 
#endif
			return 1;} //mem allocate failure

		malloc_a = cudaMalloc( ( void** ) &d_C, size * sizeof( half_float::half ) );
		erro_malloc = cudaGetErrorString(malloc_a);
		if(strcmp(erro_malloc, "no error") != 0) if(strcmp(erro_malloc, "no error") != 0) {
#ifdef LOGS
			log_error_detail("error c"); end_log_file(); 
#endif
			return 1;} //mem allocate failure
		// =======>


		malloc_mem1 = cudaMemcpy( d_C, h_A, size * sizeof( half_float::half ), cudaMemcpyHostToDevice ); // ZERA C
		erro_malloc = cudaGetErrorString(malloc_mem1);
		if(strcmp(erro_malloc, "no error") != 0) if(strcmp(erro_malloc, "no error") != 0) {
#ifdef LOGS
			log_error_detail("error load c"); end_log_file(); 
#endif
			return 1;} //mem allocate failure
	
		malloc_mem1 = cudaMemcpy( d_A, h_A, size * sizeof( half_float::half ), cudaMemcpyHostToDevice ); // PUSH A
		erro_malloc = cudaGetErrorString(malloc_mem1);
		if(strcmp(erro_malloc, "no error") != 0) if(strcmp(erro_malloc, "no error") != 0) {
#ifdef LOGS
			log_error_detail("error load a"); end_log_file(); 
#endif
			return 1;} //mem allocate failure

		malloc_mem1 = cudaMemcpy( d_B, h_B_T, size * sizeof( half_float::half ), cudaMemcpyHostToDevice ); // PUSH B
		erro_malloc = cudaGetErrorString(malloc_mem1);
		if(strcmp(erro_malloc, "no error") != 0) if(strcmp(erro_malloc, "no error") != 0) {
#ifdef LOGS
			log_error_detail("error load b"); end_log_file(); 
#endif
			return 1;} //mem allocate failure

		kernel_errors=0;
double time = mysecond();
#ifdef LOGS
		start_iteration();
#endif
		MatrixMulKernel<<<dimGrid,dimBlock>>>(d_A, d_B, d_C, k);
		cudaDeviceSynchronize();
#ifdef LOGS
		end_iteration();
#endif
time = mysecond() - time;


		malloc_mem1 = cudaMemcpy(d_C, h_C, size * sizeof( half_float::half ), cudaMemcpyHostToDevice );
		erro_malloc = cudaGetErrorString(malloc_mem1);
		if(strcmp(erro_malloc, "no error") != 0) {
#ifdef LOGS
			log_error_detail("error mem load gold"); end_log_file(); 
#endif
			return 1;} //mem allocate failure
		// =======>

		if (badass_memcmp(h_GOLD, h_C, size))
		{
			char error_detail[150];
			kernel_errors = 0;

			printf("!");

			malloc_mem1 = cudaMemcpy(h_A, d_C, size * sizeof( half_float::half ), cudaMemcpyDeviceToHost);
			erro_malloc = cudaGetErrorString(malloc_mem1);
			if(strcmp(erro_malloc, "no error") != 0) {
#ifdef LOGS
				log_error_detail("error mem load c"); end_log_file(); 
#endif
			return 1;} //mem allocate failure

			for(i=0; (i<k) && ; i++)
			{
				for(j=0; (j<k) && (ea < N_ERRORS_LOG); j++)
				{
					kernel_errors++;
					if (ea < N_ERRORS_LOG) {
						if ((fabs((h_A[i+k*j]-h_GOLD[i+k*j])/h_A[i+k*j]) > 0.0000000001)||(fabs((h_A[i+k*j]-h_GOLD[i+k*j])/h_GOLD[i+k*j]) > 0.0000000001))
						{
							snprintf(error_detail, 150, "p: [%d, %d], r: %1.16e, e: %1.16e", i, j, h_A[i + k * j], h_GOLD[i + k * j]);
#ifdef LOGS
							log_error_detail(error_detail);
#endif
							ea++;
						}
					}
				}
			}


#ifdef LOGS
				log_error_count(host_errors);
#endif
			printf("kernel_errors:%d\n", kernel_errors);
			ReadMatrixFromFile();
		}



		if(kernel_errors > 0 || (loop2 % 10 == 0))
		{
			printf("\ntest number: %d", loop2);
			printf("\nerrors: %d", kernel_errors);
			printf(" time: %f\n", time);
		}
		else
		{
			printf(".");
			fflush(stdout);
		}

		cudaFree( d_A );
		cudaFree( d_B );
		cudaFree( d_C );
	}

	free(h_A);
	free(h_B_T);
	free(h_GOLD);
#ifdef LOGS
	end_log_file();
#endif

	return 0;
}
