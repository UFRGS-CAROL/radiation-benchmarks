#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <string>

#ifdef LOGS
#include "log_helper.h"
#endif

#include "cuda.h"
#include "cublas.h"

#define N_ERRORS_LOG 500
#define BLOCK_SIZE 32
int k=0; // N will be received on runtime
int iteractions=1; // iteractions will be received on runtime

double *h_A;
double *h_B;
double *h_GOLD;
double *d_A;
double *d_B;
double *d_C;

unsigned int *d_errpos;
unsigned int *errpos;

FILE* file;
FILE* log_file;
FILE* timefile;

using namespace std;

string gold_matrix_path, a_matrix_path, b_matrix_path;

void usage() {
    printf("Usage: cudaMxM <input_size> <A_MATRIX> <B_MATRIX> <GOLD_MATRIX> <#iteractions>\n");
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
    fread(h_A,sizeof(double)*k*k, 1, f_A);
    fread(h_B,sizeof(double)*k*k, 1, f_B);
    fread(h_GOLD,sizeof(double)*k*k, 1, f_GOLD);

	printf("Done reading matrices in %f\n", mysecond() - time);

    fclose(f_A);
    fclose(f_B);
    fclose(f_GOLD);
}

__device__ int kerrors;

__global__ void GoldChkKernel (double *gk, double *ck, int n)
{
	int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                                                      
	int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y; 
	if ((fabs((gk[ty*n+tx]-ck[ty*n+tx])/gk[ty*n+tx]) > 0.0000000001)||(fabs((gk[ty*n+tx]-ck[ty*n+tx])/ck[ty*n+tx]) > 0.0000000001))
		atomicAdd(&kerrors, 1);

}

__global__ void MatrixMulKernel (double *d_A, double *d_B, double *d_C, int n)
{
	int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                                                      
	int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y; 
	int k;
	
	d_C[ty*n + tx] = 0.0;
	for (k = 0;  k < n; k++)
	  d_C[ty*n + tx] += d_A[ty*n + k]*d_B[k*n + tx];

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
	start_log_file("cudaMxM", test_info);
#endif

	int size = k*k;

	h_A = ( double* ) malloc( size * sizeof( double ) );
	h_B = ( double* ) malloc( size * sizeof( double ) );
	h_GOLD = ( double* ) malloc( size * sizeof( double ) );

	ReadMatrixFromFile();


	kernel_errors=0;
	
	GetDevice();

	printf( "Cuda MxM Not optimized - %ix%i\n", k, k );


	for(loop2=0; loop2<iteractions; loop2++)
	{

	//	file = fopen(file_name, "a");	
	
		// ======> DEVICE MEMORY ALLOC
		malloc_a = cudaMalloc( ( void** ) &d_A, size * sizeof( double ) );
		erro_malloc = cudaGetErrorString(malloc_a);
		if(strcmp(erro_malloc, "no error") != 0) if(strcmp(erro_malloc, "no error") != 0) {
#ifdef LOGS
			log_error_detail("error a"); end_log_file(); 
#endif
			return 1;} //mem allocate failure

		malloc_a = cudaMalloc( ( void** ) &d_B, size * sizeof( double ) );
		erro_malloc = cudaGetErrorString(malloc_a);
		if(strcmp(erro_malloc, "no error") != 0) if(strcmp(erro_malloc, "no error") != 0) {
#ifdef LOGS
			log_error_detail("error b"); end_log_file(); 
#endif
			return 1;} //mem allocate failure

		malloc_a = cudaMalloc( ( void** ) &d_C, size * sizeof( double ) );
		erro_malloc = cudaGetErrorString(malloc_a);
		if(strcmp(erro_malloc, "no error") != 0) if(strcmp(erro_malloc, "no error") != 0) {
#ifdef LOGS
			log_error_detail("error c"); end_log_file(); 
#endif
			return 1;} //mem allocate failure
		// =======>


		malloc_mem1 = cudaMemcpy( d_C, h_A, size * sizeof( double ), cudaMemcpyHostToDevice ); // ZERA C
		erro_malloc = cudaGetErrorString(malloc_mem1);
		if(strcmp(erro_malloc, "no error") != 0) if(strcmp(erro_malloc, "no error") != 0) {
#ifdef LOGS
			log_error_detail("error load c"); end_log_file(); 
#endif
			return 1;} //mem allocate failure
	
		malloc_mem1 = cudaMemcpy( d_A, h_A, size * sizeof( double ), cudaMemcpyHostToDevice ); // PUSH A
		erro_malloc = cudaGetErrorString(malloc_mem1);
		if(strcmp(erro_malloc, "no error") != 0) if(strcmp(erro_malloc, "no error") != 0) {
#ifdef LOGS
			log_error_detail("error load a"); end_log_file(); 
#endif
			return 1;} //mem allocate failure

		malloc_mem1 = cudaMemcpy( d_B, h_B, size * sizeof( double ), cudaMemcpyHostToDevice ); // PUSH B
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


		malloc_mem1 = cudaMemcpy(d_A, h_GOLD, size * sizeof( double ), cudaMemcpyHostToDevice );
		erro_malloc = cudaGetErrorString(malloc_mem1);
		if(strcmp(erro_malloc, "no error") != 0) {
#ifdef LOGS
			log_error_detail("error mem load gold"); end_log_file(); 
#endif
			return 1;} //mem allocate failure
		// =======>

		cudaMemcpyToSymbol(kerrors, &zero, sizeof(int));

		GoldChkKernel<<<dimGrid,dimBlock>>>(d_A, d_C, k);


		cudaMemcpyFromSymbol(&kernel_errors, kerrors, sizeof(unsigned int));
	
#ifdef LOGS
		log_error_count(kernel_errors);
#endif

		if (kernel_errors!=0)
		{
			char error_detail[150];

			printf("\n kernel error: %d\n", kernel_errors);

			malloc_mem1 = cudaMemcpy(h_A, d_C, size * sizeof( double ), cudaMemcpyDeviceToHost);
			erro_malloc = cudaGetErrorString(malloc_mem1);
			if(strcmp(erro_malloc, "no error") != 0) {
#ifdef LOGS
				log_error_detail("error mem load c"); end_log_file(); 
#endif
			return 1;} //mem allocate failure

			for(i=0; (i<k) && (ea < N_ERRORS_LOG); i++)
			{
				for(j=0; (j<k) && (ea < N_ERRORS_LOG); j++)
				{
					if ((fabs((h_A[i+k*j]-h_GOLD[i+k*j])/h_A[i+k*j]) > 0.0000000001)||(fabs((h_A[i+k*j]-h_GOLD[i+k*j])/h_GOLD[i+k*j]) > 0.0000000001))
					{
						snprintf(error_detail, 150, "p: [%d, %d], r: %1.16e, e: %1.16e", i, j, h_A[i + k * j], h_GOLD[i + k * j]);
#ifdef LOGS
						log_error_detail(error_detail);
#endif
						//ea++;
						//fprintf(file, "\n p: [%d, %d], r: %1.16e, e: %1.16e, error: %d\n", i, j, h_A[i + k * j], h_GOLD[i + k * j], ea);
										
					}
				}
			}

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
	free(h_B);
	free(h_GOLD);
#ifdef LOGS
	end_log_file();
#endif

	return 0;
}
