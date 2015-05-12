#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <string>

//#include "log_helper.h"

#include "cuda.h"
#include "cublas.h"

#define MATRIX_PATH "/home/carol/TestGPU/GenerateGoldMatrix/Double_"

#define N_ERRORS_LOG 500
#define BLOCK_SIZE 32

#ifndef ITERACTIONS
#define ITERACTIONS 100000000000000000
#endif

int k=0; // N will be received on runtime

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

void log_error_detail(char* err)
{printf(err);
return;}

void end_log_file(){return;}

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

		
printf("open matrix...");
	f_A = fopen(a_matrix_path.c_str(),"rb");
	f_B = fopen(b_matrix_path.c_str(),"rb");
	f_GOLD = fopen(gold_matrix_path.c_str(),"rb");
	if (!(f_A && f_B && f_GOLD)) { printf ("Error opening matrix.\n"); getchar(); exit(-1); }
printf("read...");
        fread(h_A,sizeof(double)*k*k, 1, f_A);
        fread(h_B,sizeof(double)*k*k, 1, f_B);
        fread(h_GOLD,sizeof(double)*k*k, 1, f_GOLD);
printf("ok in %f\n", mysecond() - time);


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
	if (argc!=2) {
		printf ("Enter the required input. (1024/2048/4096/8192)\n");
		exit (-1);
	}
	k = atoi (argv[1]);
	if (((k%32)!=0)||(k<0)){
		printf ("Enter a valid input. (k=%i)\n", k);
		exit (-1);
	}
	string matrix_size_str(argv[1]);

	a_matrix_path = MATRIX_PATH;
	b_matrix_path = MATRIX_PATH;
	gold_matrix_path = MATRIX_PATH;
	a_matrix_path += "A_8192.matrix";
	b_matrix_path += "B_8192.matrix";
	gold_matrix_path += "GOLD_" + matrix_size_str + ".matrix";

	//////////BLOCK and GRID size///////////////////////
	int gridsize = k/BLOCK_SIZE < 1 ? 1 : k/BLOCK_SIZE;
	int blocksize = k/BLOCK_SIZE < 1 ? k : BLOCK_SIZE;
	dim3 dimBlock(blocksize,blocksize);
	dim3 dimGrid(gridsize,gridsize);
	////////////////////////////////////////////////////


	char test_info[100];
	snprintf(test_info, 100, "size:%d",k);
	//start_log_file(LOGFILE_MATRIXNAME, test_info);

	int size = k*k;

	h_A = ( double* ) malloc( size * sizeof( double ) );
	h_B = ( double* ) malloc( size * sizeof( double ) );
	h_GOLD = ( double* ) malloc( size * sizeof( double ) );

	ReadMatrixFromFile();


	kernel_errors=0;
	
	GetDevice();

	printf( "Cuda MxM Not optimized - %ix%i\n", k, k );


	for(loop2=0; loop2<ITERACTIONS; loop2++)
	{

	//	file = fopen(file_name, "a");	
	
		// ======> DEVICE MEMORY ALLOC
		malloc_a = cudaMalloc( ( void** ) &d_A, size * sizeof( double ) );
		erro_malloc = cudaGetErrorString(malloc_a);
		if(strcmp(erro_malloc, "no error") != 0) {printf("error a"); log_error_detail("error a"); end_log_file(); return 1;} //mem allocate failure

		malloc_a = cudaMalloc( ( void** ) &d_B, size * sizeof( double ) );
		erro_malloc = cudaGetErrorString(malloc_a);
		if(strcmp(erro_malloc, "no error") != 0) {printf("error b"); log_error_detail("error b"); end_log_file(); return 1;} //mem allocate failure

		malloc_a = cudaMalloc( ( void** ) &d_C, size * sizeof( double ) );
		erro_malloc = cudaGetErrorString(malloc_a);
		if(strcmp(erro_malloc, "no error") != 0) {printf("error c"); log_error_detail("error c"); end_log_file(); return 1;} //mem allocate failure
		// =======>


		malloc_mem1 = cudaMemcpy( d_C, h_A, size * sizeof( double ), cudaMemcpyHostToDevice ); // ZERA C
		erro_malloc = cudaGetErrorString(malloc_mem1);
		if(strcmp(erro_malloc, "no error") != 0) {printf("error mem load c %s", erro_malloc); log_error_detail("error mem load c"); end_log_file(); return 1;}
	
		malloc_mem1 = cudaMemcpy( d_A, h_A, size * sizeof( double ), cudaMemcpyHostToDevice ); // PUSH A
		erro_malloc = cudaGetErrorString(malloc_mem1);
		if(strcmp(erro_malloc, "no error") != 0) {printf("error mem load a %s", erro_malloc); log_error_detail("error mem load a"); end_log_file(); return 1;}

		malloc_mem1 = cudaMemcpy( d_B, h_B, size * sizeof( double ), cudaMemcpyHostToDevice ); // PUSH B
		erro_malloc = cudaGetErrorString(malloc_mem1);
		if(strcmp(erro_malloc, "no error") != 0) {printf("error mem load b %s", erro_malloc); log_error_detail("error mem load b"); end_log_file(); return 1;}


		kernel_errors=0;
	
		//timeG = mysecond();
		//start_iteration();
		MatrixMulKernel<<<dimGrid,dimBlock>>>(d_A, d_B, d_C, k);
		cudaDeviceSynchronize();
		//end_iteration();
		//timeG = mysecond() - timeG;

		//total_time += timeG;


		malloc_mem1 = cudaMemcpy(d_A, h_GOLD, size * sizeof( double ), cudaMemcpyHostToDevice );
		erro_malloc = cudaGetErrorString(malloc_mem1);
		if(strcmp(erro_malloc, "no error") != 0) {printf("error mem load gold %s", erro_malloc); log_error_detail("error mem load gold"); end_log_file(); return 1;}
		// =======>

		cudaMemcpyToSymbol(kerrors, &zero, sizeof(int));

		GoldChkKernel<<<dimGrid,dimBlock>>>(d_A, d_C, k);


		cudaMemcpyFromSymbol(&kernel_errors, kerrors, sizeof(unsigned int));
	

	//	///////////UPDATE FILE//////////////////////
	//	file_time = time(NULL);
	//	ptm = gmtime(&file_time);
	//	snprintf(hour, sizeof(hour + 1), "%d", ptm->tm_hour);
	//	snprintf(minute, sizeof(minute + 1), "%d", ptm->tm_min);
	//	snprintf(second, sizeof(second + 1), "%d", ptm->tm_sec);
	//	fprintf(file, "\n start time: %s/%s_%s:%s:%s", day,month,hour,minute,second);
	//	fclose(file);

	//	ea = 0;
	//	t_ea += kernel_errors;
 
	//	/////////////UPDATE TIMESTAMP///////////////////
	//	UpdateTimestamp(); // UNCOMENT THIS AFTER
		////////////////////////////////////////////////
		
		//log_error_count(kernel_errors);

		if (kernel_errors!=0)
		{
			//file = fopen(file_name, "a");
			char error_detail[150];

			printf("\n kernel error: %d\n", kernel_errors);

			malloc_mem1 = cudaMemcpy(h_A, d_C, size * sizeof( double ), cudaMemcpyDeviceToHost);
			erro_malloc = cudaGetErrorString(malloc_mem1);
			if(strcmp(erro_malloc, "no error") != 0)
				{printf("error mem load MEMDUMP %s", erro_malloc); fprintf(file, "error mem load MEMDUMP %s", erro_malloc); return 1;}

			for(i=0; (i<k) && (ea < N_ERRORS_LOG); i++)
			{
				for(j=0; (j<k) && (ea < N_ERRORS_LOG); j++)
				{
					if ((fabs((h_A[i+k*j]-h_GOLD[i+k*j])/h_A[i+k*j]) > 0.0000000001)||(fabs((h_A[i+k*j]-h_GOLD[i+k*j])/h_GOLD[i+k*j]) > 0.0000000001))
					{
						snprintf(error_detail, 150, "p: [%d, %d], r: %1.16e, e: %1.16e", i, j, h_A[i + k * j], h_GOLD[i + k * j]);
						log_error_detail(error_detail);
						//ea++;
						//fprintf(file, "\n p: [%d, %d], r: %1.16e, e: %1.16e, error: %d\n", i, j, h_A[i + k * j], h_GOLD[i + k * j], ea);
										
					}
				}
			}

			///////////UPDATE LOG FILE//////////////////////
			//log_file = fopen(file_name_log, "a");
			//fprintf(log_file, "\ntest number: %d", loop2);
			//fprintf(log_file, "\ntime: %f", timeG);
			//fprintf(log_file, "\ntotal time: %f", total_time);
			//fprintf(log_file, "\nerrors: %d", kernel_errors);
			//fprintf(log_file, "\ntotal errors: %d", t_ea);
			//fclose(log_file);
			//fclose(file);

			ReadMatrixFromFile();
		}



		if(kernel_errors > 0 || (loop2 % 10 == 0))
		{
			printf("\ntest number: %d", loop2);
			//printf("\ntotal time: %f", total_time);
			printf("\nerrors: %d", kernel_errors);
			//printf("\ntotal errors: %d\n", t_ea);
		//	if((kernel_errors != 0) && (kernel_errors == old_ea))
		//		{
		//			old_ea = 0;
		//			return 1;
		//		}
		//		
		//		old_ea = kernel_errors;
		}
		else
		{
			printf(".");
		}

		cudaFree( d_A );
		cudaFree( d_B );
		cudaFree( d_C );
	}

	free(h_A);
	free(h_B);
	free(h_GOLD);

	end_log_file();

	return 0;
}
