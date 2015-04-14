#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "./log_helper.h"

#include "cuda.h"
//#include "cublas.h"
#include "cublas_v2.h"

#define N 1024 
#define GOLD_MATRIX_PATH "/home/carol/TestGPU/GenerateGoldMatrix/Double_GOLD_1024.matrix"
#define LOG_NAME "cudaGEMM1024"
#define SWITCH_CHAR  '-'

#define BLOCK_SIZE 32
#define ITERACTIONS 1


#undef min
#define min( x, y ) ( (x) < (y) ? (x) : (y) )
#undef max
#define max( x, y ) ( (x) > (y) ? (x) : (y) )

// ----------------------------------- TO DO
// Modificar o make para que este arquivo seja compilado com nvcc. Ou entender porque o nvcc nao reconhece o kernel ou as macros threadIdx ou blockDim,=.

double *A;
double *B;
double *d_A;
double *d_B;
double *d_C;

   int lda, ldb, ldc;

double *GOLD;


FILE* f_A;
FILE* f_B;
FILE* f_GOLD;

FILE* file;
FILE* log_file;
FILE* timefile;

void UpdateTimestamp(){
	time_t timestamp = time(NULL);
	char time_s[50];
	sprintf(time_s, "%d", int(timestamp));

	char string[100] = "echo ";
	strcat(string, time_s);
	strcat(string, " > /home/carol/TestGPU/timestamp.txt");
	system(string);

//	printf("\n%s\n", string);
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
	printf("\ndevice: %d %s", *ndevice, prop.name);

}

double mysecond()
{
   struct timeval tp;
   struct timezone tzp;
   int i = gettimeofday(&tp,&tzp);
   return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void ReadMatrixFromFile(){	
	
	int i;
	int j;
	double temp=0;
	double time = mysecond();
printf("open...");
	f_A = fopen("/home/carol/TestGPU/GenerateGoldMatrix/Double_A_8192.matrix","rb");
	f_B = fopen("/home/carol/TestGPU/GenerateGoldMatrix/Double_B_8192.matrix","rb");
	f_GOLD = fopen(GOLD_MATRIX_PATH,"rb");
printf("read...");
	for(i=0; i<N; i++)
	{
		fread (&A[ lda * i ], sizeof(double)*N, 1, f_A);
		fread (&B[ lda * i ], sizeof(double)*N, 1, f_B);
		fread (&GOLD[ lda * i ], sizeof(double)*N, 1, f_GOLD);
		//for(j=0; j<N; j++){
//
//			A[i + lda * j] = 0.0;
//			B[j + ldb * i] = 0.0;
//
//			GOLD[i + ldc * j] = 0.0; 
//
//			fread(&A[i + lda * j],sizeof(double), 1, f_A);
//			fread(&B[j + ldb * i],sizeof(double), 1, f_B);
//			fread(&GOLD[i + ldc * j],sizeof(double), 1, f_GOLD);
//			
//		}
	}	printf("\n");
	for (i=0; i<N; i++)
	{ 
		for (j=0; (j<N)&&(j<i); j++)
		{
			temp = GOLD [i + ldc * j];
			GOLD [i + ldc * j] = GOLD [j + ldc * i];
			GOLD [j + ldc * i] = temp;
		}
	}
printf("ok in %f\n", mysecond() - time);

//A[45] = 5.5;
	fclose(f_A);
	fclose(f_B);
	fclose(f_GOLD);
}

__device__ int kerrors;

__global__ void GoldChkKernel (double *gk, double *ck, int n)//, int *kerrors)
{
	//ck[4] = 4.5;
	int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                                                      
	int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y; 
	if ((fabs((gk[ty*n+tx]-ck[ty*n+tx])/gk[ty*n+tx]) > 0.0000000001)||(fabs((gk[ty*n+tx]-ck[ty*n+tx])/ck[ty*n+tx]) > 0.0000000001))
//	if (gk[ty*n + tx]!=ck[ty*n + tx])
		atomicAdd(&kerrors, 1);//kerrors++;

}



int main( int argc, char* argv[] )
{

	
	cudaError_t malloc_mem1;
	cudaError_t malloc_a;
	const char *erro_malloc;

	int m = N;
	int n = N;
	int k = N;

	int ea=0; //wrong integers in the current loop
	int t_ea=0; //total number of wrong integers
	//int old_ea = 0;

	//double tmp_time = 0.0;
	double total_time = 0.0;

	int j;

	const double alpha = 1.0;
	const double beta = 1.0;

	int repeats = 5;

	char transa = 't', transb = 't';
	int i, loop2;

	//int kerrors = 0;
	int kernel_errors=0;
	int zero = 0;


	int sizea, sizeb, sizec;
	double timeG;//, gflops;

	//////////BLOCK and GRID size///////////////////////
	int gridsize = N/BLOCK_SIZE < 1 ? 1 : N/BLOCK_SIZE;
	int blocksize = N/BLOCK_SIZE < 1 ? N : BLOCK_SIZE;
	dim3 dimBlock(blocksize,blocksize);
	dim3 dimGrid(gridsize,gridsize);
	////////////////////////////////////////////////////


	///////////////////////////////////////////////////////
	////////////////FILE NAME//////////////////////////////
//	time_t file_time;
//	struct tm *ptm;
//	char day[2], month[2], year[4], hour[2], second[2], minute[2];
//	char file_name[60];
//	char file_name_log[60];
//	
//	file_time = time(NULL);
//	ptm = gmtime(&file_time);
//
//	snprintf(day, sizeof(day + 1), "%d", ptm->tm_mday);
//	snprintf(month, sizeof(month + 1), "%d", ptm->tm_mon+1);
//	snprintf(year, sizeof(year + 1), "%d", ptm->tm_year+1900);
//	snprintf(hour, sizeof(hour + 1), "%d", ptm->tm_hour);
//	snprintf(minute, sizeof(minute + 1), "%d", ptm->tm_min);
//	snprintf(second, sizeof(second + 1), "%d", ptm->tm_sec);
//	strcpy(file_name,day);strcat(file_name,"_");
//	strcat(file_name,month);strcat(file_name,"_");
//	strcat(file_name,year);strcat(file_name,"_");
//	strcat(file_name,hour);strcat(file_name,"_");
//	strcat(file_name,minute);strcat(file_name,"_");
//	strcat(file_name,second);strcat(file_name,"_");
//	strcat(file_name,LOG_NAME);
//	strcpy(file_name_log, file_name);
//	
//	strcat(file_name,".txt");
//	strcat(file_name_log,"log.txt");
	///////////////////////////////////////////////////////
	///////////////////////////////////////////////////////
	char test_info[90];
	snprintf(test_info, 90, "size:%d", N);
	start_log_file("my_benchmark", test_info);

	if ( m < 0 || n < 0 || k < 0 )
	{
		printf( "Error: Problem size must be nonnegative.\n" );
		return -1;
	}

	if ( repeats < 0 )
	{
		printf( "Error: Number of repeats must be nonnegative.\n" );
		return -1;
	}   

	printf( "transa = %c, transb = %c\n", transa, transb );
	printf( "m = %d, n = %d, k = %d\n", m, n, k );
	printf( "alpha = %g, beta = %g\n", alpha, beta );
	printf( "number of repeats = %d\n", repeats );

	lda = max( 1, m + 16 );
	sizea = lda * k;
	ldb = max( 1, n + 16 );
	sizeb = ldb * k;
	ldc = max( 1, m + 16 );
	sizec = ldc * n;

	A = ( double* ) malloc( sizea * sizeof( double ) );
	B = ( double* ) malloc( sizeb * sizeof( double ) );

	GOLD = ( double* ) malloc( sizec * sizeof( double ) );

	kernel_errors=0;
	
//	cudaSetDevice( 0 );
	GetDevice();
	
	ReadMatrixFromFile();

	//A[72] = 7.2;

	printf( "cublasDGEMM\n" );

   
	for(loop2=0; loop2<ITERACTIONS; loop2++)
	{


	//cudaMalloc((void*) &kerrors, sizeof(int));

	malloc_a = cudaMalloc( ( void** ) &d_A, sizea * sizeof( double ) );
	erro_malloc = cudaGetErrorString(malloc_a);
	if(strcmp(erro_malloc, "no error") != 0) {log_error_detail("error a"); end_log_file(); return 1;} //mem allocate failure

	malloc_a = cudaMalloc( ( void** ) &d_B, sizea * sizeof( double ) );
	erro_malloc = cudaGetErrorString(malloc_a);
	if(strcmp(erro_malloc, "no error") != 0) {log_error_detail("error a"); end_log_file(); return 1;} //mem allocate failure

	malloc_a = cudaMalloc( ( void** ) &d_C, sizea * sizeof( double ) );
	erro_malloc = cudaGetErrorString(malloc_a);
	if(strcmp(erro_malloc, "no error") != 0) {log_error_detail("error a"); end_log_file(); return 1;} //mem allocate failure


	malloc_mem1 = cudaMemcpy( d_C, A, sizeb * sizeof( double ), cudaMemcpyHostToDevice ); // ZERA C
	erro_malloc = cudaGetErrorString(malloc_mem1);
	if(strcmp(erro_malloc, "no error") != 0) {log_error_detail("error mem load a "); end_log_file(); return 1;}
	
	malloc_mem1 = cudaMemcpy( d_A, A, sizeb * sizeof( double ), cudaMemcpyHostToDevice ); // PUSH A
	erro_malloc = cudaGetErrorString(malloc_mem1);
	if(strcmp(erro_malloc, "no error") != 0) {log_error_detail("error mem load a "); end_log_file(); return 1;}

	malloc_mem1 = cudaMemcpy( d_B, B, sizeb * sizeof( double ), cudaMemcpyHostToDevice ); // PUSH B
	erro_malloc = cudaGetErrorString(malloc_mem1);
	if(strcmp(erro_malloc, "no error") != 0) {log_error_detail("error mem load a "); end_log_file(); return 1;}

	kernel_errors=0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	

	//timeG = mysecond();
	start_iteration();
	cublasDgemm( handle, (cublasOperation_t)transa, (cublasOperation_t)transb,
		   m, n, k,
		   &alpha,
		   d_A, lda,
		   d_B, ldb,
		   &beta,
		   d_C, ldc );

	cudaDeviceSynchronize();
	end_iteration();
	//timeG = mysecond() - timeG;

	//total_time += timeG;


	//gflops = 2.0 * m * n * k / ( timeG ) / 1e9; 
	//printf( "Time: %f | GFLOPS: %6.3f\n", timeG, gflops );


	malloc_mem1 = cudaMemcpy(d_A, GOLD, sizea * sizeof( double ), cudaMemcpyHostToDevice );
	erro_malloc = cudaGetErrorString(malloc_mem1);
	if(strcmp(erro_malloc, "no error") != 0) {printf("error mem load a %s", erro_malloc); fprintf(file, "error mem load a %s", erro_malloc); return 1;}

	//cudaMemcpy(&kerrors, &kernel_errors, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(kerrors, &zero, sizeof(int));

	GoldChkKernel<<<dimGrid,dimBlock>>>(d_A, d_C, ldc);


	cudaMemcpyFromSymbol(&kernel_errors, kerrors, sizeof(unsigned int));
	

	///////////UPDATE FILE//////////////////////
//	file_time = time(NULL);
//	ptm = gmtime(&file_time);
//	snprintf(hour, sizeof(hour + 1), "%d", ptm->tm_hour);
//	snprintf(minute, sizeof(minute + 1), "%d", ptm->tm_min);
//	snprintf(second, sizeof(second + 1), "%d", ptm->tm_sec);
//	fprintf(file, "\n start time: %s/%s_%s:%s:%s", day,month,hour,minute,second);
//	fclose(file);

//	ea = 0;
//	t_ea += kernel_errors;
 
	/////////////UPDATE TIMESTAMP///////////////////
//	UpdateTimestamp();
	////////////////////////////////////////////////

	log_error_count(kernel_errors);

	if (kernel_errors!=0)
	{

	//file = fopen(file_name, "a");

		printf("\n kernel error: %d\n", kernel_errors);

		malloc_mem1 = cudaMemcpy(A, d_C, sizec * sizeof( double ), cudaMemcpyDeviceToHost);
		erro_malloc = cudaGetErrorString(malloc_mem1);
		if(strcmp(erro_malloc, "no error") != 0)
			{printf("error mem load a %s", erro_malloc); fprintf(file, "error mem load a %s", erro_malloc); return 1;}
		char error_detail[150];

		for(i=0; (i<N) && (ea < 500); i++)
		{
			for(j=0; (j<N) && (ea < 500); j++)
			{
				if ((fabs((A[i+ldc*j]-GOLD[i+ldc*j])/A[i+ldc*j]) > 0.0000000001)||(fabs((A[i+ldc*j]-GOLD[i+ldc*j])/GOLD[i+ldc*j]) > 0.0000000001))
				{
					snprintf(error_detail, 150, "p: [%d, %d], r: %1.16e, e: %1.16e", i, j, A[i + ldc * j], GOLD[i + ldc * j]);
					log_error_detail(error_detail);
					//ea++;			
					//fprintf(file, "\n p: [%d, %d], r: %1.16e, e: %1.16e, error: %d\n", i, j, A[i + ldc * j], GOLD[i + ldc * j], t_ea);
										
				}
			}
		}

		//	///////////UPDATE LOG FILE//////////////////////
		//	log_file = fopen(file_name_log, "a");
		//	fprintf(log_file, "\ntest number: %d", loop2);
		//	fprintf(log_file, "\ntime: %f", timeG);
		//	fprintf(log_file, "\ntotal time: %f", total_time);
		//	fprintf(log_file, "\nerrors: %d", kernel_errors);
		//	fprintf(log_file, "\ntotal errors: %d", t_ea);
		//	fclose(log_file);
		//	fclose(file);

			ReadMatrixFromFile();	
	}



	if(kernel_errors > 0 || (loop2 % 10 == 0))
	{
		printf("\ntest number: %d", loop2);
		printf("\ntest time: %f", timeG);
		printf("\ntotal time: %f", total_time);
		printf("\nerrors: %d", kernel_errors);
		printf("\ntotal errors: %d", t_ea);
	//	if((kernel_errors != 0) && (kernel_errors == old_ea))
	//		{
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
	//cudaFree( kerrors );
	}

	free( A );
	free( B );


	return 0;
}
