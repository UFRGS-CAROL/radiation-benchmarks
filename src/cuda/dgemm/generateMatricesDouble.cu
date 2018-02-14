#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <unistd.h>
#include <string>
#include <sys/time.h>
// helper functions
#include "helper_string.h"
#include "helper_cuda.h"

#include<cublas.h>

#define MATRIX_PATH "./Double_"
#define DEFAULT_INPUT_SIZE 8192

int k=0;
int sizea, sizeb, sizec;
double *A, *B, *GOLD;

bool host_check = false;

char *gold_matrix_path, *a_matrix_path, *b_matrix_path;

void usage() {
    printf("Usage: generateMatrices -size=N [-host_check] [-input_a=<path>] [-input_b=<path>] [-gold=<path>]\n");
}

void generateInputMatrices()
{
	double temp;
	int i, j;
	FILE *f_A, *f_B;

	f_A = fopen(a_matrix_path, "wb");
	f_B = fopen(b_matrix_path, "wb");


	srand ( time(NULL) );

    int numZerosA = 0;
    int numZerosB = 0;
    #pragma omp parallel for
	for(i=0; i<DEFAULT_INPUT_SIZE; i++)
	{
		for(j=0; j<DEFAULT_INPUT_SIZE+16; j++){
			temp = (rand()/((double)(RAND_MAX)+1)*(-4.06e16-4.0004e16))+4.1e16;
			fwrite( &temp, sizeof(double), 1, f_A );
            if (temp == 0 || isnan(temp) || isinf(temp)) {
                numZerosA++;
            }

			temp = (rand()/((double)(RAND_MAX)+1)*(-4.06e16-4.4e16))+4.1e16;
			fwrite( &temp, sizeof(double), 1, f_B );
            if (temp == 0 || isnan(temp) || isinf(temp)) {
                numZerosB++;
            }


		}
	}
	printf("Number of zeros/NaNs/INFs on A: %d\n", numZerosA);
	printf("Number of zeros/NaNs/INFs on B: %d\n", numZerosB);

	fclose(f_A);
	fclose(f_B);

	return;
}

void ReadMatrixFromFile(){

	int i;
	FILE *f_A, *f_B;
    printf("Each matrix size: %.4fGB\n", (float)sizeof(double) * DEFAULT_INPUT_SIZE*DEFAULT_INPUT_SIZE / (1024*1024*1024));

	f_A = fopen(a_matrix_path,"rb");
	f_B = fopen(b_matrix_path,"rb");
	if (!(f_A&&f_B))
	{
		printf("Error opening matrices A, B.\n");
		exit(-1);
	}
	for(i=0; i<k; i++)
	{
		fread (&A[ k * i ], sizeof(double)*k, 1, f_A);
		fread (&B[ k * i ], sizeof(double)*k, 1, f_B);
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

double* openmpMul(double* a, double* b, size_t size) {
	double time = mysecond();

	double* bT = (double*) malloc(sizeof(double)*size*size);
	double* c = (double*) calloc(size*size, sizeof(double));

	if (c == NULL || bT == NULL) {
		printf("could not alloc hostGold matrix.");
		return NULL;
	}

	#pragma omp parallel for
	for (int i=0;i<size;i++)
		for (int j=0;j<size;j++)
			bT[j*size+i] = b[i*size+j];
	
	#pragma omp parallel for
	for (int i=0;i<size;i++)
		for (int j=0;j<size;j++)
			for (int k=0; k<size;k++) 
				c[i*size+j] += a[j*size+k] * bT[i*size+k];

	printf("host mmul time: %.2f seconds\n", mysecond()-time);

	return c;
}

void generateGoldMatrix()
{
	////////////////////////////////////////////////////
	/////////////CUBLAS GEMM VARS///////////////////////
	const double alpha = 1.0;
	const double beta = 1.0;
	char transa = 't', transb = 't';
	////////////////////////////////////////////////////

	////////////////////////////////////////////////////
	//////////DEVICE VARS///////////////////////////////
	cudaError_t cumalloc_err;
	const char *cumalloc_err_str;

	double *d_A;
	double *d_B;
	double *d_C;
	////////////////////////////////////////////////////

	A = ( double* ) malloc( sizea * sizeof( double ) );
	B = ( double* ) malloc( sizeb * sizeof( double ) );
	GOLD = ( double* ) malloc( sizec * sizeof( double ) );

	GetDevice();

	ReadMatrixFromFile();

	cumalloc_err = cudaMalloc( ( void** ) &d_A, sizea * sizeof( double ) );
	cumalloc_err_str = cudaGetErrorString(cumalloc_err);
	if(strcmp(cumalloc_err_str, "no error") != 0) {exit(-3);} //mem allocate failure

	cumalloc_err = cudaMalloc( ( void** ) &d_B, sizea * sizeof( double ) );
	cumalloc_err_str = cudaGetErrorString(cumalloc_err);
	if(strcmp(cumalloc_err_str, "no error") != 0) {exit(-3);}

	cumalloc_err = cudaMalloc( ( void** ) &d_C, sizea * sizeof( double ) );
	cumalloc_err_str = cudaGetErrorString(cumalloc_err);
	if(strcmp(cumalloc_err_str, "no error") != 0) {exit(-3);}


	cumalloc_err = cudaMemset( d_C, 0, sizeb * sizeof( double )); // ZERA C
	cumalloc_err_str = cudaGetErrorString(cumalloc_err);
	if(strcmp(cumalloc_err_str, "no error") != 0) {exit(-3);}

	cumalloc_err = cudaMemcpy( d_A, A, sizeb * sizeof( double ), cudaMemcpyHostToDevice ); // PUSH A
	cumalloc_err_str = cudaGetErrorString(cumalloc_err);
	if(strcmp(cumalloc_err_str, "no error") != 0) {exit(-3);}

	cumalloc_err = cudaMemcpy( d_B, B, sizeb * sizeof( double ), cudaMemcpyHostToDevice ); // PUSH B
	cumalloc_err_str = cudaGetErrorString(cumalloc_err);
	if(strcmp(cumalloc_err_str, "no error") != 0) {exit(-3);}

	printf("cublasDgemm... k=%d transa=%c transb=%c\n", k, transa, transb);
	double time = mysecond();


	cublasDgemm( (cublasOperation_t)transa, (cublasOperation_t)transb,
			   k, k, k,
			   alpha,
			   d_A, k,
			   d_B, k,
			   beta,
			   d_C, k );
	cudaDeviceSynchronize();

	time=mysecond()-time;

	/////////// PERF
    double flops = 2.0*(double)k*k*k;
    double gflops = flops / time;
    double outputpersec = (double)k*k/time;
    printf("kernel time: %lf\n",time);
    printf("SIZE:%d OUTPUT/S:%f FLOPS:%f (GFLOPS:%.2f)\n",k, outputpersec, gflops, gflops/1000000000);
	///////////

	cumalloc_err = cudaMemcpy(GOLD, d_C, sizec * sizeof( double ), cudaMemcpyDeviceToHost);
	cumalloc_err_str = cudaGetErrorString(cumalloc_err);
	if(strcmp(cumalloc_err_str, "no error") != 0) {exit(-3);}

	cudaFree( d_A );
	cudaFree( d_B );
	cudaFree( d_C );

	int i, j;
	FILE *f_GOLD;

	f_GOLD = fopen(gold_matrix_path, "wb");

	//printf("-------------------------\n%.10f\n%.10f\n%.10f\n", GOLD[0], GOLD[1], GOLD[2]);

	if (host_check) {
		printf("Calculating mMul using OpenMP on Host...\n");
		double *hostGold = openmpMul(A, B, k);
		printf("Comparing GPU result with Host result...\n");
		double maxDiff = 0.0;
		#pragma omp parallel for
		for (i=0; i<k; i++) {
			for (j=0; j<k; j++) {
				register double diff = fabs((hostGold[i*k+j]-GOLD[i*k+j])/hostGold[i*k+j]);
				if (diff > maxDiff) {
					#pragma omp critical
					maxDiff = max(diff, maxDiff);
				}
				if (diff > 0.1) {
					printf("Fail! hostGold!=gpuGold %f != %f (diff: %e)\n", hostGold[i*k+j], GOLD[i*k+j], fabs((hostGold[i*k+j]-GOLD[i*k+j])/hostGold[i*k+j]));
					fflush(stdout);
					exit(-1);
				}
			}
		}
		printf("CPU and GPU match by an error of up to %e element difference. Writing to file...\n", maxDiff);
	}

    int numZeros = 0;
	for(i=0; i<k; i++)
	{
		fwrite( &GOLD[i * k], sizeof(double)*k, 1, f_GOLD );
        for(j=0; j<k; j++) {
            if (isnan(GOLD[i*k + j]) || isinf(GOLD[i*k + j]) || GOLD[i*k + j]==0) {
                numZeros++;
            }
        }
	}
	printf("Number of zeros/NaNs/INFs on GOLD: %d\n", numZeros);

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
        snprintf(a_matrix_path, 100, "dgemm_a_%i", (signed int)DEFAULT_INPUT_SIZE);
        printf("Using default input_a path: %s\n", a_matrix_path);
    }

	if (checkCmdLineFlag(argc, (const char **)argv, "input_b"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "input_b", &b_matrix_path);
    }
    else
    {
        b_matrix_path = new char[100];
        snprintf(b_matrix_path, 100, "dgemm_b_%i", (signed int)DEFAULT_INPUT_SIZE);
        printf("Using default input_a path: %s\n", b_matrix_path);
    }

	if (checkCmdLineFlag(argc, (const char **)argv, "gold"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "gold", &gold_matrix_path);
    }
    else
    {
        gold_matrix_path = new char[100];
        snprintf(gold_matrix_path, 100, "dgemm_gold_%i", (signed int)k);
        printf("Using default gold path: %s\n", gold_matrix_path);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "host_check"))
    {
		host_check = true;
	}
//====================================

	sizea = k * k;
	sizeb = k * k;
	sizec = k * k;


    printf("Each matrix size: %.4fGB\n", (float)sizeof(double) * DEFAULT_INPUT_SIZE*DEFAULT_INPUT_SIZE / (1024*1024*1024));

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
