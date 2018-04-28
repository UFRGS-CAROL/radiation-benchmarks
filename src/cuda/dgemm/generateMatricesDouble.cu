#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <unistd.h>
#include <string>
#include <sys/time.h>
#include <float.h>

#include <cublas.h>

// helper functions
#include "helper_string.h"
#include "helper_cuda.h"

#define DEFAULT_INPUT_SIZE 8192

int k=0;
int size;
double *A, *B, *GOLD;

bool host_check = false;
bool generator_debug = false;

char *gold_matrix_path, *a_matrix_path, *b_matrix_path;

void usage() {
    printf("Usage: generateMatricesSingle -size=N [-generator_debug] [-host_check] [-input_a=<path>] [-input_b=<path>] [-gold=<path>]\n");
}

void generateInputMatrices()
{
	double *h_A, *h_B;
	FILE *f_A, *f_B;

    h_A = (double*)malloc(sizeof(double) * DEFAULT_INPUT_SIZE*DEFAULT_INPUT_SIZE);
    h_B = (double*)malloc(sizeof(double) * DEFAULT_INPUT_SIZE*DEFAULT_INPUT_SIZE);
    printf("Max value: %f Min: %f\n", (-4.06e16-4.0004e16)+4.1e16, 4.1e16);

	srand(time(NULL));

	if (!generator_debug) {
		for (int i=0; i<DEFAULT_INPUT_SIZE; i++) {
			for (int j=0; j<DEFAULT_INPUT_SIZE; j++) {
				h_A[i * DEFAULT_INPUT_SIZE + j] = (rand()/((double)(RAND_MAX)+1)*(-4.06e16-4.4e16))+4.1e16;
	
				h_B[i * DEFAULT_INPUT_SIZE + j] = (rand()/((double)(RAND_MAX)+1)*(-4.06e16-4.4e16))+4.1e16;
			}
		}
	} else {
		for (int i=0; i<DEFAULT_INPUT_SIZE; i++) {
			for (int j=0; j<DEFAULT_INPUT_SIZE; j++) {
				h_A[i * DEFAULT_INPUT_SIZE + j] = double(2.0);
				h_B[i * DEFAULT_INPUT_SIZE + j] = double(2.0);
			}
		}
	}

	int numZeros;
    int numNans;
    int numInfs;
// printf("Write\n");
	f_A = fopen(a_matrix_path, "wb");
	f_B = fopen(b_matrix_path, "wb");

    double val;

	numZeros = 0;
    numNans = 0;
    numInfs = 0;
	for (int i = 0; i<DEFAULT_INPUT_SIZE*DEFAULT_INPUT_SIZE; i++) {
        val=h_A[i];
		if (val == 0) numZeros++;
        if (isnan(val)) numNans++;
        if (isinf(val)) numInfs++;
	}
	printf("Number of zeros/NaNs/INFs on matrix A: %d/%d/%d\n", numZeros, numNans, numInfs);

	numZeros = 0;
    numNans = 0;
    numInfs = 0;
	for (int i = 0; i<DEFAULT_INPUT_SIZE*DEFAULT_INPUT_SIZE; i++) {
        val=h_B[i];
		if (val == 0) numZeros++;
        if (isnan(val)) numNans++;
        if (isinf(val)) numInfs++;
	}
	printf("Number of zeros/NaNs/INFs on matrix B: %d/%d/%d\n", numZeros, numNans, numInfs);

	for(int i=0; i<DEFAULT_INPUT_SIZE; i++)
	{
		fwrite(&(h_A[i * DEFAULT_INPUT_SIZE]), sizeof(double) * DEFAULT_INPUT_SIZE, 1, f_A);
	}

	printf("Element 32 of matrix A: %f\n", (double)h_A[32]);

	printf("Element 50 of matrix B: %f\n", (double)h_B[50]);


	for(int i=0; i<DEFAULT_INPUT_SIZE; i++)
	{
		fwrite(&(h_B[i * DEFAULT_INPUT_SIZE]), sizeof(double) * DEFAULT_INPUT_SIZE, 1, f_B);
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
		ret_value[0] = fread (&A[ k * i ], sizeof(double)*k, 1, f_A);
		ret_value[1] = fread (&B[ k * i ], sizeof(double)*k, 1, f_B);
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
	for (int i=0;i<size;i++) {
		for (int j=0;j<size;j++) {
			for (int k=0;k<size;k++)  {
				c[j*size+i] += a[j*size+k] * bT[i*size+k];
			}
		}
	}

	printf("host mmul time: %.2f seconds\n", mysecond()-time);

	return c;
}

void generateGoldMatrixHalf()
{
	////////////////////////////////////////////////////
	/////////////CUBLAS GEMM VARS///////////////////////
	const double alpha = 1.0;
	const double beta = 1.0;
	char transa = 't', transb = 't';
	////////////////////////////////////////////////////

	////////////////////////////////////////////////////
	//////////DEVICE VARS///////////////////////////////
	double *d_A;
	double *d_B;
	double *d_C;
	////////////////////////////////////////////////////

	A = ( double* ) malloc( size * sizeof( double ) );
	B = ( double* ) malloc( size * sizeof( double ) );
	GOLD = ( double* ) malloc( size * sizeof( double ) );

	ReadMatrixFromFile();
	if (k <= 16) {
		printf("\nMatrix A: \n");
		for (int i = 0; i<k*k; i++) {
			printf(" %.2e", (double)A[i]);
			if ((i+1)%k == 0) printf("\n");
		}
		printf("\nMatrix B: \n");
		for (int i = 0; i<k*k; i++) {
			printf(" %.2e", (double)B[i]);
			if ((i+1)%k == 0) printf("\n");
		}
	}

	checkCudaErrors( cudaMalloc( ( void** ) &d_A, size * sizeof( double ) ));

	checkCudaErrors( cudaMalloc( ( void** ) &d_B, size * sizeof( double ) ));

	checkCudaErrors( cudaMalloc( ( void** ) &d_C, size * sizeof( double ) ));


	checkCudaErrors( cudaMemset( d_C, 0, size * sizeof( double )) ); // ZERA C

	checkCudaErrors( cudaMemcpy( d_A, A, size * sizeof( double ), cudaMemcpyHostToDevice ) ); // PUSH A

	checkCudaErrors( cudaMemcpy( d_B, B, size * sizeof( double ), cudaMemcpyHostToDevice ) ); // PUSH B

	printf("cudaDGEMM... k=%d\n", k);
	double time = mysecond();
	
	cublasDgemm( (cublasOperation_t)transa, (cublasOperation_t)transb,
			   k, k, k,
			   alpha,
			   d_A, k,
			   d_B, k,
			   beta,
			   d_C, k );

	checkCudaErrors( cudaPeekAtLastError() );
	
	checkCudaErrors( cudaDeviceSynchronize() );
	checkCudaErrors( cudaPeekAtLastError() );

	time=mysecond()-time;

	/////////// PERF
    double flops = 2.0*(double)k*k*k;
    double gflops = flops / time;
    double outputpersec = (double)k*k/time;
    printf("kernel time: %lf\n",time);
    printf("SIZE:%d OUTPUT/S:%f FLOPS:%f (GFLOPS:%.2f)\n",k, outputpersec, gflops, gflops/1000000000);
	///////////

	checkCudaErrors( cudaMemcpy(GOLD, d_C, size * sizeof( double ), cudaMemcpyDeviceToHost) );

	cudaFree( d_A );
	cudaFree( d_B );
	cudaFree( d_C );

	printf("Analysing output on host...\n");

	int i, j;
	FILE *f_GOLD;

	f_GOLD = fopen(gold_matrix_path, "wb");

    double val;

	int numZeros = 0;
    int numNans = 0;
	int numInfs = 0;
	double maxAbsVal = 0.0;
	#pragma omp parallel for
	for (int i = 0; i<k*k; i++) {
		val=GOLD[i];
		if (fabs(val) > maxAbsVal) {
			#pragma omp critical
			maxAbsVal = max(fabs(val), maxAbsVal);
		}
		if (val == 0) {
			#pragma omp atomic
			numZeros++;
			if (numZeros<5) printf("Zero in position (%d,%d)\n", (int)floor(i / k), (int)(i - floor(i / k) * k));
		}
        if (isnan(val)) {
			#pragma omp atomic
			numNans++;
			if (numNans<5) printf("NaN in position (%d,%d)\n", (int)floor(i / k), (int)(i - floor(i / k) * k));
		}
        if (isinf(val))  {
			#pragma omp atomic
			numInfs++;
			if (numInfs<5) printf("INF in position (%d,%d)\n", (int)floor(i / k), (int)(i - floor(i / k) * k));
		}
	}
	printf("Number of zeros/NaNs/INFs on gold: %d/%d/%d\n", numZeros, numNans, numInfs);
	printf("Maximum absolute value on gold: %f\n", maxAbsVal);

	if (k <= 16) {
		for (int i = 0; i<k*k; i++) {
			printf(" %.2e", (double)GOLD[i]);
			if ((i+1)%k == 0) printf("\n");
		}
	}

	if (host_check) {
		printf("Calculating mMul using OpenMP on Host...\n");
		double *hostGold = openmpMul(A, B, k);
		if (k <= 16) {
			printf("Host CPU Gold:\n");
			for (int i = 0; i<k*k; i++) {
				printf(" %.2e", (double)hostGold[i]);
				if ((i+1)%k == 0) printf("\n");
			}
		}
		printf("Comparing GPU result with Host result...\n");
		double maxDiff = 0.0;
		double maxAbsDiff = 0.0;
		for (i=0; i<k; i++) {
			for (j=0; j<k; j++) {
				register double diff = fabs((hostGold[i*k+j]-GOLD[i*k+j])/hostGold[i*k+j]);
				register double absDiff = hostGold[i*k+j]-GOLD[i*k+j];
				if (diff > maxDiff) {
					maxDiff = max(diff, maxDiff);
					printf("New diff! (%d,%d) hostGold!=gpuGold %e != %e (diff: %e)\n", i, j, hostGold[i*k+j], GOLD[i*k+j], diff);
				}
				if (absDiff > maxAbsDiff) {
					maxAbsDiff = max(absDiff, maxAbsDiff);
				}
				// if (diff > 0.1) {
				// 	printf("Fail! (%d,%d) hostGold!=gpuGold %f != %f (diff: %e)\n", i, j, (double)hostGold[i*k+j], (double)GOLD[i*k+j], diff);
				// 	fflush(stdout);
				// 	exit(-1);
				// }
			}
		}
		printf("CPU and GPU match by a relative error of up to %e element difference.\nMaximum element absolute difference: %e (relatively to double representation: %e)\nWriting to file...\n", 
		maxDiff, maxAbsDiff, maxAbsDiff / DBL_MAX);
	}

	//printf("-------------------------\n%.10f\n%.10f\n%.10f\n", GOLD[0], GOLD[1], GOLD[2]);

	for(i=0; i<k; i++)
	{
		fwrite( &(GOLD[i * k]), sizeof(double)*k, 1, f_GOLD );
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
        snprintf(a_matrix_path, 100, "dgemm_a_%i.matrix", (signed int)DEFAULT_INPUT_SIZE);
        printf("Using default input_a path: %s\n", a_matrix_path);
    }

	if (checkCmdLineFlag(argc, (const char **)argv, "input_b"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "input_b", &b_matrix_path);
    }
    else
    {
        b_matrix_path = new char[100];
        snprintf(b_matrix_path, 100, "dgemm_b_%i.matrix", (signed int)DEFAULT_INPUT_SIZE);
        printf("Using default input_a path: %s\n", b_matrix_path);
    }

	if (checkCmdLineFlag(argc, (const char **)argv, "gold"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "gold", &gold_matrix_path);
    }
    else
    {
        gold_matrix_path = new char[100];
        snprintf(gold_matrix_path, 100, "dgemm_gold_%i.matrix", (signed int)k);
        printf("Using default gold path: %s\n", gold_matrix_path);
    }

	if (checkCmdLineFlag(argc, (const char **)argv, "host_check"))
    {
		host_check = true;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "generator_debug"))
    {
		generator_debug = true;
	}
//====================================

	GetDevice();

	size = k * k;

    printf("Each input matrix size: %.4fGB\n", (double)sizeof(double) * DEFAULT_INPUT_SIZE*DEFAULT_INPUT_SIZE / (1024*1024*1024));

	FILE *test_file;
	test_file=fopen(a_matrix_path, "rb");
	if (!test_file)
	{
		printf("Generating input matrices...\n");
		generateInputMatrices();
	}
	else
	{	printf("Input matrices already exist...\n");	}

	generateGoldMatrixHalf();

	return 0;
}