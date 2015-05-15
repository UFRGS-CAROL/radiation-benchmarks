#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <unistd.h>
#include <string>
#include <sys/time.h>

#define MATRIX_PATH "./Double_"
#define INMATRIXSIZE 8192

#define BLOCK_SIZE 32
using namespace std;

int k=0;
int size;	
double *A, *B, *GOLD;

string gold_matrix_path, a_matrix_path, b_matrix_path;

__global__ void MatrixMulKernel (double *d_A, double *d_B, double *d_C, int n)
{
	int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                                                      
	int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y; 
	int k;
	
	d_C[ty*n + tx] = 0.0;
	for (k = 0;  k < n; k++)
	  d_C[ty*n + tx] += d_A[ty*n + k]*d_B[k*n + tx];

}

void generateInputMatrices()
{
	double temp;
	int i, j;
	FILE *f_A, *f_B;

	f_A = fopen(a_matrix_path.c_str(), "wb");
	f_B = fopen(b_matrix_path.c_str(), "wb");


	srand ( time(NULL) );

	for(i=0; i<INMATRIXSIZE; i++)
	{
		for(j=0; j<INMATRIXSIZE; j++){
			temp = (rand()/((double)(RAND_MAX)+1)*(-4.06e16-4.0004e16))+4.1e16;
			fwrite( &temp, sizeof(double), 1, f_A );
		

			temp = (rand()/((double)(RAND_MAX)+1)*(-4.06e16-4.4e16))+4.1e16;
			fwrite( &temp, sizeof(double), 1, f_B );
			
			
		}
	}

	fclose(f_A);
	fclose(f_B);

	return;
}

void ReadMatrixFromFile()
{	
	FILE *f_A, *f_B;

	f_A = fopen(a_matrix_path.c_str(),"rb");
	f_B = fopen(b_matrix_path.c_str(),"rb");
	if (!(f_A&&f_B))
	{
		printf("Error opening matrices A, B.\n");
		exit(-1);
	}
	fread(A,sizeof(double)*size, 1, f_A);
    fread(B,sizeof(double)*size, 1, f_B);

	printf("Done reading matrices\n");

	fclose(f_A);
	fclose(f_B);
}

double mysecond()
{
   struct timeval tp;
   struct timezone tzp;
   int i = gettimeofday(&tp,&tzp);
   return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
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

void generateGoldMatrix()
{
	////////////////////////////////////////////////////
	/////////////KERNELMXM VARS/////////////////////////
	int gridsize = k/BLOCK_SIZE < 1 ? 1 : k/BLOCK_SIZE;
	int blocksize = k/BLOCK_SIZE < 1 ? k : BLOCK_SIZE;
	dim3 dimBlock(blocksize,blocksize);
	dim3 dimGrid(gridsize,gridsize);
	////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////
	//////////DEVICE VARS///////////////////////////////	
	cudaError_t cumalloc_err;
	const char *cumalloc_err_str;

	double *d_A;
	double *d_B;
	double *d_C;
	////////////////////////////////////////////////////

	A = ( double* ) malloc( size * sizeof( double ) );
	B = ( double* ) malloc( size * sizeof( double ) );
	GOLD = ( double* ) malloc( size * sizeof( double ) );

	GetDevice();
	
	ReadMatrixFromFile();

	cumalloc_err = cudaMalloc( ( void** ) &d_A, size * sizeof( double ) );
	cumalloc_err_str = cudaGetErrorString(cumalloc_err);
	if(strcmp(cumalloc_err_str, "no error") != 0) {exit(-3);} //mem allocate failure

	cumalloc_err = cudaMalloc( ( void** ) &d_B, size * sizeof( double ) );
	cumalloc_err_str = cudaGetErrorString(cumalloc_err);
	if(strcmp(cumalloc_err_str, "no error") != 0) {exit(-3);}

	cumalloc_err = cudaMalloc( ( void** ) &d_C, size * sizeof( double ) );
	cumalloc_err_str = cudaGetErrorString(cumalloc_err);
	if(strcmp(cumalloc_err_str, "no error") != 0) {exit(-3);}


	cumalloc_err = cudaMemcpy( d_C, A, size * sizeof( double ), cudaMemcpyHostToDevice ); // ZERA C
	cumalloc_err_str = cudaGetErrorString(cumalloc_err);
	if(strcmp(cumalloc_err_str, "no error") != 0) {exit(-3);}

	cumalloc_err = cudaMemcpy( d_A, A, size * sizeof( double ), cudaMemcpyHostToDevice ); // PUSH A
	cumalloc_err_str = cudaGetErrorString(cumalloc_err);
	if(strcmp(cumalloc_err_str, "no error") != 0) {exit(-3);}

	cumalloc_err = cudaMemcpy( d_B, B, size * sizeof( double ), cudaMemcpyHostToDevice ); // PUSH B
	cumalloc_err_str = cudaGetErrorString(cumalloc_err);
	if(strcmp(cumalloc_err_str, "no error") != 0) {exit(-3);}

	printf("cudaMxM... k=%d", k);
	double time = mysecond();

	MatrixMulKernel<<<dimGrid,dimBlock>>>(d_A, d_B, d_C, k);
	cudaDeviceSynchronize();
	
	printf("\nend in %f\n", mysecond()-time);

	cumalloc_err = cudaMemcpy(GOLD, d_C, size * sizeof( double ), cudaMemcpyDeviceToHost);
	cumalloc_err_str = cudaGetErrorString(cumalloc_err);
	if(strcmp(cumalloc_err_str, "no error") != 0) {exit(-3);}

	cudaFree( d_A );
	cudaFree( d_B );
	cudaFree( d_C );

	FILE *f_GOLD;

	f_GOLD = fopen(gold_matrix_path.c_str(), "wb");

	//printf("-------------------------\n%.10f\n%.10f\n%.10f\n", GOLD[0], GOLD[1], GOLD[2]);

	fwrite(GOLD, sizeof(double)*size, 1, f_GOLD );

	fclose(f_GOLD);

	return;
}

int main (int argc, char** argv)
{
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
	////////////////////////////////////////////////////

	size = k*k;
	
	FILE *test_file;
	test_file=fopen(a_matrix_path.c_str(), "rb");
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
