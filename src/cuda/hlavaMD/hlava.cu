//============================================================================
//	UPDATE
//============================================================================

//	14 APR 2011 Lukasz G. Szafaryn

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdbool.h>			 // (in path known to compiler) needed by true/false
#include <omp.h>

// Helper functions
#include "helper_cuda.h"
#include "helper_string.h"

#include "half.hpp"
#include <cuda_fp16.h>

#ifdef LOGS
#include "log_helper.h"
#endif

//=============================================================================
//	DEFINE / INCLUDE
//=============================================================================
#define NUMBER_PAR_PER_BOX 192	 // keep this low to allow more blocks that share shared memory to run concurrently, code does not work for larger than 110, more speedup can be achieved with larger number and no shared memory used

#define NUMBER_THREADS 192		 // this should be roughly equal to NUMBER_PAR_PER_BOX for best performance

								 // STABLE
#define DOT(A,B) ((A.x)*(B.x)+(A.y)*(B.y)+(A.z)*(B.z))
#define H_DOT(A,B) (__hfma((A.x), (B.x), __hfma((A.y), (B.y), __hmul((A.z), (B.z)))))
#define H2_DOT(A,B) (__hfma2((A.x), (B.x), __hfma2((A.y), (B.y), __hmul2((A.z), (B.z)))))

//=============================================================================
//	STRUCTURES
//=============================================================================

typedef struct
{
	half x, y, z;
} THREE_VECTOR;

typedef struct
{
	half v, x, y, z;
} FOUR_VECTOR;

typedef struct
{
	half2 x, y, z;
} THREE_H2_VECTOR;

typedef struct
{
	half2 v, x, y, z;
} FOUR_H2_VECTOR;

typedef struct nei_str
{
	// neighbor box
	int x, y, z;
	int number;
	long offset;
} nei_str;

typedef struct box_str
{
	// home box
	int x, y, z;
	int number;
	long offset;
	// neighbor boxes
	int nn;
	nei_str nei[26];
} box_str;

typedef struct par_str
{
	half alpha;
} par_str;

typedef struct dim_str
{
	// input arguments
	int cur_arg;
	int arch_arg;
	int cores_arg;
	int boxes1d_arg;
	// system memory
	long number_boxes;
	long box_mem;
	long space_elem;
	long space_mem;
	long space_mem2;
} dim_str;

// Returns the current system time in microseconds
/*long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}*/

void usage(int argc, char** argv) {
    printf("Usage: %s -boxes=N [-generate] [-input_distances=<path>] [-input_charges=<path>] [-output_gold=<path>] [-iterations=N] [-streams=N] [-debug] [-verbose]\n", argv[0]);
}

void getParams(int argc, char** argv, int *boxes, int *generate, char **input_distances, char **input_charges, char **output_gold, int *iterations, int *verbose, int *fault_injection, int *nstreams)
{
	if (argc<2) {
		usage(argc, argv);
		exit(EXIT_FAILURE);
	}

	*generate = 0;
	*iterations = 1000000;
	*nstreams = 1;
	*fault_injection = 0;
	*verbose = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "boxes"))
    {
        *boxes  = getCmdLineArgumentInt(argc, (const char **)argv, "boxes");

        if (*boxes <= 0)
        {
            printf("Invalid input size given on the command-line: %d\n", *boxes);
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        usage(argc, argv);
        exit(EXIT_FAILURE);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "generate"))
    {
        *generate = 1;
        printf(">> Output will be written to file. Only stream #0 output will be considered.\n");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input_distances"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "input_distances", input_distances);
    }
    else
    {
        *input_distances = new char[100];
        snprintf(*input_distances, 100, "input_distances_half_%i", *boxes);
        printf("Using default input_distances path: %s\n", *input_distances);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input_charges"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "input_charges", input_charges);
    }
    else
    {
        *input_charges = new char[100];
        snprintf(*input_charges, 100, "input_charges_half_%i", *boxes);
        printf("Using default input_charges path: %s\n", *input_charges);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "output_gold"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "output_gold", output_gold);
    }
    else
    {
        *output_gold = new char[100];
        snprintf(*output_gold, 100, "output_gold_half_%i", *boxes);
        printf("Using default output_gold path: %s\n", *output_gold);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "iterations"))
    {
        *iterations = getCmdLineArgumentInt(argc, (const char **)argv, "iterations");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "streams"))
    {
        *nstreams = getCmdLineArgumentInt(argc, (const char **)argv, "streams");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "verbose"))
    {
        *verbose = 1;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "debug"))
    {
        *fault_injection = 1;
        printf("!! Will be injected an input error\n");
    }
}

//-----------------------------------------------------------------------------
//	plasmaKernel_gpu_2
//-----------------------------------------------------------------------------

__global__ void kernel_gpu_cuda(par_str d_par_gpu, dim_str d_dim_gpu, box_str* d_box_gpu, FOUR_VECTOR* d_rv_gpu, half* d_qv_gpu, FOUR_VECTOR* d_fv_gpu) {

	//---------------------------------------------------------------------
	//	THREAD PARAMETERS
	//---------------------------------------------------------------------

	int bx = blockIdx.x;		 // get current horizontal block index (0-n)
	int tx = threadIdx.x;		 // get current horizontal thread index (0-n)
	int wtx = tx;

	//---------------------------------------------------------------------
	//	DO FOR THE NUMBER OF BOXES
	//---------------------------------------------------------------------

	if(bx<d_dim_gpu.number_boxes) {

		//-------------------------------------------------------------
		//	Extract input parameters
		//-------------------------------------------------------------

		// parameters
		half a2 = __hmul(__hmul(__float2half(2.0), d_par_gpu.alpha), d_par_gpu.alpha);
		half2 h2_a2 = __half2half2(a2);

		// home box
		int first_i;
		FOUR_VECTOR* rA;
		FOUR_VECTOR* fA;
		FOUR_H2_VECTOR h2_fA[200];
		__shared__ FOUR_H2_VECTOR h2_rA_shared[200];

		// nei box
		int pointer;
		int k = 0;
		int first_j;
		FOUR_VECTOR* rB;
		half* qB;
		int j = 0;
		__shared__ FOUR_H2_VECTOR h2_rB_shared[100];
		__shared__ half2 h2_qB_shared[100];

		// common
		half2 r2;
		half2 u2;
		half2 vij;
		half2 fs;
		half2 fxij;
		half2 fyij;
		half2 fzij;
		THREE_H2_VECTOR d;

		//-------------------------------------------------------------
		//	Home box
		//-------------------------------------------------------------

		//-------------------------------------------------------------
		//	Setup parameters
		//-------------------------------------------------------------

		// home box - box parameters
		first_i = d_box_gpu[bx].offset;

		// home box - distance, force, charge and type parameters
		rA = &d_rv_gpu[first_i];
		fA = &d_fv_gpu[first_i];

		//-------------------------------------------------------------
		//	Copy to shared memory
		//-------------------------------------------------------------

		// home box - shared memory - INCLUDES HALF2 transformation -redundant- on shared memory
		while(wtx<NUMBER_PAR_PER_BOX) {
			h2_rA_shared[wtx].x = __half2half2(rA[wtx].x);
			h2_rA_shared[wtx].y = __half2half2(rA[wtx].y);
			h2_rA_shared[wtx].z = __half2half2(rA[wtx].z);
			h2_rA_shared[wtx].v = __half2half2(rA[wtx].v);
			h2_fA[wtx].x = __half2half2(fA[wtx].x);
			h2_fA[wtx].y = __half2half2(fA[wtx].y);
			h2_fA[wtx].z = __half2half2(fA[wtx].z);
			h2_fA[wtx].v = __half2half2(fA[wtx].v);
			wtx = wtx + NUMBER_THREADS;
		}
		wtx = tx;

		// synchronize threads  - not needed, but just to be safe
		__syncthreads();

		//-------------------------------------------------------------
		//	nei box loop
		//-------------------------------------------------------------

		// loop over neiing boxes of home box
		for (k=0; k<(1+d_box_gpu[bx].nn); k++) {

			//---------------------------------------------
			//	nei box - get pointer to the right box
			//---------------------------------------------

			if(k==0) {
				pointer = bx;	 // set first box to be processed to home box
			}
			else {
								 // remaining boxes are nei boxes
				pointer = d_box_gpu[bx].nei[k-1].number;
			}

			//-----------------------------------------------------
			//	Setup parameters
			//-----------------------------------------------------

			// nei box - box parameters
			first_j = d_box_gpu[pointer].offset;

			// nei box - distance, (force), charge and (type) parameters
			rB = &d_rv_gpu[first_j];
			qB = &d_qv_gpu[first_j];

			//-----------------------------------------------------
			//	Setup parameters
			//-----------------------------------------------------

			// nei box - shared memory - INCLUDES HALF2 optimization on shared memory
			int corrWTX;
			while(wtx<NUMBER_PAR_PER_BOX) {
				corrWTX = floor(wtx / 2.0);
				h2_rB_shared[corrWTX].x.x = rB[wtx + 0].x;
				h2_rB_shared[corrWTX].x.y = rB[wtx + NUMBER_THREADS].x;
				h2_rB_shared[corrWTX].y.x = rB[wtx + 0].y;
				h2_rB_shared[corrWTX].y.y = rB[wtx + NUMBER_THREADS].y;
				h2_rB_shared[corrWTX].z.x = rB[wtx + 0].z;
				h2_rB_shared[corrWTX].z.y = rB[wtx + NUMBER_THREADS].z;
				h2_rB_shared[corrWTX].v.x = rB[wtx + 0].v;
				h2_rB_shared[corrWTX].v.y = rB[wtx + NUMBER_THREADS].v;
				h2_qB_shared[corrWTX].x = qB[wtx + 0];
				h2_qB_shared[corrWTX].y = qB[wtx + NUMBER_THREADS];
				wtx = wtx + NUMBER_THREADS * 2.0;
			}
			wtx = tx;

			// synchronize threads because in next section each thread accesses data brought in by different threads here
			__syncthreads();

			//-----------------------------------------------------
			//	Calculation
			//-----------------------------------------------------

			// Common
			FOUR_H2_VECTOR h2_fA_local;

			// loop for the number of particles in the home box
			// for (int i=0; i<nTotal_i; i++){
			while(wtx<NUMBER_PAR_PER_BOX) {

				h2_fA_local.x = __float2half2_rn(0.0);
				h2_fA_local.y = __float2half2_rn(0.0);
				h2_fA_local.z = __float2half2_rn(0.0);
				h2_fA_local.v = __float2half2_rn(0.0);
				
				// loop for the number of particles in the current nei box
				for (j=0; j<floor(NUMBER_PAR_PER_BOX / 2.0); j++) {
					// Convert input vars from HALF to HALF2 for local work

					// r2 = (half)h2_rA_shared[wtx].v + (half)h2_rB_shared[j].v - H_DOT((half)h2_rA_shared[wtx],(half)h2_rB_shared[j]);
					r2 = __hsub2(__hadd2(h2_rA_shared[wtx].v, h2_rB_shared[j].v),  H2_DOT(h2_rA_shared[wtx], h2_rB_shared[j]));

					// u2 = a2*r2;
					u2 = __hmul2(h2_a2, r2);
					// vij= exp(-u2);
					vij= h2exp(__hneg2(u2));
					// fs = 2*vij;
					fs = __hmul2(__float2half2_rn(2.0), vij);

					// d.x = (half)h2_rA_shared[wtx].x  - (half)h2_rB_shared[j].x;
					d.x = __hsub2(h2_rA_shared[wtx].x, h2_rB_shared[j].x);
					// fxij=fs*d.x;
					fxij=__hmul2(fs, d.x);
					// d.y = (half)h2_rA_shared[wtx].y  - (half)h2_rB_shared[j].y;
					d.y = __hsub2(h2_rA_shared[wtx].y, h2_rB_shared[j].y);
					// fyij=fs*d.y;
					fyij=__hmul2(fs, d.y);
					// d.z = (half)h2_rA_shared[wtx].z  - (half)h2_rB_shared[j].z;
					d.z = __hsub2(h2_rA_shared[wtx].z, h2_rB_shared[j].z);
					// fzij=fs*d.z;
					fzij=__hmul2(fs, d.z);

					// fA[wtx].v +=  (half)((half)h2_qB_shared[j]*vij);
					h2_fA_local.v = __hfma2(h2_qB_shared[j], vij, h2_fA_local.v);
					// fA[wtx].x +=  (half)((half)h2_qB_shared[j]*fxij);
					h2_fA_local.x = __hfma2(h2_qB_shared[j], fxij, h2_fA_local.x);
					// fA[wtx].y +=  (half)((half)h2_qB_shared[j]*fyij);
					h2_fA_local.y = __hfma2(h2_qB_shared[j], fyij, h2_fA_local.y);
					// fA[wtx].z +=  (half)((half)h2_qB_shared[j]*fzij);
					h2_fA_local.z = __hfma2(h2_qB_shared[j], fzij, h2_fA_local.z);
				}

				// Copy back data from local memory to global memory
				fA[wtx].x = __hadd(fA[wtx].x, __hadd(h2_fA_local.x.x, h2_fA_local.x.y));
				fA[wtx].y = __hadd(fA[wtx].y, __hadd(h2_fA_local.y.x, h2_fA_local.y.y));
				fA[wtx].z = __hadd(fA[wtx].z, __hadd(h2_fA_local.z.x, h2_fA_local.z.y));
				fA[wtx].v = __hadd(fA[wtx].v, __hadd(h2_fA_local.v.x, h2_fA_local.v.y));

				// increment work thread index
				wtx = wtx + NUMBER_THREADS;
			}

			// reset work index
			wtx = tx;

			// synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
			__syncthreads();

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Calculation END
			//----------------------------------------------------------------------------------------------------------------------------------140
		}
		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	nei box loop END
		//------------------------------------------------------------------------------------------------------------------------------------------------------160
	}
}

double mysecond()
{
   struct timeval tp;
   struct timezone tzp;
   int i = gettimeofday(&tp,&tzp);
   return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void generateInput(dim_str dim_cpu, char *input_distances, FOUR_VECTOR **rv_cpu, char *input_charges, half_float::half **qv_cpu)
{
	// random generator seed set to random value - time in this case
	FILE *fp;
	int i;

	srand(time(NULL));

	// input (distances)
	if( (fp = fopen(input_distances, "wb" )) == 0 ) {
		printf( "The file 'input_distances' was not opened\n" );
		#ifdef LOGS
			log_error_detail("The file 'input_distances' was not opened"); end_log_file();
		#endif
	 	exit(EXIT_FAILURE);
	}
	*rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
	for(i=0; i<dim_cpu.space_elem; i=i+1) {
		half_float::half tempValue((rand()%10 + 1) / 10.0);

								 // get a number in the range 0.1 - 1.0
		tempValue = half_float::half((rand()%10 + 1) / 10.0);
		(*rv_cpu)[i].v = *((half*)&tempValue);
		fwrite(&((*rv_cpu)[i].v), 1, sizeof(half), fp);

								 // get a number in the range 0.1 - 1.0
		tempValue = half_float::half((rand()%10 + 1) / 10.0);
		(*rv_cpu)[i].x = *((half*)&tempValue);
		fwrite(&((*rv_cpu)[i].x), 1, sizeof(half), fp);

								 // get a number in the range 0.1 - 1.0
		tempValue = half_float::half((rand()%10 + 1) / 10.0);
		(*rv_cpu)[i].y = *((half*)&tempValue);
		fwrite(&((*rv_cpu)[i].y), 1, sizeof(half), fp);

								 // get a number in the range 0.1 - 1.0
		tempValue = half_float::half((rand()%10 + 1) / 10.0);
		(*rv_cpu)[i].z = *((half*)&tempValue);
		fwrite(&((*rv_cpu)[i].z), 1, sizeof(half), fp);
	}
	fclose(fp);

	// input (charge)
	if( (fp = fopen(input_charges, "wb" )) == 0 ) {
		printf( "The file 'input_charges' was not opened\n" );
		#ifdef LOGS
			log_error_detail("The file 'input_charges' was not opened"); end_log_file();
		#endif
	 	exit(EXIT_FAILURE);
	}

	*qv_cpu = (half_float::half*)malloc(dim_cpu.space_mem2);
	for(i=0; i<dim_cpu.space_elem; i=i+1) {
								 // get a number in the range 0.1 - 1.0
		half_float::half tempValue((rand()%10 + 1) / 10.0);
		(*qv_cpu)[i] = *((half_float::half*)&tempValue);
		fwrite(&((*qv_cpu)[i]), 1, sizeof(half), fp);
	}
	fclose(fp);
}

void readInput(dim_str dim_cpu, char *input_distances, FOUR_VECTOR **rv_cpu, char *input_charges, half_float::half **qv_cpu, int fault_injection)
{
	FILE *fp;
	int i;
	size_t return_value[4];

	// input (distances)
	if( (fp = fopen(input_distances, "rb" )) == 0 ) {
		printf( "The file 'input_distances' was not opened\n" );
		#ifdef LOGS
			log_error_detail("The file 'input_distances' was not opened"); end_log_file();
		#endif
	 	exit(EXIT_FAILURE);
	}

	*rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
	if(*rv_cpu == NULL) {
		printf("error rv_cpu malloc\n");
		#ifdef LOGS
			log_error_detail("error rv_cpu malloc"); end_log_file();
		#endif
		exit(1);
	}
	for(i=0; i<dim_cpu.space_elem; i=i+1) {
		return_value[0] = fread(&((*rv_cpu)[i].v), 1, sizeof(half), fp);
		return_value[1] = fread(&((*rv_cpu)[i].x), 1, sizeof(half), fp);
		return_value[2] = fread(&((*rv_cpu)[i].y), 1, sizeof(half), fp);
		return_value[3] = fread(&((*rv_cpu)[i].z), 1, sizeof(half), fp);
		if (return_value[0] == 0 || return_value[1] == 0 || return_value[2] == 0 || return_value[3] == 0) {
			printf("error reading rv_cpu from file\n");
			#ifdef LOGS
				log_error_detail("error reading rv_cpu from file"); end_log_file();
			#endif
			exit(1);
		}
	}
	fclose(fp);

	// input (charge)
	if( (fp = fopen(input_charges, "rb" )) == 0 ) {
		printf( "The file 'input_charges' was not opened\n" );
		#ifdef LOGS
			log_error_detail("The file 'input_charges' was not opened"); end_log_file();
		#endif
	 	exit(EXIT_FAILURE);
	}

	*qv_cpu = (half_float::half*)malloc(dim_cpu.space_mem2);
	if(*qv_cpu == NULL) {
		printf("error qv_cpu malloc\n");
		#ifdef LOGS
			log_error_detail("error qv_cpu malloc"); end_log_file();
		#endif
		exit(1);
	}
	for(i=0; i<dim_cpu.space_elem; i=i+1) {
		return_value[0] = fread(&((*qv_cpu)[i]), 1, sizeof(half), fp);
		if (return_value[0] == 0) {
			printf("error reading qv_cpu from file\n");
			#ifdef LOGS
				log_error_detail("error reading qv_cpu from file"); end_log_file();
			#endif
			exit(1);
		}
	}
	fclose(fp);

	// =============== Fault injection
	if (fault_injection) {
		half_float::half tempValue(0.732637263);
		(*qv_cpu)[2] = *((half_float::half*)&tempValue); // must be in range 0.1 - 1.0
		printf("!!> Fault injection: qv_cpu[2]=%f\n", *((float*)&((*qv_cpu)[2])));
	}
	// ========================
}

void readGold(dim_str dim_cpu, char *output_gold, FOUR_VECTOR **fv_cpu_GOLD)
{
	FILE *fp;
	size_t return_value[4];
	int i;

	if( (fp = fopen(output_gold, "rb" )) == 0 )
	{
		printf( "The file 'output_forces' was not opened\n" ); exit(EXIT_FAILURE);
	}

	*fv_cpu_GOLD = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
	if(*fv_cpu_GOLD == NULL) {
		printf("error fv_cpu_GOLD malloc\n");
		#ifdef LOGS
			log_error_detail("error fv_cpu_GOLD malloc"); end_log_file();
		#endif
		exit(1);
	}
	for(i=0; i<dim_cpu.space_elem; i=i+1) {
		return_value[0] = fread(&((*fv_cpu_GOLD)[i].v), 1, sizeof(half), fp);
		return_value[1] = fread(&((*fv_cpu_GOLD)[i].x), 1, sizeof(half), fp);
		return_value[2] = fread(&((*fv_cpu_GOLD)[i].y), 1, sizeof(half), fp);
		return_value[3] = fread(&((*fv_cpu_GOLD)[i].z), 1, sizeof(half), fp);
		if ((float)return_value[0] == 0 || (float)return_value[1] == 0 || (float)return_value[2] == 0 || (float)return_value[3] == 0) {
			printf("error reading rv_cpu from file\n");
			#ifdef LOGS
				log_error_detail("error reading rv_cpu from file"); end_log_file();
			#endif
			exit(1);
		}
	}
	fclose(fp);
}

void writeGold(dim_str dim_cpu, char *output_gold, FOUR_VECTOR **fv_cpu)
{
	FILE *fp;
	int i;

	if( (fp = fopen(output_gold, "wb" )) == 0 ) {
		printf( "The file 'output_forces' was not opened\n" ); exit(EXIT_FAILURE);
	}
	int number_zeros = 0;
	for(i=0; i<dim_cpu.space_elem; i=i+1) {
		half_float::half tempValue;
		tempValue = *((half_float::half*)&((*fv_cpu)[i].v));
		if(tempValue == 0.0 || isnan(tempValue))
			number_zeros++;
		tempValue = *((half_float::half*)&((*fv_cpu)[i].x));
		if(tempValue == 0.0 || isnan(tempValue))
			number_zeros++;
		tempValue = *((half_float::half*)&((*fv_cpu)[i].y));
		if(tempValue == 0.0 || isnan(tempValue))
			number_zeros++;
		tempValue = *((half_float::half*)&((*fv_cpu)[i].z));
		if(tempValue == 0.0 || isnan(tempValue))
			number_zeros++;

		fwrite(&((*fv_cpu)[i].v), 1, sizeof(half), fp);
		fwrite(&((*fv_cpu)[i].x), 1, sizeof(half), fp);
		fwrite(&((*fv_cpu)[i].y), 1, sizeof(half), fp);
		fwrite(&((*fv_cpu)[i].z), 1, sizeof(half), fp);
	}
	printf("Number of zeros/NaNs written on output: %d\n", number_zeros);
	fclose(fp);
}

//=============================================================================
//	MAIN FUNCTION
//=============================================================================

int main(int argc, char *argv []) {

	//=====================================================================
	//	CPU/MCPU VARIABLES
	//=====================================================================

	// timer
	double timestamp;

	// counters
	int i, j, k, l, m, n;
	int iterations;

	int generate, verbose, fault_injection;

	// system memory
	par_str par_cpu;
	dim_str dim_cpu;
	box_str* box_cpu;
	FOUR_VECTOR* rv_cpu;
	half_float::half* qv_cpu;
	FOUR_VECTOR* fv_cpu;
	FOUR_VECTOR* fv_cpu_GOLD;
	int nh;
	int nstreams, streamIdx;

	cudaError_t cuda_error;
	const char *error_string;

	char *input_distances, *input_charges, *output_gold;

	int number_nn = 0;

	//=====================================================================
	//	CHECK INPUT ARGUMENTS
	//=====================================================================

	getParams(argc, argv, &dim_cpu.boxes1d_arg, &generate, &input_distances, &input_charges, &output_gold, &iterations, &verbose, &fault_injection, &nstreams);

	printf("streams: %d boxes:%d block_size:%d\n", nstreams, dim_cpu.boxes1d_arg, NUMBER_THREADS);

	#ifdef LOGS
		char test_info[100];
		snprintf(test_info, 100, "streams: %d boxes:%d block_size:%d", nstreams, dim_cpu.boxes1d_arg, NUMBER_THREADS);
		if (!generate) start_log_file("cudaHalfLavaMD", test_info);
	#endif

	//=====================================================================
	//	INPUTS
	//=====================================================================
	half_float::half tempValue(0.5);
	par_cpu.alpha = *((half*)&tempValue);
	//=====================================================================
	//	DIMENSIONS
	//=====================================================================

	// total number of boxes
	dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;

	// how many particles space has in each direction
	dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
	dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
	dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(half);

	// box array
	dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

	//=====================================================================
	//	SYSTEM MEMORY
	//=====================================================================

	//=====================================================================
	//	BOX
	//=====================================================================

	// allocate boxes
	box_cpu = (box_str*)malloc(dim_cpu.box_mem);
	if(box_cpu == NULL) {
		printf("error box_cpu malloc\n");
		#ifdef LOGS
			if (!generate) log_error_detail("error box_cpu malloc"); end_log_file();
		#endif
		exit(1);
	}

	// initialize number of home boxes
	nh = 0;

	// home boxes in z direction
	for(i=0; i<dim_cpu.boxes1d_arg; i++) {
		// home boxes in y direction
		for(j=0; j<dim_cpu.boxes1d_arg; j++) {
			// home boxes in x direction
			for(k=0; k<dim_cpu.boxes1d_arg; k++) {

				// current home box
				box_cpu[nh].x = k;
				box_cpu[nh].y = j;
				box_cpu[nh].z = i;
				box_cpu[nh].number = nh;
				box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

				// initialize number of neighbor boxes
				box_cpu[nh].nn = 0;

				// neighbor boxes in z direction
				for(l=-1; l<2; l++) {
					// neighbor boxes in y direction
					for(m=-1; m<2; m++) {
						// neighbor boxes in x direction
						for(n=-1; n<2; n++) {

							// check if (this neighbor exists) and (it is not the same as home box)
							if(     (((i+l)>=0 && (j+m)>=0 && (k+n)>=0)==true && ((i+l)<dim_cpu.boxes1d_arg && (j+m)<dim_cpu.boxes1d_arg && (k+n)<dim_cpu.boxes1d_arg)==true)   &&
							(l==0 && m==0 && n==0)==false   ) {

								// current neighbor box
								box_cpu[nh].nei[box_cpu[nh].nn].x = (k+n);
								box_cpu[nh].nei[box_cpu[nh].nn].y = (j+m);
								box_cpu[nh].nei[box_cpu[nh].nn].z = (i+l);
								box_cpu[nh].nei[box_cpu[nh].nn].number = (box_cpu[nh].nei[box_cpu[nh].nn].z * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg) +
								(box_cpu[nh].nei[box_cpu[nh].nn].y * dim_cpu.boxes1d_arg) + box_cpu[nh].nei[box_cpu[nh].nn].x;
								box_cpu[nh].nei[box_cpu[nh].nn].offset = box_cpu[nh].nei[box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;

								// increment neighbor box
								box_cpu[nh].nn = box_cpu[nh].nn + 1;
								number_nn += box_cpu[nh].nn;

							}

						}	 // neighbor boxes in x direction
					}		 // neighbor boxes in y direction
				}			 // neighbor boxes in z direction

				// increment home box
				nh = nh + 1;

			}				 // home boxes in x direction
		}					 // home boxes in y direction
	}						 // home boxes in z direction

	//=====================================================================
	//	PARAMETERS, DISTANCE, CHARGE AND FORCE
	//=====================================================================

	if (generate) {
		generateInput(dim_cpu, input_distances, &rv_cpu, input_charges, &qv_cpu);
	} else {
		readInput(dim_cpu, input_distances, &rv_cpu, input_charges, &qv_cpu, fault_injection);
		readGold(dim_cpu, output_gold, &fv_cpu_GOLD);
	}

	//=====================================================================
	//	EXECUTION PARAMETERS
	//=====================================================================

	dim3 threads;
	dim3 blocks;

	blocks.x = dim_cpu.number_boxes;
	blocks.y = 1;
	// define the number of threads in the block
	threads.x = NUMBER_THREADS;
	threads.y = 1;

	cudaStream_t *streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));

	//LOOP START
	int loop;
	for(loop=0; loop<iterations; loop++) {

		if (verbose) {
			printf("[Iteration #%i]=====================================\n", loop); fflush(stdout);
		}

		double globaltimer = mysecond();
		timestamp = mysecond();

		// prepare host memory to receive kernel output
		// output (forces)
		fv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
		if(fv_cpu == NULL) {
			printf("error fv_cpu malloc\n");
			#ifdef LOGS
				if (!generate) log_error_detail("error fv_cpu malloc"); end_log_file();
			#endif
			exit(1);
		}
		for(i=0; i<dim_cpu.space_elem; i=i+1) {
			// set to 0, because kernels keeps adding to initial value
			half_float::half zeroValue(0.0);
			fv_cpu[i].v = *((half*)&zeroValue);
			fv_cpu[i].x = *((half*)&zeroValue);
			fv_cpu[i].y = *((half*)&zeroValue);
			fv_cpu[i].z = *((half*)&zeroValue);
		}

		//=====================================================================
		//	GPU_CUDA
		//=====================================================================

		//=====================================================================
		//	VARIABLES
		//=====================================================================

		box_str* d_box_gpu[nstreams];
		FOUR_VECTOR* d_rv_gpu[nstreams];
		half* d_qv_gpu[nstreams];
		FOUR_VECTOR* d_fv_gpu[nstreams];


		//=====================================================================
		//	GPU SETUP
		//=====================================================================
		for (streamIdx = 0; streamIdx < nstreams; streamIdx++) {
      		cudaStreamCreateWithFlags(&(streams[streamIdx]), cudaStreamNonBlocking);

			//==================================================
			//	boxes
			//==================================================

			cuda_error = cudaMalloc( (void **)&(d_box_gpu[streamIdx]), dim_cpu.box_mem);
			error_string = cudaGetErrorString(cuda_error);
			if(strcmp(error_string, "no error") != 0) {
				printf("error d_box_gpu cudaMalloc\n");
				#ifdef LOGS
					if (!generate) log_error_detail("error d_box_gpu cudamalloc"); end_log_file();
				#endif
				exit(1);
			}
			//==================================================
			//	rv
			//==================================================

			cuda_error = cudaMalloc( (void **)&(d_rv_gpu[streamIdx]), dim_cpu.space_mem);
			error_string = cudaGetErrorString(cuda_error);
			if(strcmp(error_string, "no error") != 0) {
				printf("error d_rv_gpu cudaMalloc\n");
				#ifdef LOGS
					if (!generate) log_error_detail("error d_box_gpu cudamalloc"); end_log_file();
				#endif
				exit(1);
			}
			//==================================================
			//	qv
			//==================================================

			cuda_error = cudaMalloc( (void **)&(d_qv_gpu[streamIdx]), dim_cpu.space_mem2);
			error_string = cudaGetErrorString(cuda_error);
			if(strcmp(error_string, "no error") != 0) {
				printf("error d_qv_gpu cudaMalloc\n");
				#ifdef LOGS
					if (!generate) log_error_detail("error d_box_gpu cudamalloc"); end_log_file();
				#endif
				exit(1);
			}
			//==================================================
			//	fv
			//==================================================

			cuda_error = cudaMalloc( (void **)&(d_fv_gpu[streamIdx]), dim_cpu.space_mem);
			error_string = cudaGetErrorString(cuda_error);
			if(strcmp(error_string, "no error") != 0) {
				printf("error d_fv_gpu cudaMalloc\n");
				#ifdef LOGS
					if (!generate) log_error_detail("error d_box_gpu cudamalloc"); end_log_file();
				#endif
				exit(1);
			}

			//=====================================================================
			//	GPU MEMORY			COPY
			//=====================================================================

			//==================================================
			//	boxes
			//==================================================

			cuda_error = cudaMemcpy(d_box_gpu[streamIdx], box_cpu, dim_cpu.box_mem, cudaMemcpyHostToDevice);
			error_string = cudaGetErrorString(cuda_error);
			if(strcmp(error_string, "no error") != 0) {
				printf("error load d_boc_gpu\n");
				#ifdef LOGS
					if (!generate) log_error_detail("error load d_box_gpu"); end_log_file();
				#endif
				exit(1);
			}
			//==================================================
			//	rv
			//==================================================

			cuda_error = cudaMemcpy( d_rv_gpu[streamIdx], rv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice);
			error_string = cudaGetErrorString(cuda_error);
			if(strcmp(error_string, "no error") != 0) {
				printf("error load d_rv_gpu\n");
				#ifdef LOGS
					if (!generate) log_error_detail("error load d_box_gpu"); end_log_file();
				#endif
				exit(1);
			}
			//==================================================
			//	qv
			//==================================================

			cuda_error = cudaMemcpy( d_qv_gpu[streamIdx], qv_cpu, dim_cpu.space_mem2, cudaMemcpyHostToDevice);
			error_string = cudaGetErrorString(cuda_error);
			if(strcmp(error_string, "no error") != 0) {
				printf("error load d_qv_gpu\n");
				#ifdef LOGS
					if (!generate) log_error_detail("error load d_box_gpu"); end_log_file();
				#endif
				exit(1);
			}
			//==================================================
			//	fv
			//==================================================

			cuda_error = cudaMemcpy( d_fv_gpu[streamIdx], fv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice);
			error_string = cudaGetErrorString(cuda_error);
			if(strcmp(error_string, "no error") != 0) {
				printf("error load d_fv_gpu\n");
				#ifdef LOGS
					if (!generate) log_error_detail("error load d_box_gpu"); end_log_file();
				#endif
				exit(1);
			}
		}
    	if (verbose) printf("[Iteration #%i] Setup prepare time: %.4fs\n", loop, mysecond()-timestamp);

		//=====================================================================
		//	KERNEL
		//=====================================================================

		double kernel_time=mysecond();
		#ifdef LOGS
			if (!generate) start_iteration();
		#endif
		// launch kernel - all boxes
		for (streamIdx = 0; streamIdx < nstreams; streamIdx++) {
			kernel_gpu_cuda<<<blocks, threads, 0, streams[streamIdx]>>>( par_cpu, dim_cpu, \
				d_box_gpu[streamIdx], d_rv_gpu[streamIdx], d_qv_gpu[streamIdx], d_fv_gpu[streamIdx]);
		}
		//printf("All kernels were commited.\n");
		for (streamIdx = 0; streamIdx < nstreams; streamIdx++) {
			cuda_error = cudaStreamSynchronize(streams[streamIdx]);
		}
		#ifdef LOGS
			if (!generate) end_iteration();
		#endif
		kernel_time = mysecond()-kernel_time;
		error_string = cudaGetErrorString(cuda_error);
		if(strcmp(error_string, "no error") != 0) {
			printf("error logic: %s\n",error_string);
			#ifdef LOGS
				if (!generate) log_error_detail("error logic:"); end_log_file();
			#endif
			exit(1);
		}


		//=====================================================================
		//	COMPARE OUTPUTS / WRITE GOLD
		//=====================================================================
		if (generate){
			cuda_error = cudaMemcpy( fv_cpu, d_fv_gpu[0], dim_cpu.space_mem, cudaMemcpyDeviceToHost);
			error_string = cudaGetErrorString(cuda_error);
			if(strcmp(error_string, "no error") != 0) {
				printf("error download fv_cpu\n");
				exit(1);
			}
			writeGold(dim_cpu, output_gold, &fv_cpu);
		} else { // Check gold
			//int ea = 0;
			int thread_error = 0;
			int kernel_errors = 0;
			char error_detail[300];
			timestamp = mysecond();
			for (streamIdx = 0; streamIdx < nstreams; streamIdx++) {

				//=====================================================================
				//	GPU MEMORY			COPY BACK
				//=====================================================================

				cuda_error = cudaMemcpy( fv_cpu, d_fv_gpu[streamIdx], dim_cpu.space_mem, cudaMemcpyDeviceToHost);
				error_string = cudaGetErrorString(cuda_error);
				if(strcmp(error_string, "no error") != 0) {
					printf("error download fv_cpu\n");
					#ifdef LOGS
						if (!generate) log_error_detail("error download fv_cpu"); end_log_file();
					#endif
					exit(1);
				}

				#pragma omp parallel for
				for(i=0; i<dim_cpu.space_elem; i=i+1) {
					if(*((half_float::half*)&(fv_cpu_GOLD[i].v)) != *((half_float::half*)&(fv_cpu[i].v))) {
						thread_error++;
					}
					if(*((half_float::half*)&(fv_cpu_GOLD[i].x)) != *((half_float::half*)&(fv_cpu[i].x))) {
						thread_error++;
					}
					if(*((half_float::half*)&(fv_cpu_GOLD[i].y)) != *((half_float::half*)&(fv_cpu[i].y))) {
						thread_error++;
					}
					if(*((half_float::half*)&(fv_cpu_GOLD[i].z)) != *((half_float::half*)&(fv_cpu[i].z))) {
						thread_error++;
					}
					if (thread_error  > 0) {
						#pragma omp critical
						{
							kernel_errors++;

							snprintf(error_detail, 300, "stream: %d, p: [%d], ea: %d, v_r: %1.16e, v_e: %1.16e, x_r: %1.16e, x_e: %1.16e, y_r: %1.16e, y_e: %1.16e, z_r: %1.16e, z_e: %1.16e\n", streamIdx, \
								i, thread_error, (float)*((half_float::half*)&(fv_cpu[i].v)), (float)*((half_float::half*)&(fv_cpu_GOLD[i].v)), (float)*((half_float::half*)&(fv_cpu[i].x)), (float)*((half_float::half*)&(fv_cpu_GOLD[i].x)), (float)*((half_float::half*)&(fv_cpu[i].y)), (float)*((half_float::half*)&(fv_cpu_GOLD[i].y)), (float)*((half_float::half*)&(fv_cpu[i].z)), (float)*((half_float::half*)&(fv_cpu_GOLD[i].z)));
							if (kernel_errors<25) printf("ERROR: %s\n", error_detail);
							if (kernel_errors>=25 && kernel_errors % 25 == 0) printf("!");
							#ifdef LOGS
								if (!generate) log_error_detail(error_detail);
							#endif
							thread_error = 0;
						}
					}
				}
			}
			#ifdef LOGS
				if (!generate) log_error_count(kernel_errors);
			#endif

			if (verbose) printf("[Iteration #%i] Gold check time: %f\n", loop, mysecond() - timestamp);
		}

		//================= PERF
		// iterate for each neighbor of a box (number_nn)
		double flop =  number_nn;
		// The last for iterate NUMBER_PAR_PER_BOX times
		flop *= NUMBER_PAR_PER_BOX;
		// the last for uses 46 operations plus 2 exp() functions
		flop *=46;
		flop *= nstreams;
	    double flops = (double)flop/kernel_time;
	    double outputpersec = (double)dim_cpu.space_elem * 4 * nstreams / kernel_time;
	    if (verbose) printf("[Iteration #%i] BOXES:%d BLOCK:%d OUTPUT/S:%.2f FLOPS:%.2f (GFLOPS:%.2f)\n", loop, dim_cpu.boxes1d_arg, NUMBER_THREADS, outputpersec, flops, flops/1000000000);
	    if (verbose) printf("[Iteration #%i] kernel_time:%f\n", loop, kernel_time);
		//=====================


		printf(".");
		fflush(stdout);
		//=====================================================================
		//	GPU MEMORY DEALLOCATION
		//=====================================================================
		for (streamIdx = 0; streamIdx < nstreams; streamIdx++) {
			cudaFree(d_rv_gpu[streamIdx]);
			cudaFree(d_qv_gpu[streamIdx]);
			cudaFree(d_fv_gpu[streamIdx]);
			cudaFree(d_box_gpu[streamIdx]);
		}

		//=====================================================================
		//	SYSTEM MEMORY DEALLOCATION
		//=====================================================================
		free(fv_cpu);

		if (verbose) printf("[Iteration #%i] Elapsed time: %.4fs\n", loop, mysecond()-globaltimer);
	}
	if (!generate) free(fv_cpu_GOLD);
	free(rv_cpu);
	free(qv_cpu);
	free(box_cpu);
	printf("\n");

	#ifdef LOGS
    	if (!generate) end_log_file();
    #endif

	return 0;
}
