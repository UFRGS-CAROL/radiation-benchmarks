//============================================================================
//	UPDATE
//============================================================================

//	14 APR 2011 Lukasz G. Szafaryn
//  2014-2018 Caio Lunardi
//  2018 Fernando Fernandes dos Santos

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdbool.h>

#ifdef USE_OMP
#include <omp.h>
#endif

// Helper functions
#include "helper_cuda.h"
#include "helper_string.h"

#ifdef LOGS
#include "log_helper.h"
#endif

#ifdef SAFE_MALLOC
#include "safe_memory/safe_memory.h"
#else
#define checkFrameworkErrors(error) __checkFrameworkErrors(error, __LINE__, __FILE__)
void __checkFrameworkErrors(cudaError_t error, int line, const char* file) {
	if (error == cudaSuccess) {
		return;
	}
	char errorDescription[250];
	snprintf(errorDescription, 250, "CUDA Framework error: %s. Bailing.",
			cudaGetErrorString(error));
#ifdef LOGS
	log_error_detail(errorDescription);
#endif

	printf("%s - Line: %d at %s\n", errorDescription, line, file);
	exit (EXIT_FAILURE);
}
#endif

#ifdef PRECISION_HALF 
#include "half.hpp"
#include <cuda_fp16.h>
#endif

//=============================================================================
//	DEFINE / INCLUDE
//=============================================================================
#define NUMBER_PAR_PER_BOX 192	 // keep this low to allow more blocks that share shared memory to run concurrently, code does not work for larger than 110, more speedup can be achieved with larger number and no shared memory used

#define NUMBER_THREADS 192		 // this should be roughly equal to NUMBER_PAR_PER_BOX for best performance

								 // STABLE
#define DOT(A,B) ((A.x)*(B.x)+(A.y)*(B.y)+(A.z)*(B.z))
#define MAX_LOGGED_ERRORS_PER_STREAM 100
#define MAX_LOGGED_INFOS_PER_STREAM 100

//=============================================================================
//	DEFINE TEST TYPE WITH INTRINSIC TYPES
//=============================================================================
#if defined(PRECISION_DOUBLE)

	const char test_precision_description[] = "double";
	typedef double tested_type;
	typedef double tested_type_host;

#elif defined(PRECISION_SINGLE)

	const char test_precision_description[] = "single";
	typedef float tested_type;
	typedef float tested_type_host;

#elif defined(PRECISION_HALF)

	#define H2_DOT(A,B) (__hfma2((A.x), (B.x), __hfma2((A.y), (B.y), __hmul2((A.z), (B.z)))))

	const char test_precision_description[] = "half";
	typedef half tested_type;
	typedef half_float::half tested_type_host;

#else 
	#error TEST TYPE NOT DEFINED OR INCORRECT. USE PRECISION=<double|single|half>.
#endif

//=============================================================================
//	STRUCTURES
//=============================================================================

typedef struct
{
	tested_type x, y, z;
} THREE_VECTOR;

typedef struct
{
	tested_type v, x, y, z;
} FOUR_VECTOR;

typedef struct
{
	tested_type_host x, y, z;
} THREE_VECTOR_HOST;

typedef struct
{
	tested_type_host v, x, y, z;
} FOUR_VECTOR_HOST;

__host__ inline bool operator==(const FOUR_VECTOR_HOST& lhs, const FOUR_VECTOR_HOST& rhs){
	return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z) && (lhs.v == rhs.v);
}
__host__ inline bool operator!=(const FOUR_VECTOR_HOST& lhs, const FOUR_VECTOR_HOST& rhs){return !operator==(lhs,rhs);}


__device__ inline bool operator==(const FOUR_VECTOR& lhs, const FOUR_VECTOR& rhs){ 
	return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z) && (lhs.v == rhs.v);
}
__device__ inline bool operator!=(const FOUR_VECTOR& lhs, const FOUR_VECTOR& rhs){return !operator==(lhs,rhs);}

#if defined(PRECISION_HALF)

typedef struct
{
	half2 x, y, z;
} THREE_H2_VECTOR;

typedef struct
{
	half2 v, x, y, z;
} FOUR_H2_VECTOR;

__device__ inline bool operator==(const FOUR_H2_VECTOR& lhs, const FOUR_H2_VECTOR& rhs){ 
	return 	__hbne2(lhs.x, rhs.x) && 
			__hbne2(lhs.y, rhs.y) && 
			__hbne2(lhs.z, rhs.z) && 
			__hbne2(lhs.v, rhs.v);
}
__device__ inline bool operator!=(const FOUR_H2_VECTOR& lhs, const FOUR_H2_VECTOR& rhs){return !operator==(lhs,rhs);}

#endif

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
	tested_type alpha;
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

__host__ __device__ inline bool operator==(const box_str& lhs, const box_str& rhs){
	return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z) && (lhs.number == rhs.number) && (lhs.offset == rhs.offset);
}
__host__ __device__ inline bool operator!=(const box_str& lhs, const box_str& rhs){return !operator==(lhs,rhs);}

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

    if (checkCmdLineFlag(argc, (const char **)argv, "input_distances"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "input_distances", input_distances);
    }
    else
    {
        *input_distances = new char[100];
        snprintf(*input_distances, 100, "lava_%s_distances_%i", test_precision_description, *boxes);
        printf("Using default input_distances path: %s\n", *input_distances);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input_charges"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "input_charges", input_charges);
    }
    else
    {
        *input_charges = new char[100];
        snprintf(*input_charges, 100, "lava_%s_charges_%i", test_precision_description, *boxes);
        printf("Using default input_charges path: %s\n", *input_charges);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "output_gold"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "output_gold", output_gold);
    }
    else
    {
        *output_gold = new char[100];
        snprintf(*output_gold, 100, "lava_%s_gold_%i", test_precision_description, *boxes);
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

    if (checkCmdLineFlag(argc, (const char **)argv, "generate"))
    {
		*generate = 1;
		*iterations = 1;
        printf(">> Output will be written to file. Only stream #0 output will be considered.\n");
    }
}

__device__ unsigned long long int is_memory_bad = 0;

template<typename T>
__device__ T inline read_voter(T *v1, T *v2, T *v3,
		int offset) {

	register T in1 = v1[offset];
	register T in2 = v2[offset];
	register T in3 = v3[offset];

	if (in1 == in2 || in1 == in3) {
		return in1;
	}

	if (in2 == in3) {
		return in2;
	}

	if (in1 != in2 && in2 != in3 && in1 != in3) {
		atomicAdd(&is_memory_bad, 1);
	}

	return in1;
}

//-----------------------------------------------------------------------------
//	plasmaKernel_gpu_2
//-----------------------------------------------------------------------------
// #if defined(PRECISION_DOUBLE) or defined(PRECISION_SINGLE)
__global__ void kernel_gpu_cuda(par_str d_par_gpu, 
								dim_str d_dim_gpu, 
								box_str* d_box_gpu_1, box_str* d_box_gpu_2, box_str* d_box_gpu_3, 
								FOUR_VECTOR* d_rv_gpu_1, FOUR_VECTOR* d_rv_gpu_2, FOUR_VECTOR* d_rv_gpu_3, 
								tested_type* d_qv_gpu_1, tested_type* d_qv_gpu_2, tested_type* d_qv_gpu_3, 
								FOUR_VECTOR* d_fv_gpu_1, FOUR_VECTOR* d_fv_gpu_2, FOUR_VECTOR* d_fv_gpu_3) {

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
		tested_type a2 = tested_type(2.0)*d_par_gpu.alpha*d_par_gpu.alpha;

		// home box
		int first_i;
		FOUR_VECTOR *rA_1, *rA_2, *rA_3;
		FOUR_VECTOR *fA_1, *fA_2, *fA_3;
		__shared__ FOUR_VECTOR rA_shared_1[200];
		// __shared__ FOUR_VECTOR rA_shared_2[200];
		// __shared__ FOUR_VECTOR rA_shared_3[200];

		// nei box
		int pointer;
		int k = 0;
		int first_j;
		FOUR_VECTOR *rB_1, *rB_2, *rB_3;
		tested_type *qB_1, *qB_2, *qB_3;
		int j = 0;
		__shared__ FOUR_VECTOR rB_shared_1[200];
		// __shared__ FOUR_VECTOR rB_shared_2[200];
		// __shared__ FOUR_VECTOR rB_shared_3[200];
		__shared__ tested_type qB_shared_1[200];
		// __shared__ tested_type qB_shared_2[200];
		// __shared__ tested_type qB_shared_3[200];

		// common
		tested_type r2;
		tested_type u2;
		tested_type vij;
		tested_type fs;
		tested_type fxij;
		tested_type fyij;
		tested_type fzij;
		THREE_VECTOR d;

		//-------------------------------------------------------------
		//	Home box
		//-------------------------------------------------------------

		//-------------------------------------------------------------
		//	Setup parameters
		//-------------------------------------------------------------

		// home box - box parameters
		first_i =read_voter(d_box_gpu_1, d_box_gpu_2, d_box_gpu_3, bx).offset;

		// home box - distance, force, charge and type parameters
		rA_1 = &d_rv_gpu_1[first_i]; 
		rA_2 = &d_rv_gpu_2[first_i];
		rA_3 = &d_rv_gpu_3[first_i];

		fA_1 = &d_fv_gpu_1[first_i];
		fA_2 = &d_fv_gpu_2[first_i];
		fA_3 = &d_fv_gpu_3[first_i];

		//-------------------------------------------------------------
		//	Copy to shared memory
		//-------------------------------------------------------------

		// home box - shared memory
		while(wtx<NUMBER_PAR_PER_BOX) {
			rA_shared_1[wtx] = read_voter(rA_1, rA_2, rA_3, wtx);
			// rA_shared_2[wtx] = read_voter(rA_1, rA_2, rA_3, wtx);
			// rA_shared_3[wtx] = read_voter(rA_1, rA_2, rA_3, wtx);
			wtx = wtx + NUMBER_THREADS;
		}
		wtx = tx;

		// synchronize threads  - not needed, but just to be safe
		__syncthreads();

		//-------------------------------------------------------------
		//	nei box loop
		//-------------------------------------------------------------

		// loop over neiing boxes of home box
		for (k=0; k<(1+read_voter(d_box_gpu_1, d_box_gpu_2, d_box_gpu_3, bx).nn); k++) {

			//---------------------------------------------
			//	nei box - get pointer to the right box
			//---------------------------------------------

			if(k==0) {
				pointer = bx;	 // set first box to be processed to home box
			}
			else {
								 // remaining boxes are nei boxes
				pointer = read_voter(d_box_gpu_1, d_box_gpu_2, d_box_gpu_3, bx).nei[k-1].number;
			}

			//-----------------------------------------------------
			//	Setup parameters
			//-----------------------------------------------------

			// nei box - box parameters
			first_j = read_voter(d_box_gpu_1, d_box_gpu_2, d_box_gpu_3, pointer).offset;

			// nei box - distance, (force), charge and (type) parameters
			rB_1 = &d_rv_gpu_1[first_j];
			rB_2 = &d_rv_gpu_2[first_j];
			rB_3 = &d_rv_gpu_3[first_j];
			
			qB_1 = &d_qv_gpu_1[first_j];
			qB_2 = &d_qv_gpu_2[first_j];
			qB_3 = &d_qv_gpu_3[first_j];

			//-----------------------------------------------------
			//	Setup parameters
			//-----------------------------------------------------

			// nei box - shared memory
			while(wtx<NUMBER_PAR_PER_BOX) {
				rB_shared_1[wtx] = read_voter(rB_1, rB_2, rB_3, wtx);
				// rB_shared_2[wtx] = read_voter(rB_1, rB_2, rB_3, wtx);
				// rB_shared_3[wtx] = read_voter(rB_1, rB_2, rB_3, wtx);

				qB_shared_1[wtx] =  read_voter(qB_1, qB_2, qB_3,wtx);
				// qB_shared_2[wtx] =  read_voter(qB_1, qB_2, qB_3,wtx);
				// qB_shared_3[wtx] =  read_voter(qB_1, qB_2, qB_3,wtx);
				wtx = wtx + NUMBER_THREADS;
			}
			wtx = tx;

			// synchronize threads because in next section each thread accesses data brought in by different threads here
			__syncthreads();

			//-----------------------------------------------------
			//	Calculation
			//-----------------------------------------------------
			// Caching
			// register FOUR_VECTOR r2_rA_shared_cached_WTX; // safe
			// register FOUR_VECTOR h2_rB_shared_cached_J; // safe
			// register tested_type h2_qB_shared_cached_J; // safe

			// loop for the number of particles in the home box
			// for (int i=0; i<nTotal_i; i++){
			while(wtx<NUMBER_PAR_PER_BOX) {

				// r2_rA_shared_cached_WTX = read_voter(rA_shared_1, rA_shared_2, rA_shared_3, wtx);

				// loop for the number of particles in the current nei box
				for (j=0; j<NUMBER_PAR_PER_BOX; j++) {

					// h2_rB_shared_cached_J = read_voter(rB_shared_1, rB_shared_2, rB_shared_3, j);

					r2 = 	rA_shared_1[wtx].v
							+ 
							rB_shared_1[j].v
							- 
							DOT( 	
								rA_shared_1[wtx],
								rB_shared_1[j]
							);

					u2 = a2*r2;
#if defined(PRECISION_DOUBLE) or defined(PRECISION_SINGLE)
					vij= exp(-u2);
#elif defined(PRECISION_HALF)
					vij= hexp(-u2);
#endif
					fs = tested_type(2.0)*vij;

					d.x = 	rA_shared_1[wtx].x  
							- 
							rB_shared_1[j].x;

					fxij=fs*d.x;

					d.y = 	rA_shared_1[wtx].y  
							- 
							rB_shared_1[j].y;

					fyij=fs*d.y;

					d.z = 	rA_shared_1[wtx].z  
							- 
							rB_shared_1[j].z;
							
					fzij=fs*d.z;

					// h2_qB_shared_cached_J = (tested_type)read_voter(qB_shared_1, qB_shared_2, qB_shared_3, j);

					fA_1[wtx].v +=  (tested_type)(
									qB_shared_1[j]
									*
									vij);
					fA_2[wtx].v +=  (tested_type)(
									qB_shared_1[j]
									*
									vij);
					fA_3[wtx].v +=  (tested_type)(
									qB_shared_1[j]
									*
									vij);
									

					fA_1[wtx].x +=  (tested_type)(
									qB_shared_1[j]
									*
									fxij);
					fA_2[wtx].x +=  (tested_type)(
									qB_shared_1[j]
									*
									fxij);
					fA_3[wtx].x +=  (tested_type)(
									qB_shared_1[j]
									*
									fxij);


					fA_1[wtx].y +=  (tested_type)(
									qB_shared_1[j]
									*
									fyij);
					fA_2[wtx].y +=  (tested_type)(
									qB_shared_1[j]
									*
									fyij);
					fA_3[wtx].y +=  (tested_type)(
									qB_shared_1[j]
									*
									fyij);


					fA_1[wtx].z +=  (tested_type)(
									qB_shared_1[j]
									*
									fzij);
					fA_2[wtx].z +=  (tested_type)(
									qB_shared_1[j]
									*
									fzij);
					fA_3[wtx].z +=  (tested_type)(
									qB_shared_1[j]
									*
									fzij);
				}
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
// #elif defined(PRECISION_HALF)
// __device__ inline void covert_FV_to_FVH2(FOUR_VECTOR fv, FOUR_H2_VECTOR *fv_h2) {
// 	(fv_h2 -> x).x = fv.x;
// 	(fv_h2 -> x).y = fv.x;

// 	(fv_h2 -> y).x = fv.y;
// 	(fv_h2 -> y).y = fv.y;

// 	(fv_h2 -> z).x = fv.z;
// 	(fv_h2 -> z).y = fv.z;

// 	(fv_h2 -> v).x = fv.v;
// 	(fv_h2 -> v).y = fv.v;
// }
// __global__ void kernel_gpu_cuda(par_str d_par_gpu, 
// 								dim_str d_dim_gpu, 
// 								box_str* d_box_gpu_1, box_str* d_box_gpu_2, box_str* d_box_gpu_3, 
// 								FOUR_VECTOR* d_rv_gpu_1, FOUR_VECTOR* d_rv_gpu_2, FOUR_VECTOR* d_rv_gpu_3, 
// 								tested_type* d_qv_gpu_1, tested_type* d_qv_gpu_2, tested_type* d_qv_gpu_3, 
// 								FOUR_VECTOR* d_fv_gpu_1, FOUR_VECTOR* d_fv_gpu_2, FOUR_VECTOR* d_fv_gpu_3) {

// 	//---------------------------------------------------------------------
// 	//	THREAD PARAMETERS
// 	//---------------------------------------------------------------------

// 	int bx = blockIdx.x;		 // get current horizontal block index (0-n)
// 	int tx = threadIdx.x;		 // get current horizontal thread index (0-n)
// 	int wtx = tx;

// 	//---------------------------------------------------------------------
// 	//	DO FOR THE NUMBER OF BOXES
// 	//---------------------------------------------------------------------

// 	if(bx<d_dim_gpu.number_boxes) {

// 		//-------------------------------------------------------------
// 		//	Extract input parameters
// 		//-------------------------------------------------------------

// 		// parameters
// 		half a2 = __float2half(2.0) * d_par_gpu.alpha * d_par_gpu.alpha;
// 		half2 h2_a2 = __half2half2(a2);

// 		// home box
// 		int first_i;
// 		FOUR_VECTOR *rA_1, *rA_2, *rA_3;
// 		FOUR_VECTOR *fA_1, *fA_2, *fA_3;
// 		__shared__ FOUR_H2_VECTOR h2_rA_shared_1[200];
// 		__shared__ FOUR_H2_VECTOR h2_rA_shared_2[200];
// 		__shared__ FOUR_H2_VECTOR h2_rA_shared_3[200];

// 		// nei box
// 		int pointer;
// 		int k = 0;
// 		int first_j;
// 		FOUR_VECTOR *rB_1, *rB_2, *rB_3;
// 		tested_type *qB_1, *qB_2, *qB_3;
// 		int j = 0;
// 		__shared__ FOUR_H2_VECTOR h2_rB_shared_1[100];
// 		__shared__ FOUR_H2_VECTOR h2_rB_shared_2[100];
// 		__shared__ FOUR_H2_VECTOR h2_rB_shared_3[100];
// 		__shared__ half2 h2_qB_shared_1[100];
// 		__shared__ half2 h2_qB_shared_2[100];
// 		__shared__ half2 h2_qB_shared_3[100];

// 		// common
// 		half2 r2;
// 		half2 u2;
// 		half2 vij;
// 		half2 fs;
// 		half2 fxij;
// 		half2 fyij;
// 		half2 fzij;
// 		THREE_H2_VECTOR d;

// 		//-------------------------------------------------------------
// 		//	Home box
// 		//-------------------------------------------------------------

// 		//-------------------------------------------------------------
// 		//	Setup parameters
// 		//-------------------------------------------------------------

// 		// home box - box parameters
// 		first_i =read_voter(d_box_gpu_1, d_box_gpu_2, d_box_gpu_3, bx).offset;

// 		// home box - distance, force, charge and type parameters
// 		rA_1 = &d_rv_gpu_1[first_i]; 
// 		rA_2 = &d_rv_gpu_2[first_i];
// 		rA_3 = &d_rv_gpu_3[first_i];

// 		fA_1 = &d_fv_gpu_1[first_i];
// 		fA_2 = &d_fv_gpu_2[first_i];
// 		fA_3 = &d_fv_gpu_3[first_i];

// 		//-------------------------------------------------------------
// 		//	Copy to shared memory
// 		//-------------------------------------------------------------

// 		// home box - shared memory - INCLUDES HALF2 transformation -redundant- on shared memory
// 		while(wtx<NUMBER_PAR_PER_BOX) {
// 			covert_FV_to_FVH2(read_voter(rA_1, rA_2, rA_3, wtx), &(h2_rA_shared_1[wtx]));
// 			covert_FV_to_FVH2(read_voter(rA_1, rA_2, rA_3, wtx), &(h2_rA_shared_2[wtx]));
// 			covert_FV_to_FVH2(read_voter(rA_1, rA_2, rA_3, wtx), &(h2_rA_shared_3[wtx]));
// 			wtx = wtx + NUMBER_THREADS;
// 		}
// 		wtx = tx;

// 		// synchronize threads  - not needed, but just to be safe
// 		__syncthreads();

// 		//-------------------------------------------------------------
// 		//	nei box loop
// 		//-------------------------------------------------------------

// 		// loop over neiing boxes of home box
// 		for (k=0; k<(1+read_voter(d_box_gpu_1, d_box_gpu_2, d_box_gpu_3, bx).nn); k++) {

// 			//---------------------------------------------
// 			//	nei box - get pointer to the right box
// 			//---------------------------------------------

// 			if(k==0) {
// 				pointer = bx;	 // set first box to be processed to home box
// 			}
// 			else {
// 								 // remaining boxes are nei boxes
// 				pointer = read_voter(d_box_gpu_1, d_box_gpu_2, d_box_gpu_3, bx).nei[k-1].number;
// 			}

// 			//-----------------------------------------------------
// 			//	Setup parameters
// 			//-----------------------------------------------------

// 			// nei box - box parameters
// 			first_j = read_voter(d_box_gpu_1, d_box_gpu_2, d_box_gpu_3, pointer).offset;

// 			// nei box - distance, (force), charge and (type) parameters
// 			rB_1 = &d_rv_gpu_1[first_j];
// 			rB_2 = &d_rv_gpu_2[first_j];
// 			rB_3 = &d_rv_gpu_3[first_j];
			
// 			qB_1 = &d_qv_gpu_1[first_j];
// 			qB_2 = &d_qv_gpu_2[first_j];
// 			qB_3 = &d_qv_gpu_3[first_j];

// 			//-----------------------------------------------------
// 			//	Setup parameters
// 			//-----------------------------------------------------

// 			// nei box - shared memory - INCLUDES HALF2 optimization on shared memory
// 			int corrWTX;
// 			register FOUR_VECTOR cached_FV_FIRST;
// 			register FOUR_VECTOR cached_FV_SECOND;
// 			register half cached_value_FIRST;
// 			register half cached_value_SECOND;
// 			while(wtx<NUMBER_PAR_PER_BOX) {
// 				corrWTX = floor(wtx / 2.0);

// 				cached_FV_FIRST = read_voter(rB_1, rB_2, rB_3, wtx + 0);
// 				cached_FV_SECOND = read_voter(rB_1, rB_2, rB_3, wtx + NUMBER_THREADS);

// 				h2_rB_shared_1[corrWTX].x.x = cached_FV_FIRST.x;
// 				h2_rB_shared_2[corrWTX].x.x = cached_FV_FIRST.x;
// 				h2_rB_shared_3[corrWTX].x.x = cached_FV_FIRST.x;
// 				h2_rB_shared_1[corrWTX].x.y = cached_FV_SECOND.x;
// 				h2_rB_shared_2[corrWTX].x.y = cached_FV_SECOND.x;
// 				h2_rB_shared_3[corrWTX].x.y = cached_FV_SECOND.x;
				
// 				h2_rB_shared_1[corrWTX].y.x = cached_FV_FIRST.y;
// 				h2_rB_shared_2[corrWTX].y.x = cached_FV_FIRST.y;
// 				h2_rB_shared_3[corrWTX].y.x = cached_FV_FIRST.y;
// 				h2_rB_shared_1[corrWTX].y.y = cached_FV_SECOND.y;
// 				h2_rB_shared_2[corrWTX].y.y = cached_FV_SECOND.y;
// 				h2_rB_shared_3[corrWTX].y.y = cached_FV_SECOND.y;
				
// 				h2_rB_shared_1[corrWTX].z.x = cached_FV_FIRST.z;
// 				h2_rB_shared_2[corrWTX].z.x = cached_FV_FIRST.z;
// 				h2_rB_shared_3[corrWTX].z.x = cached_FV_FIRST.z;
// 				h2_rB_shared_1[corrWTX].z.y = cached_FV_SECOND.z;
// 				h2_rB_shared_2[corrWTX].z.y = cached_FV_SECOND.z;
// 				h2_rB_shared_3[corrWTX].z.y = cached_FV_SECOND.z;

// 				h2_rB_shared_1[corrWTX].v.x = cached_FV_FIRST.v;
// 				h2_rB_shared_2[corrWTX].v.x = cached_FV_FIRST.v;
// 				h2_rB_shared_3[corrWTX].v.x = cached_FV_FIRST.v;
// 				h2_rB_shared_1[corrWTX].v.y = cached_FV_SECOND.v;
// 				h2_rB_shared_2[corrWTX].v.y = cached_FV_SECOND.v;
// 				h2_rB_shared_3[corrWTX].v.y = cached_FV_SECOND.v;
				
// 				cached_value_FIRST = read_voter(qB_1, qB_2, qB_3, wtx + 0);
// 				cached_value_SECOND = read_voter(qB_1, qB_2, qB_3, wtx + NUMBER_THREADS);

// 				h2_qB_shared_1[corrWTX].x = cached_value_FIRST;
// 				h2_qB_shared_2[corrWTX].x = cached_value_FIRST;
// 				h2_qB_shared_3[corrWTX].x = cached_value_FIRST;
// 				h2_qB_shared_1[corrWTX].y = cached_value_SECOND;
// 				h2_qB_shared_2[corrWTX].y = cached_value_SECOND;
// 				h2_qB_shared_3[corrWTX].y = cached_value_SECOND;

// 				wtx = wtx + NUMBER_THREADS * 2.0;
// 			}
// 			wtx = tx;

// 			// synchronize threads because in next section each thread accesses data brought in by different threads here
// 			__syncthreads();

// 			//-----------------------------------------------------
// 			//	Calculation
// 			//-----------------------------------------------------

// 			// Common
// 			register FOUR_H2_VECTOR h2_fA_local;

// 			// Caching
// 			register FOUR_H2_VECTOR r2_rA_shared_cached_WTX; // safe
// 			register FOUR_H2_VECTOR h2_rB_shared_cached_J; // safe
// 			register half2 h2_qB_shared_cached_J; // safe

// 			register half add_cache;

// 			// loop for the number of particles in the home box
// 			// for (int i=0; i<nTotal_i; i++){
// 			while(wtx<NUMBER_PAR_PER_BOX) {

// 				h2_fA_local.x = __float2half2_rn(0.0);
// 				h2_fA_local.y = __float2half2_rn(0.0);
// 				h2_fA_local.z = __float2half2_rn(0.0);
// 				h2_fA_local.v = __float2half2_rn(0.0);

// 				r2_rA_shared_cached_WTX = read_voter(h2_rA_shared_1, h2_rA_shared_2, h2_rA_shared_3, wtx);
				
// 				// loop for the number of particles in the current nei box
// 				for (j=0; j<floor(NUMBER_PAR_PER_BOX / 2.0); j++) {
// 					// Convert input vars from HALF to HALF2 for local work

// 					h2_rB_shared_cached_J = read_voter(h2_rB_shared_1, h2_rB_shared_2, h2_rB_shared_3, j);

// 					// r2 = (half)h2_rA_shared[wtx].v + (half)h2_rB_shared[j].v - H_DOT((half)h2_rA_shared[wtx],(half)h2_rB_shared[j]);
// 					r2 = __hsub2(
// 							__hadd2(
// 								r2_rA_shared_cached_WTX.v, 
// 								h2_rB_shared_cached_J.v),  
// 							H2_DOT(
// 								r2_rA_shared_cached_WTX, 
// 								h2_rB_shared_cached_J));

// 					// u2 = a2*r2;
// 					u2 = __hmul2(h2_a2, r2);
// 					// vij= exp(-u2);
// 					vij= h2exp(__hneg2(u2));
// 					// fs = 2*vij;
// 					fs = __hmul2(__float2half2_rn(2.0), vij);

// 					// d.x = (half)h2_rA_shared[wtx].x  - (half)h2_rB_shared[j].x;
// 					d.x = __hsub2(
// 							r2_rA_shared_cached_WTX.x, 
// 							h2_rB_shared_cached_J.x);
// 					// fxij=fs*d.x;
// 					fxij=__hmul2(fs, d.x);
// 					// d.y = (half)h2_rA_shared[wtx].y  - (half)h2_rB_shared[j].y;
// 					d.y = __hsub2(
// 							r2_rA_shared_cached_WTX.y, 
// 							h2_rB_shared_cached_J.y);
// 					// fyij=fs*d.y;
// 					fyij=__hmul2(fs, d.y);
// 					// d.z = (half)h2_rA_shared[wtx].z  - (half)h2_rB_shared[j].z;
// 					d.z = __hsub2(
// 							r2_rA_shared_cached_WTX.z, 
// 							h2_rB_shared_cached_J.z);
// 					// fzij=fs*d.z;
// 					fzij=__hmul2(fs, d.z);

// 					h2_qB_shared_cached_J = read_voter(h2_qB_shared_1, h2_qB_shared_2, h2_qB_shared_3, j);

// 					// fA[wtx].v +=  (half)((half)h2_qB_shared[j]*vij);
// 					h2_fA_local.v = __hfma2(
// 							h2_qB_shared_cached_J, 
// 							vij, 
// 							h2_fA_local.v);
// 					// fA[wtx].x +=  (half)((half)h2_qB_shared[j]*fxij);
// 					h2_fA_local.x = __hfma2(
// 							h2_qB_shared_cached_J, 
// 							fxij, 
// 							h2_fA_local.x);
// 					// fA[wtx].y +=  (half)((half)h2_qB_shared[j]*fyij);
// 					h2_fA_local.y = __hfma2(
// 							h2_qB_shared_cached_J, 
// 							fyij, 
// 							h2_fA_local.y);
// 					// fA[wtx].z +=  (half)((half)h2_qB_shared[j]*fzij);
// 					h2_fA_local.z = __hfma2(
// 							h2_qB_shared_cached_J, 
// 							fzij, 
// 							h2_fA_local.z);
// 				}

// 				// Copy back data from local memory to global memory
// 				add_cache = __hadd(h2_fA_local.x.x, h2_fA_local.x.y);
// 				fA_1[wtx].x = __hadd(fA_1[wtx].x, add_cache);
// 				fA_2[wtx].x = __hadd(fA_2[wtx].x, add_cache);
// 				fA_3[wtx].x = __hadd(fA_3[wtx].x, add_cache);

// 				add_cache = __hadd(h2_fA_local.y.x, h2_fA_local.y.y);
// 				fA_1[wtx].y = __hadd(fA_1[wtx].y, add_cache);
// 				fA_2[wtx].y = __hadd(fA_2[wtx].y, add_cache);
// 				fA_3[wtx].y = __hadd(fA_3[wtx].y, add_cache);

// 				add_cache = __hadd(h2_fA_local.z.x, h2_fA_local.z.y);
// 				fA_1[wtx].z = __hadd(fA_1[wtx].z, add_cache);
// 				fA_2[wtx].z = __hadd(fA_2[wtx].z, add_cache);
// 				fA_3[wtx].z = __hadd(fA_3[wtx].z, add_cache);

// 				add_cache = __hadd(h2_fA_local.v.x, h2_fA_local.v.y);
// 				fA_1[wtx].v = __hadd(fA_1[wtx].v, add_cache);
// 				fA_2[wtx].v = __hadd(fA_2[wtx].v, add_cache);
// 				fA_3[wtx].v = __hadd(fA_3[wtx].v, add_cache);

// 				// increment work thread index
// 				wtx = wtx + NUMBER_THREADS;
// 			}

// 			// reset work index
// 			wtx = tx;

// 			// synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
// 			__syncthreads();

// 			//----------------------------------------------------------------------------------------------------------------------------------140
// 			//	Calculation END
// 			//----------------------------------------------------------------------------------------------------------------------------------140
// 		}
// 		//------------------------------------------------------------------------------------------------------------------------------------------------------160
// 		//	nei box loop END
// 		//------------------------------------------------------------------------------------------------------------------------------------------------------160
// 	}
// }
// #endif

double mysecond()
{
   struct timeval tp;
   struct timezone tzp;
   int i = gettimeofday(&tp,&tzp);
   return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void generateInput(dim_str dim_cpu, char *input_distances, FOUR_VECTOR_HOST **rv_cpu, char *input_charges, tested_type_host **qv_cpu)
{
	// random generator seed set to random value - time in this case
	FILE *fp;
	int i;

	srand(time(NULL));

	// input (distances)
	if( (fp = fopen(input_distances, "wb" )) == 0 ) {
		printf( "The file 'input_distances' was not opened\n" ); exit(EXIT_FAILURE);
	}
	*rv_cpu = (FOUR_VECTOR_HOST*)malloc(dim_cpu.space_mem);
	for(i=0; i<dim_cpu.space_elem; i=i+1) {
								 // get a number in the range 0.1 - 1.0
		(*rv_cpu)[i].v = (tested_type_host)(rand()%10 + 1) / 10.0;
		fwrite(&((*rv_cpu)[i].v), 1, sizeof(tested_type), fp);
								 // get a number in the range 0.1 - 1.0
		(*rv_cpu)[i].x = (tested_type_host)(rand()%10 + 1) / 10.0;
		fwrite(&((*rv_cpu)[i].x), 1, sizeof(tested_type), fp);
								 // get a number in the range 0.1 - 1.0
		(*rv_cpu)[i].y = (tested_type_host)(rand()%10 + 1) / 10.0;
		fwrite(&((*rv_cpu)[i].y), 1, sizeof(tested_type), fp);
								 // get a number in the range 0.1 - 1.0
		(*rv_cpu)[i].z = (tested_type_host)(rand()%10 + 1) / 10.0;
		fwrite(&((*rv_cpu)[i].z), 1, sizeof(tested_type), fp);
	}
	fclose(fp);

	// input (charge)
	if( (fp = fopen(input_charges, "wb" )) == 0 ) {
		printf( "The file 'input_charges' was not opened\n" ); exit(EXIT_FAILURE);
	}

	*qv_cpu = (tested_type_host*)malloc(dim_cpu.space_mem2);
	for(i=0; i<dim_cpu.space_elem; i=i+1) {
								 // get a number in the range 0.1 - 1.0
		(*qv_cpu)[i] = (tested_type_host)(rand()%10 + 1) / 10.0;
		fwrite(&((*qv_cpu)[i]), 1, sizeof(tested_type), fp);
	}
	fclose(fp);
}

void readInput(dim_str dim_cpu, char *input_distances, FOUR_VECTOR_HOST **rv_cpu, char *input_charges, tested_type_host **qv_cpu, int fault_injection)
{
	FILE *fp;
	int i;
	size_t return_value[4];

	// input (distances)
	if( (fp = fopen(input_distances, "rb" )) == 0 ) {
		printf( "The file 'input_distances' was not opened\n" ); exit(EXIT_FAILURE);
	}

	*rv_cpu = (FOUR_VECTOR_HOST*)malloc(dim_cpu.space_mem);
	if(*rv_cpu == NULL) {
		printf("error rv_cpu malloc\n");
		#ifdef LOGS
			log_error_detail((char *)"error rv_cpu malloc"); end_log_file();
		#endif
		exit(1);
	}
	for(i=0; i<dim_cpu.space_elem; i=i+1) {
		return_value[0] = fread(&((*rv_cpu)[i].v), 1, sizeof(tested_type), fp);
		return_value[1] = fread(&((*rv_cpu)[i].x), 1, sizeof(tested_type), fp);
		return_value[2] = fread(&((*rv_cpu)[i].y), 1, sizeof(tested_type), fp);
		return_value[3] = fread(&((*rv_cpu)[i].z), 1, sizeof(tested_type), fp);
		if (return_value[0] == 0 || return_value[1] == 0 || return_value[2] == 0 || return_value[3] == 0) {
			printf("error reading rv_cpu from file\n");
			#ifdef LOGS
				log_error_detail((char *)"error reading rv_cpu from file"); end_log_file();
			#endif
			exit(1);
		}
	}
	fclose(fp);

	// input (charge)
	if( (fp = fopen(input_charges, "rb" )) == 0 ) {
		printf( "The file 'input_charges' was not opened\n" ); exit(EXIT_FAILURE);
	}

	*qv_cpu = (tested_type_host*)malloc(dim_cpu.space_mem2);
	if(*qv_cpu == NULL) {
		printf("error qv_cpu malloc\n");
		#ifdef LOGS
			log_error_detail((char *)"error qv_cpu malloc"); end_log_file();
		#endif
		exit(1);
	}
	for(i=0; i<dim_cpu.space_elem; i=i+1) {
		return_value[0] = fread(&((*qv_cpu)[i]), 1, sizeof(tested_type), fp);
		if (return_value[0] == 0) {
			printf("error reading qv_cpu from file\n");
			#ifdef LOGS
				log_error_detail((char *)"error reading qv_cpu from file"); end_log_file();
			#endif
			exit(1);
		}
	}
	fclose(fp);

	// =============== Fault injection
	if (fault_injection) {
		(*qv_cpu)[2] = 0.732637263; // must be in range 0.1 - 1.0
		printf("!!> Fault injection: qv_cpu[2]=%f\n", (double)(*qv_cpu)[2]);
	}
	// ========================
}

void readGold(dim_str dim_cpu, char *output_gold, FOUR_VECTOR_HOST *fv_cpu_GOLD)
{
	FILE *fp;
	size_t return_value[4];
	int i;

	if( (fp = fopen(output_gold, "rb" )) == 0 )
	{
		printf( "The file 'output_forces' was not opened\n" ); exit(EXIT_FAILURE);
	}
	for(i=0; i<dim_cpu.space_elem; i=i+1) {
		return_value[0] = fread(&(fv_cpu_GOLD[i].v), 1, sizeof(tested_type), fp);
		return_value[1] = fread(&(fv_cpu_GOLD[i].x), 1, sizeof(tested_type), fp);
		return_value[2] = fread(&(fv_cpu_GOLD[i].y), 1, sizeof(tested_type), fp);
		return_value[3] = fread(&(fv_cpu_GOLD[i].z), 1, sizeof(tested_type), fp);
		if (return_value[0] == 0 || return_value[1] == 0 || return_value[2] == 0 || return_value[3] == 0) {
			printf("error reading rv_cpu from file\n");
			#ifdef LOGS
				log_error_detail((char *)"error reading rv_cpu from file"); end_log_file();
			#endif
			exit(1);
		}
	}
	fclose(fp);
}

void writeGold(dim_str dim_cpu, char *output_gold, FOUR_VECTOR_HOST **fv_cpu)
{
	FILE *fp;
	int i;

	if( (fp = fopen(output_gold, "wb" )) == 0 ) {
		printf( "The file 'output_forces' was not opened\n" ); exit(EXIT_FAILURE);
	}
	int number_zeros = 0;
	for(i=0; i<dim_cpu.space_elem; i=i+1) {
		if((*fv_cpu)[i].v == tested_type_host(0.0))
			number_zeros++;
		if((*fv_cpu)[i].x == tested_type_host(0.0))
			number_zeros++;
		if((*fv_cpu)[i].y == tested_type_host(0.0))
			number_zeros++;
		if((*fv_cpu)[i].z == tested_type_host(0.0))
			number_zeros++;

		fwrite(&((*fv_cpu)[i].v), 1, sizeof(tested_type), fp);
		fwrite(&((*fv_cpu)[i].x), 1, sizeof(tested_type), fp);
		fwrite(&((*fv_cpu)[i].y), 1, sizeof(tested_type), fp);
		fwrite(&((*fv_cpu)[i].z), 1, sizeof(tested_type), fp);
	}
	fclose(fp);
}

// Returns true if no errors are found. False if otherwise.
// Set votedOutput pointer to retrieve the voted matrix
bool checkOutputErrors(	dim_str dim_cpu, int streamIdx,
						FOUR_VECTOR_HOST* fv_cpu_1, FOUR_VECTOR_HOST* fv_cpu_2, FOUR_VECTOR_HOST* fv_cpu_3, 
						FOUR_VECTOR_HOST* fv_cpu_GOLD, 
						FOUR_VECTOR_HOST* votedOutput,
						unsigned long long int host_is_memory_bad, 
						bool check = true, bool verbose = false) {
	int host_errors = 0;
	int memory_errors = 0;

	if (host_is_memory_bad != 0) {
		char info_detail[150];
		snprintf(info_detail, 150,
				"b: is_memory_bad: %llu",
				host_is_memory_bad);
		if (verbose)
			printf("%s\n", info_detail);

#ifdef LOGS
		if (check) 
			log_info_detail(info_detail);
#endif
		memory_errors++;
	} 

#pragma omp parallel for shared(host_errors)
	for (int i=0; i<dim_cpu.space_elem; i=i+1) {
		register FOUR_VECTOR_HOST valGold = fv_cpu_GOLD[i];
		register FOUR_VECTOR_HOST valOutput1 = fv_cpu_1[i];
		register FOUR_VECTOR_HOST valOutput2 = fv_cpu_2[i];
		register FOUR_VECTOR_HOST valOutput3 = fv_cpu_3[i];
		register FOUR_VECTOR_HOST valOutput = valOutput1;
		if ((valOutput1 != valOutput2) || (valOutput1 != valOutput3)) {
			#pragma omp critical
			{
				char info_detail[500];
				snprintf(info_detail, 500, "m: [%d], stream: %d, v1: %1.20e, v2: %1.20e, v3: %1.20e, vE: %1.20e, x1: %1.20e, x2: %1.20e, x3: %1.20e, xE: %1.20e, y1: %1.20e, y2: %1.20e, y3: %1.20e, yE: %1.20e, z1: %1.20e, z2: %1.20e, z3: %1.20e, zE: %1.20e\n", 
					i, streamIdx, 
					(double)valOutput1.v, (double)valOutput2.v, (double)valOutput3.v, (double)valGold.v,
					(double)valOutput1.x, (double)valOutput2.x, (double)valOutput3.x, (double)valGold.x,
					(double)valOutput1.y, (double)valOutput2.y, (double)valOutput3.y, (double)valGold.y,
					(double)valOutput1.z, (double)valOutput2.z, (double)valOutput3.z, (double)valGold.z);
				if (verbose && (memory_errors < 10))
					printf("%s\n", info_detail);

#ifdef LOGS
				if ((check) && (memory_errors < MAX_LOGGED_INFOS_PER_STREAM)) 
					log_info_detail(info_detail);
#endif
				memory_errors++;
			}
			if ((valOutput1 != valOutput2) && (valOutput2 != valOutput3) && (valOutput1 != valOutput3)) {
				// All 3 values diverge
				if (valOutput1 == valGold) {
					valOutput = valOutput1;
				} else if (valOutput2 == valGold) {
					valOutput = valOutput2;
				} else if (valOutput3 == valGold) {
					valOutput = valOutput3;
				} else {
					// NO VALUE MATCHES THE GOLD AND ALL 3 DIVERGE!
					printf("&");
				}
			} else if (valOutput2 == valOutput3) {
				// Only value 0 diverge
				valOutput = valOutput2;
			} else if (valOutput1 == valOutput3) {
				// Only value 1 diverge
				valOutput = valOutput1;
			} else if (valOutput1 == valOutput2) {
				// Only value 2 diverge
				valOutput = valOutput1;
			}
		}
		if (votedOutput != NULL) 
			votedOutput[i] = valOutput;
		// if ((fabs((tested_type_host)(valOutput-valGold)/valGold) > 1e-10)||(fabs((tested_type_host)(valOutput-valGold)/valGold) > 1e-10)) {
		if (check) {
			if (valGold != valOutput) {
				#pragma omp critical
				{
					char error_detail[500];
					host_errors++;

					snprintf(error_detail, 500, "stream: %d, p: [%d], v_r: %1.20e, v_e: %1.20e, x_r: %1.20e, x_e: %1.20e, y_r: %1.20e, y_e: %1.20e, z_r: %1.20e, z_e: %1.20e\n", streamIdx, \
						i, 
						(double)valOutput.v, (double)valGold.v, 
						(double)valOutput.x, (double)valGold.x, 
						(double)valOutput.y, (double)valGold.y, 
						(double)valOutput.z, (double)valGold.z);
						if (verbose && (host_errors < 10))
							printf("%s\n", error_detail);
#ifdef LOGS
						if ((check) && (host_errors<MAX_LOGGED_INFOS_PER_STREAM)) 
							log_error_detail(error_detail);
#endif
				}
			}
		}
	}

	// printf("numErrors:%d", host_errors);

#ifdef LOGS
	if (check) {
		log_info_count(memory_errors);
		log_error_count(host_errors);
	}
#endif
	if (memory_errors != 0) printf("M");
	if (host_errors != 0) printf("#");

	return (host_errors == 0) && (host_is_memory_bad == 0);
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

	unsigned long long int host_is_memory_bad = 0;

	// system memory
	par_str par_cpu;
	dim_str dim_cpu;
	box_str *box_cpu;
	FOUR_VECTOR_HOST *rv_cpu;
	tested_type_host *qv_cpu;
	FOUR_VECTOR_HOST *fv_cpu_1, *fv_cpu_2, *fv_cpu_3;
	FOUR_VECTOR_HOST *fv_cpu_GOLD;
	int nh;
	int nstreams, streamIdx;

	char *input_distances, *input_charges, *output_gold;

	int number_nn = 0;

	//=====================================================================
	//	CHECK INPUT ARGUMENTS
	//=====================================================================

	getParams(argc, argv, &dim_cpu.boxes1d_arg, &generate, &input_distances, &input_charges, &output_gold, &iterations, &verbose, &fault_injection, &nstreams);

	char test_info[200];
	char test_name[200];
	snprintf(test_info, 200, "type:%s-precision-triplicated streams:%d boxes:%d block_size:%d", test_precision_description, nstreams, dim_cpu.boxes1d_arg, NUMBER_THREADS);
	snprintf(test_name, 200, "cuda_trip_%s_lava", test_precision_description);
	printf("%s\n", test_info);
	#ifdef LOGS
		if (!generate) {
			start_log_file(test_name, test_info);
			set_max_infos_iter(MAX_LOGGED_INFOS_PER_STREAM * nstreams + 32);
			set_max_errors_iter(MAX_LOGGED_ERRORS_PER_STREAM * nstreams + 32);
		}
	#endif

	//=====================================================================
	//	INPUTS
	//=====================================================================
	par_cpu.alpha = 0.5;
	//=====================================================================
	//	DIMENSIONS
	//=====================================================================

	// total number of boxes
	dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;

	// how many particles space has in each direction
	dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
	dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
	dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(tested_type);

	// box array
	dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

	//=====================================================================
	//	SYSTEM MEMORY
	//=====================================================================

	fv_cpu_GOLD = (FOUR_VECTOR_HOST*)malloc(dim_cpu.space_mem);
	if(fv_cpu_GOLD == NULL) {
		printf("error fv_cpu_GOLD malloc\n");
		#ifdef LOGS
			log_error_detail((char *)"error fv_cpu_GOLD malloc"); end_log_file();
		#endif
		exit(1);
	}

	//=====================================================================
	//	BOX
	//=====================================================================

	// allocate boxes
	box_cpu = (box_str*)malloc(dim_cpu.box_mem);
	if(box_cpu == NULL) {
		printf("error box_cpu malloc\n");
		#ifdef LOGS
			if (!generate) log_error_detail((char *)"error box_cpu malloc"); end_log_file();
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
		readGold(dim_cpu, output_gold, fv_cpu_GOLD);
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

		host_is_memory_bad = 0;

		double globaltimer = mysecond();
		timestamp = mysecond();

		// prepare host memory to receive kernel output
		// output (forces)
		fv_cpu_1 = (FOUR_VECTOR_HOST*)malloc(dim_cpu.space_mem);
		fv_cpu_2 = (FOUR_VECTOR_HOST*)malloc(dim_cpu.space_mem);
		fv_cpu_3 = (FOUR_VECTOR_HOST*)malloc(dim_cpu.space_mem);
		if(fv_cpu_1 == NULL || fv_cpu_2 == NULL || fv_cpu_3 == NULL) {
			printf("error fv_cpu malloc\n");
			#ifdef LOGS
				if (!generate) log_error_detail((char *)"error fv_cpu malloc"); end_log_file();
			#endif
			exit(1);
		}
		for(i=0; i<dim_cpu.space_elem; i=i+1) {
			// set to 0, because kernels keeps adding to initial value
			fv_cpu_1[i].v = tested_type_host(0.0);
			fv_cpu_1[i].x = tested_type_host(0.0);
			fv_cpu_1[i].y = tested_type_host(0.0);
			fv_cpu_1[i].z = tested_type_host(0.0);

			fv_cpu_2[i] = fv_cpu_1[i];
			fv_cpu_3[i] = fv_cpu_1[i];
		}

		//=====================================================================
		//	GPU_CUDA
		//=====================================================================

		//=====================================================================
		//	VARIABLES
		//=====================================================================

		box_str *d_box_gpu_1[nstreams], *d_box_gpu_2[nstreams], *d_box_gpu_3[nstreams];
		FOUR_VECTOR *d_rv_gpu_1[nstreams], *d_rv_gpu_2[nstreams], *d_rv_gpu_3[nstreams];
		tested_type *d_qv_gpu_1[nstreams], *d_qv_gpu_2[nstreams], *d_qv_gpu_3[nstreams];
		FOUR_VECTOR *d_fv_gpu_1[nstreams], *d_fv_gpu_2[nstreams], *d_fv_gpu_3[nstreams];


		//=====================================================================
		//	GPU SETUP
		//=====================================================================

		checkFrameworkErrors( 
			cudaMemcpyToSymbol(is_memory_bad, &host_is_memory_bad,
				sizeof(unsigned long long int), 0, cudaMemcpyHostToDevice) );

		for (streamIdx = 0; streamIdx < nstreams; streamIdx++) {
			checkFrameworkErrors( cudaStreamCreateWithFlags(&(streams[streamIdx]), cudaStreamNonBlocking) );

			//==================================================
			//	boxes
			//==================================================
#ifdef SAFE_MALLOC
			safe_cuda_malloc_cover((void **)&(d_box_gpu_1[streamIdx]), dim_cpu.box_mem);
			safe_cuda_malloc_cover((void **)&(d_box_gpu_2[streamIdx]), dim_cpu.box_mem);
			safe_cuda_malloc_cover((void **)&(d_box_gpu_3[streamIdx]), dim_cpu.box_mem);
#else
			checkFrameworkErrors( cudaMalloc( (void **)&(d_box_gpu_1[streamIdx]), dim_cpu.box_mem) );
			checkFrameworkErrors( cudaMalloc( (void **)&(d_box_gpu_2[streamIdx]), dim_cpu.box_mem) );
			checkFrameworkErrors( cudaMalloc( (void **)&(d_box_gpu_3[streamIdx]), dim_cpu.box_mem) );
#endif
			//==================================================
			//	rv
			//==================================================
#ifdef SAFE_MALLOC
			safe_cuda_malloc_cover( (void **)&(d_rv_gpu_1[streamIdx]), dim_cpu.space_mem);
			safe_cuda_malloc_cover( (void **)&(d_rv_gpu_2[streamIdx]), dim_cpu.space_mem);
			safe_cuda_malloc_cover( (void **)&(d_rv_gpu_3[streamIdx]), dim_cpu.space_mem);
#else
			checkFrameworkErrors( cudaMalloc( (void **)&(d_rv_gpu_1[streamIdx]), dim_cpu.space_mem) );
			checkFrameworkErrors( cudaMalloc( (void **)&(d_rv_gpu_2[streamIdx]), dim_cpu.space_mem) );
			checkFrameworkErrors( cudaMalloc( (void **)&(d_rv_gpu_3[streamIdx]), dim_cpu.space_mem) );
#endif
			//==================================================
			//	qv
			//==================================================
#ifdef SAFE_MALLOC
			safe_cuda_malloc_cover( (void **)&(d_qv_gpu_1[streamIdx]), dim_cpu.space_mem2);
			safe_cuda_malloc_cover( (void **)&(d_qv_gpu_2[streamIdx]), dim_cpu.space_mem2);
			safe_cuda_malloc_cover( (void **)&(d_qv_gpu_3[streamIdx]), dim_cpu.space_mem2);
#else
			checkFrameworkErrors( cudaMalloc( (void **)&(d_qv_gpu_1[streamIdx]), dim_cpu.space_mem2) );
			checkFrameworkErrors( cudaMalloc( (void **)&(d_qv_gpu_2[streamIdx]), dim_cpu.space_mem2) );
			checkFrameworkErrors( cudaMalloc( (void **)&(d_qv_gpu_3[streamIdx]), dim_cpu.space_mem2) );
#endif

			//==================================================
			//	fv
			//==================================================
#ifdef SAFE_MALLOC
			safe_cuda_malloc_cover( (void **)&(d_fv_gpu_1[streamIdx]), dim_cpu.space_mem);
			safe_cuda_malloc_cover( (void **)&(d_fv_gpu_2[streamIdx]), dim_cpu.space_mem);
			safe_cuda_malloc_cover( (void **)&(d_fv_gpu_3[streamIdx]), dim_cpu.space_mem);
#else
			checkFrameworkErrors( cudaMalloc( (void **)&(d_fv_gpu_1[streamIdx]), dim_cpu.space_mem) );
			checkFrameworkErrors( cudaMalloc( (void **)&(d_fv_gpu_2[streamIdx]), dim_cpu.space_mem) );
			checkFrameworkErrors( cudaMalloc( (void **)&(d_fv_gpu_3[streamIdx]), dim_cpu.space_mem) );
#endif

			//=====================================================================
			//	GPU MEMORY			COPY
			//=====================================================================

			//==================================================
			//	boxes
			//==================================================

			checkFrameworkErrors( cudaMemcpy(d_box_gpu_1[streamIdx], box_cpu, dim_cpu.box_mem, cudaMemcpyHostToDevice) );
			checkFrameworkErrors( cudaMemcpy(d_box_gpu_2[streamIdx], box_cpu, dim_cpu.box_mem, cudaMemcpyHostToDevice) );
			checkFrameworkErrors( cudaMemcpy(d_box_gpu_3[streamIdx], box_cpu, dim_cpu.box_mem, cudaMemcpyHostToDevice) );
			//==================================================
			//	rv
			//==================================================

			checkFrameworkErrors( cudaMemcpy( d_rv_gpu_1[streamIdx], rv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice) );
			checkFrameworkErrors( cudaMemcpy( d_rv_gpu_2[streamIdx], rv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice) );
			checkFrameworkErrors( cudaMemcpy( d_rv_gpu_3[streamIdx], rv_cpu, dim_cpu.space_mem, cudaMemcpyHostToDevice) );
			//==================================================
			//	qv
			//==================================================

			checkFrameworkErrors( cudaMemcpy( d_qv_gpu_1[streamIdx], qv_cpu, dim_cpu.space_mem2, cudaMemcpyHostToDevice) );
			checkFrameworkErrors( cudaMemcpy( d_qv_gpu_2[streamIdx], qv_cpu, dim_cpu.space_mem2, cudaMemcpyHostToDevice) );
			checkFrameworkErrors( cudaMemcpy( d_qv_gpu_3[streamIdx], qv_cpu, dim_cpu.space_mem2, cudaMemcpyHostToDevice) );
			//==================================================
			//	fv
			//==================================================

			checkFrameworkErrors( cudaMemcpy( d_fv_gpu_1[streamIdx], fv_cpu_1, dim_cpu.space_mem, cudaMemcpyHostToDevice) );
			checkFrameworkErrors( cudaMemcpy( d_fv_gpu_2[streamIdx], fv_cpu_2, dim_cpu.space_mem, cudaMemcpyHostToDevice) );
			checkFrameworkErrors( cudaMemcpy( d_fv_gpu_3[streamIdx], fv_cpu_3, dim_cpu.space_mem, cudaMemcpyHostToDevice) );
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
			kernel_gpu_cuda<<<blocks, threads, 0, streams[streamIdx]>>>
				( par_cpu, dim_cpu,
					d_box_gpu_1[streamIdx], d_box_gpu_2[streamIdx], d_box_gpu_3[streamIdx], 
					d_rv_gpu_1[streamIdx], d_rv_gpu_2[streamIdx], d_rv_gpu_3[streamIdx], 
					d_qv_gpu_1[streamIdx], d_qv_gpu_2[streamIdx], d_qv_gpu_3[streamIdx], 
					d_fv_gpu_1[streamIdx], d_fv_gpu_2[streamIdx], d_fv_gpu_3[streamIdx]);
			checkFrameworkErrors( cudaPeekAtLastError() );
		}
		//printf("All kernels were commited.\n");
		for (streamIdx = 0; streamIdx < nstreams; streamIdx++) {
			checkFrameworkErrors( cudaStreamSynchronize(streams[streamIdx]) );
			checkFrameworkErrors( cudaPeekAtLastError() );
		}
		#ifdef LOGS
			if (!generate) end_iteration();
		#endif
		kernel_time = mysecond()-kernel_time;


		//=====================================================================
		//	COMPARE OUTPUTS / WRITE GOLD
		//=====================================================================
		checkFrameworkErrors(
			cudaMemcpyFromSymbol(&host_is_memory_bad, is_memory_bad,
					sizeof(unsigned long long int), 0,
					cudaMemcpyDeviceToHost) );

		if (generate){
			checkFrameworkErrors( cudaMemcpy( fv_cpu_1, d_fv_gpu_1[0], dim_cpu.space_mem, cudaMemcpyDeviceToHost) );
			checkFrameworkErrors( cudaMemcpy( fv_cpu_2, d_fv_gpu_2[0], dim_cpu.space_mem, cudaMemcpyDeviceToHost) );
			checkFrameworkErrors( cudaMemcpy( fv_cpu_3, d_fv_gpu_3[0], dim_cpu.space_mem, cudaMemcpyDeviceToHost) );
			// TODO: Fix this RIGHT
			checkOutputErrors(	dim_cpu, 0,
								fv_cpu_1, fv_cpu_2, fv_cpu_3, 
								fv_cpu_GOLD, fv_cpu_GOLD, 
								host_is_memory_bad, false, true);
			writeGold(dim_cpu, output_gold, &fv_cpu_GOLD);
		} else { // Check gold
			timestamp = mysecond();
			bool reloadFlag = false;
			for (int streamIdx = 0; streamIdx < nstreams; streamIdx++) {
				checkFrameworkErrors( cudaMemcpy( fv_cpu_1, d_fv_gpu_1[streamIdx], dim_cpu.space_mem, cudaMemcpyDeviceToHost) );
				checkFrameworkErrors( cudaMemcpy( fv_cpu_2, d_fv_gpu_2[streamIdx], dim_cpu.space_mem, cudaMemcpyDeviceToHost) );
				checkFrameworkErrors( cudaMemcpy( fv_cpu_3, d_fv_gpu_3[streamIdx], dim_cpu.space_mem, cudaMemcpyDeviceToHost) );
				reloadFlag = reloadFlag || 
				checkOutputErrors(	dim_cpu, streamIdx,
									fv_cpu_1, fv_cpu_2, fv_cpu_3, 
									fv_cpu_GOLD, fv_cpu_GOLD, 
									host_is_memory_bad, true, true );
			}
			if (verbose) printf("[Iteration #%i] Gold check time: %f\n", loop, mysecond() - timestamp);
			if (reloadFlag) {
				readInput(dim_cpu, input_distances, &rv_cpu, input_charges, &qv_cpu, fault_injection);
				readGold(dim_cpu, output_gold, fv_cpu_GOLD);
			}
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
			cudaFree(d_rv_gpu_1[streamIdx]);
			cudaFree(d_rv_gpu_2[streamIdx]);
			cudaFree(d_rv_gpu_3[streamIdx]);
			
			cudaFree(d_qv_gpu_1[streamIdx]);
			cudaFree(d_qv_gpu_2[streamIdx]);
			cudaFree(d_qv_gpu_3[streamIdx]);
			
			cudaFree(d_fv_gpu_1[streamIdx]);
			cudaFree(d_fv_gpu_2[streamIdx]);
			cudaFree(d_fv_gpu_3[streamIdx]);
			
			cudaFree(d_box_gpu_1[streamIdx]);
			cudaFree(d_box_gpu_2[streamIdx]);
			cudaFree(d_box_gpu_3[streamIdx]);
		}

		//=====================================================================
		//	SYSTEM MEMORY DEALLOCATION
		//=====================================================================
		free(fv_cpu_1);
		free(fv_cpu_2);
		free(fv_cpu_3);

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
