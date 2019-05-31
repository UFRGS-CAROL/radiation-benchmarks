/*
 Code modified from https://github.com/ParRes
 * */
/*
 Copyright (c) 2013, Intel Corporation

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:

 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above
 copyright notice, this list of conditions and the following
 disclaimer in the documentation and/or other materials provided
 with the distribution.
 * Neither the name of Intel Corporation nor the names of its
 contributors may be used to endorse or promote products
 derived from this software without specific prior written
 permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
 */

/*********************************************************************************

 NAME:    dgemm

 PURPOSE: This program tests the efficiency with which a dense matrix
 dense multiplication is carried out

 USAGE:   The program takes as input the number of threads, the matrix
 order, the number of times the matrix-matrix multiplication
 is carried out, and, optionally, a tile size for matrix
 blocking

 <progname> <# threads> <# iterations> <matrix order> [<tile size>]

 The output consists of diagnostics to make sure the
 algorithm worked, and of timing statistics.

 HISTORY: Written by Rob Van der Wijngaart, September 2006.
 Made array dimensioning dynamic, October 2007
 Allowed arbitrary block size, November 2007
 Removed reverse-engineered MKL source code option, November 2007
 Changed from row- to column-major storage order, November 2007
 Stored blocks of B in transpose form, November 2007

 ***********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <omp.h>
#include <time.h>

#ifndef MAXTHREADS
#define MAX_THREADS 1024
#else
#define MAX_THREADS MAXTHREADS
#endif

#define DEFAULTBLOCK 32

#define AA_arr(i,j) AA[(i)+(block+BOFFSET)*(j)]
#define BB_arr(i,j) BB[(i)+(block+BOFFSET)*(j)]
#define CC_arr(i,j) CC[(i)+(block+BOFFSET)*(j)]
#define  A_arr(i,j)  A[(i)+(order)*(j)]
#define  B_arr(i,j)  B[(i)+(order)*(j)]
#define  C_arr(i,j)  C[(i)+(order)*(j)]

#define forder (1.0*order)

#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif
#ifndef MAX
#define MAX(x,y) ((x)>(y)?(x):(y))
#endif
#ifndef ABS
#define ABS(a) ((a) >= 0 ? (a) : -(a))
#endif

/* This function is separate from prk_malloc() because
 * we need it when calling prk_shmem_align(..)           */
static inline int prk_get_alignment(void) {
	/* a := alignment */
# ifdef PRK_ALIGNMENT
	int a = PRK_ALIGNMENT;
# else
	char* temp = getenv("PRK_ALIGNMENT");
	int a = (temp != NULL) ? atoi(temp) : 64;
	if (a < 8)
		a = 8;
	assert((a & (~a + 1)) == a);
#endif
	return a;
}

/* There are a variety of reasons why this function is not declared by stdlib.h. */
#if defined(__UPC__)
int posix_memalign(void **memptr, size_t alignment, size_t size);
#endif

static inline void* prk_malloc(size_t bytes) {
#ifndef PRK_USE_MALLOC
	int alignment = prk_get_alignment();
#endif

	/* Berkeley UPC throws warnings related to this function for no obvious reason... */
#if !defined(__UPC__) && defined(__INTEL_COMPILER) && !defined(PRK_USE_POSIX_MEMALIGN)
	return (void*)_mm_malloc(bytes,alignment);
#elif defined(PRK_HAS_C11)
	/* From ISO C11:
	 *
	 * "The aligned_alloc function allocates space for an object
	 *  whose alignment is specified by alignment, whose size is
	 *  specified by size, and whose value is indeterminate.
	 *  The value of alignment shall be a valid alignment supported
	 *  by the implementation and the value of size shall be an
	 *  integral multiple of alignment."
	 *
	 *  Thus, if we do not round up the bytes to be a multiple
	 *  of the alignment, we violate ISO C.
	 */
	size_t padded = bytes;
	size_t excess = bytes % alignment;
	if (excess>0) padded += (alignment - excess);
	return aligned_alloc(alignment,padded);
#elif defined(PRK_USE_MALLOC)
#warning PRK_USE_MALLOC prevents the use of alignmed memory.
	return prk_malloc(bytes);
#else /* if defined(PRK_USE_POSIX_MEMALIGN) */
	void * ptr = NULL;
	int ret;
	ret = posix_memalign(&ptr, alignment, bytes);
	if (ret)
		ptr = NULL;
	return ptr;
#endif
}

void dgemm(double *A, double *B, double *C, long order, int block) {

	int i, ii, j, jj, k, kk, ig, jg, kg;

#pragma omp parallel private (i,j,k,ii,jj,kk,ig,jg,kg)
	{
		double *AA = NULL, *BB = NULL, *CC = NULL;

		if (block > 0) {
			/* matrix blocks for local temporary copies*/
			AA = (double *) prk_malloc(
					block * (block + BOFFSET) * 3 * sizeof(double));
			if (!AA) {
				printf(
						"Could not allocate space for matrix tiles on thread %d\n",
						omp_get_thread_num());
				exit(1);
			}
			BB = AA + block * (block + BOFFSET);
			CC = BB + block * (block + BOFFSET);
		}

		if (block > 0) {

#pragma omp for
			for (jj = 0; jj < order; jj += block) {
				for (kk = 0; kk < order; kk += block) {

					for (jg = jj, j = 0; jg < MIN(jj + block, order); j++, jg++)
						for (kg = kk, k = 0; kg < MIN(kk + block, order);
								k++, kg++)
							BB_arr(j,k)= B_arr(kg,jg);

							for(ii = 0; ii < order; ii+=block) {

								for (kg=kk,k=0; kg<MIN(kk+block,order); k++,kg++)
								for (ig=ii,i=0; ig<MIN(ii+block,order); i++,ig++)
								AA_arr(i,k) = A_arr(ig,kg);

								for (jg=jj,j=0; jg<MIN(jj+block,order); j++,jg++)
								for (ig=ii,i=0; ig<MIN(ii+block,order); i++,ig++)
								CC_arr(i,j) = 0.0;

								for (kg=kk,k=0; kg<MIN(kk+block,order); k++,kg++)
								for (jg=jj,j=0; jg<MIN(jj+block,order); j++,jg++)
								for (ig=ii,i=0; ig<MIN(ii+block,order); i++,ig++)
								CC_arr(i,j) += AA_arr(i,k)*BB_arr(j,k);

								for (jg=jj,j=0; jg<MIN(jj+block,order); j++,jg++)
								for (ig=ii,i=0; ig<MIN(ii+block,order); i++,ig++)
								C_arr(ig,jg) += CC_arr(i,j);

							}
						}
					}
				}
				else {
#pragma omp for
					for (jg=0; jg<order; jg++)
					for (kg=0; kg<order; kg++)
					for (ig=0; ig<order; ig++)
					C_arr(ig,jg) += A_arr(ig,kg)*B_arr(kg,jg);
				}

			}

		}

void read_input(double *A, double *B, char * fileA, char * fileB,
		long int order) {
	FILE *file, *file2;
	int i, j;

	if ((file = fopen(fileA, "rb")) == 0) {
		printf("The inputA file was not opened\n");
		exit(1);
	}
	if ((file2 = fopen(fileB, "rb")) == 0) {
		printf("The inputB file was not opened\n");
		exit(1);
	}

	for (j = 0; j < order; j++)
		for (i = 0; i < order; i++) {
			fread(&A[(i) + (order) * (j)], 1, sizeof(double), file);
			fread(&B[(i) + (order) * (j)], 1, sizeof(double), file2);
		}
	fclose(file);
	fclose(file2);
}

int main(int argc, char **argv) {

	int i, j;
	int nthread_input; /* thread parameters                              */
	static
	double *A, *B, *C; /* input (A,B) and output (C) matrices            */
	long order; /* number of rows and columns of matrices         */
	int block; /* tile size of matrices                          */
	char *inputA, *inputB, *fileGold;

	printf("OpenMP Dense matrix-matrix multiplication\n");

	if (argc < 6) {
		printf(
				"Usage: %s <# threads> <matrix order> <tile size> <matrix A> <matrix B> <gold output (optional)>\n",
				*argv);
		exit(1);
	}

	/* Take number of threads to request from command line                          */
	nthread_input = atoi(*++argv);

	if ((nthread_input < 1) || (nthread_input > MAX_THREADS)) {
		printf("ERROR: Invalid number of threads: %d\n", nthread_input);
		exit(1);
	}

	omp_set_num_threads(nthread_input);

	order = atol(*++argv);
	if (order < 0) {
		order = -order;
	}
	if (order < 1) {
		printf("ERROR: Matrix order must be positive: %ld\n", order);
		exit(1);
	}
	A = (double *) prk_malloc(order * order * sizeof(double));
	B = (double *) prk_malloc(order * order * sizeof(double));
	C = (double *) prk_malloc(order * order * sizeof(double));
	if (!A || !B || !C) {
		printf("ERROR: Could not allocate space for global matrices\n");
		exit(1);
	}

	block = atoi(*++argv);
	if (block < 1) {
		printf("ERROR: block size must be positive: %d\n", block);
		exit(1);
	}
	inputA = *++argv;
	inputB = *++argv;

	read_input(A, B, inputA, inputB, order);

	printf("Matrix order          = %ld\n", order);
	printf("Number of threads     = %d\n", nthread_input);
	if (block > 0)
		printf("Blocking factor       = %d\n", block);
	else
		printf("No blocking\n");
	printf("Block offset          = %d\n", BOFFSET);

	dgemm(A, B, C, order, block);

	FILE *file;
	char *output_gold;
	char out_tmp[150];
	if (argc == 7){
		output_gold = *++argv;
	}else{
		output_gold = out_tmp;
		snprintf(output_gold, 150, "gold_%ld_m-order_%d_ths_%d_blocks", order,
				nthread_input, block);
	}

	if ((file = fopen(output_gold, "wb")) == 0)
		printf("The GOLD file was not opened\n");
	int zero_sum = 0;
	for (j = 0; j < order; j++)
		for (i = 0; i < order; i++) {
			if (C_arr(i,j)< 0.000001 && C_arr(i,j) > -0.000001) zero_sum++;
			fwrite(&C_arr(i,j), 1, sizeof(double), file);
		}
	fclose(file);
	printf("Number of zeros in the output: %d\n", zero_sum);

	exit(0);

}
