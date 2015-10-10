/*
 * =====================================================================================
 *
 *       Filename:  lud.cu
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include <cuda.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
// helper functions
#include "helper_string.h"
#include "helper_cuda.h"

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 16
#endif

typedef struct parameters {
	int matrix_size;
	double *host_m, *host_gold, *host_out, *dev_m;
	char *inputName;
	char *goldName;
	int generate;
	int debug;
	int reps;
	int verbose;
} parameters_t;

void print_usage(const char *exec_name)
{
    printf("Usage: %s -matrix_size=N <-reps=N> <-generate> <-generategold> <-input=inputpath> <-gold=goldpath> (optional)\n", exec_name);
    printf("  matrix_size: the size of a NxN matrix. It must be greater than 0.\n");
}

void hostSetup(parameters_t *params)
{
	FILE *finput, *fgold;
	int size = params->matrix_size * params->matrix_size * sizeof(double);

	params->host_m = (double*)malloc(size);
	params->host_out = (double*)malloc(size);
	params->host_gold = (double*)malloc(size);	

	if (params->generate == 1)
	{ // Generate input and write to file
		for (int i=0; i<params->matrix_size; i++)
		    params->host_m[i] = (double) rand() / 32768.0;

		if (!(finput = fopen(params->inputName, "wb")))
		{
		  printf("Error: Could not open input file in wb mode. %s\n", params->inputName);
		  exit(EXIT_FAILURE);
		}
		else
		{
		  fwrite(params->host_m, size, 1, finput);
		  fclose(finput);
		}
	}
	else
	{ // Read input from file
		if (!(finput = fopen(params->inputName, "rb")))
		{
		  printf("Error: Could not open input file in rb mode. %s\n", params->inputName);
		  exit(EXIT_FAILURE);
		}
		else
		{
		  fread(params->host_m, size, 1, finput);
		  fclose(finput);
		}
	}

	if (params->debug)
	{
///////////////////// DEBUG
////////////////////// FAULT INJECTION
		params->host_m[11] = 6.0;
		printf("====> Fault injection: host_m[11]=6.0 set.\n");
	}

    if (params->generate == 0)
    { // No generate, read gold from file.
      if (!(fgold = fopen(params->goldName, "rb")))
      {
        printf("Error: Could not open gold file in rb mode. %s\n", params->goldName);
        exit(EXIT_FAILURE);
      }
      else
      {
        fread(params->host_gold, size, 1, fgold);
        fclose(fgold);
      }
    }
}

double mysecond()
{
   struct timeval tp;
   struct timezone tzp;
   int i = gettimeofday(&tp,&tzp);
   return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

extern void
lud_cuda(double *d_m, int matrix_size);

void getParams(parameters_t *params, int argc, char *argv[])
{
    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "h"))
    {
        print_usage(argv[0]);
        exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "matrix_size"))
    {
        params->matrix_size = getCmdLineArgumentInt(argc, (const char **)argv, "matrix_size");

        if (params->matrix_size <= 0)
        {
            printf("Invalid matrix size given on the command-line: %d\n", params->matrix_size);
            exit(EXIT_FAILURE);
        }
    }
	else
	{
		params->matrix_size = 1024;
	}

    if (checkCmdLineFlag(argc, (const char **)argv, "reps"))
    {
        params->reps = getCmdLineArgumentInt(argc, (const char **)argv, "reps");

        if (params->reps <= 0)
        {
            printf("Invalid reps given on the command-line: %d\n", params->reps);
            exit(EXIT_FAILURE);
        }
    }
	else
	{
		params->reps = 1000000;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "debug"))
    {
        params->debug = 1;
    }
	else
	{
		params->debug = 0;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "verbose"))
    {
        params->verbose = 1;
    }
	else
	{
		params->verbose = 0;
	}

    if (checkCmdLineFlag(argc, (const char **)argv, "generate"))
    { // 0 = no / 1 = input+gold / 2 = gold
        params->generate = 1;
    }
	else if (checkCmdLineFlag(argc, (const char **)argv, "generategold"))
    {
        params->generate = 2;
    }
	else
	{
		params->generate = 0;
	}

    if (checkCmdLineFlag(argc, (const char **)argv, "gold"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "gold", &(params->goldName));
    }
    else
    {
        params->goldName = new char[100];
        snprintf(params->goldName, 100, "lud_gold_%i", (signed int)(params->matrix_size));
        printf("Using default gold path: %s\n", params->goldName);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "input", &(params->inputName));
    }
    else
    {
        params->inputName = new char[100];
        snprintf(params->inputName, 100, "lud_input_%i", (signed int)(params->matrix_size));
        printf("Using default input path: %s\n", params->inputName);
    }
}

void devSetup(parameters_t *params)
{
	int size = params->matrix_size * params->matrix_size * sizeof(double);

	cudaMalloc((void**)&(params->dev_m), size);

	cudaMemcpy(params->dev_m, params->host_m, size, cudaMemcpyHostToDevice);
}

void devRelease(parameters_t *params)
{
	cudaFree(params->dev_m);
}

void hostRelease(parameters_t *params)
{
	free(params->host_m);
	free(params->host_gold);
}

int validateOutput(parameters_t *params)
{
	int size = params->matrix_size * params->matrix_size * sizeof(double);

	cudaMemcpy(params->host_out, params->dev_m, size, cudaMemcpyDeviceToHost);

	if (params->generate == 1)
	{ // Write gold to file.
		FILE *fgold;

		if (!(fgold = fopen(params->goldName, "wb")))
		{
		  printf("Error: Could not open gold file in wb mode. %s\n", params->goldName);
		  exit(EXIT_FAILURE);
		}
		else
		{
		  fwrite(params->host_out, size, 1, fgold);
		  fclose(fgold);
		}
	}
	else
	{
		int nerrors=0;
		double *ptr = params->host_out;
		double *goldptr = params->host_gold;

		#pragma omp parallel for
		for (int i=0; i<params->matrix_size; i++)
		{
			if (ptr[i]!=goldptr[i])
			{
				printf("CHECK: Error detected: [%d] read: %lf expected: %lf\n", i, ptr[i], goldptr[i]);
				#pragma omp critical
				{
					nerrors++;
				}
				if (nerrors>=20) exit(EXIT_FAILURE);
			}
		}
		if (nerrors)
		{
			printf("errors: %d\n", nerrors);
			return nerrors;
		}
	}
	return 0;
}

int main ( int argc, char *argv[] )
{
	double timer, globaltimer;
	parameters_t params;
	printf("Starting LU Decomposition (NO cdp)\n");

	getParams(&params, argc, argv);

	hostSetup(&params);

	for (int loop = 0; loop < params.reps; loop++)
	{
		globaltimer = mysecond();
		timer = mysecond();
		int errors = 0;

		devSetup(&params);
		if (params.verbose) printf("Iteration[%d]: Setup time: %.3fs\n", loop, mysecond() - timer);

		timer = mysecond();
		lud_cuda(params.dev_m, params.matrix_size);
		if (params.verbose) printf("Iteration[%d]: Device time: %.3fs\n", loop, mysecond() - timer);

		timer = mysecond();
		errors = validateOutput(&params);
		if (params.verbose) printf("Iteration[%d]: Validation time %.3fs\n", loop, mysecond() - timer);

		if (errors)
		{
			hostRelease(&params);
			hostSetup(&params);
			printf("Iteration: %d\n------------------\n", loop);
		}
		else
		{
			printf(".");
		}
		fflush(stdout);

		devRelease(&params);
		if (params.verbose) printf("Iteration[%d]: Global time %.3fs\n", loop, mysecond() - globaltimer);
		if (params.verbose) printf("------------------\n");
	}

	hostRelease(&params);

	return EXIT_SUCCESS;
}				/* ----------  end of function main  ---------- */
