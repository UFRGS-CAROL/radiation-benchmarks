#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "helper_string.h"
#include "helper_cuda.h"
#include "mergeSort_common.h"
#include <sys/time.h>

#define INPUTSIZE 134217728

int generate;
#ifdef LOGS
	#include "log_helper.h"
#endif


typedef struct parameters_s {
	int size;
	int iterations;
	int verbose;
	int debug;
	int generate;
	int fault_injection;
	char *goldName, *valName, *keyName;
    uint *h_SrcKey, *h_SrcVal, *h_GoldVal, *h_GoldKey, *h_DstKey, *h_DstVal;
    uint *d_SrcKey, *d_SrcVal, *d_BufKey, *d_BufVal, *d_DstKey, *d_DstVal;
} parameters_t;

double mysecond()
{
	struct timeval tp;
	struct timezone tzp;
	int i = gettimeofday(&tp,&tzp);
	return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void fatal(const char *str)
{
	printf("FATAL: %s\n", str);
	#ifdef LOGS
		if (generate) end_log_file();
	#endif
	exit(EXIT_FAILURE);
}

static void usage(int argc, char *argv[])
{
	printf("Syntax: %s -size=N [-generate] [-verbose] [-debug] [-inputkey=<path>] [-inputval=<path>] [-gold=<path>] [-iterations=N]\n", argv[0]);
	exit(EXIT_FAILURE);
}

void getParams(int argc, char *argv[], parameters_t *params)
{
	params->size = 10000;
	params->iterations = 100000000;
	params->verbose = 0;
	params->generate = 0;
	params->fault_injection = 0;
	generate = 0;

	if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
		checkCmdLineFlag(argc, (const char **)argv, "h"))
	{
		usage(argc, argv);
		exit(EXIT_WAIVED);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "size")) {
		params->size = getCmdLineArgumentInt(argc, (const char **)argv, "size");
		if (params->size > INPUTSIZE) {
			fatal("Maximum size reached, please increase the input size on the code source and recompile.");
		}
	} else {
		printf("Missing -size parameter.\n");
		usage(argc, argv);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "verbose")) {
		params->verbose = 1;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "generate")) {
		params->generate = 1;
		generate = 1;
		params->iterations = 1;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "iterations")) {
		params->iterations = getCmdLineArgumentInt(argc, (const char **)argv, "iterations");
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "debug")) {
		params->fault_injection = 1;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "gold")) {
		getCmdLineArgumentString(argc, (const char **)argv, "gold", &(params->goldName));
	} else {
		params->goldName = new char[100];
		snprintf(params->goldName, 100, "mergesortGold%i", (signed int)params->size);
		printf("Using default gold filename: %s\n", params->goldName);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "inputval")) {
		getCmdLineArgumentString(argc, (const char **)argv, "inputval", &(params->valName));
	} else {
		params->valName = new char[100];
		snprintf(params->valName, 100, "mergesortValInput%i", (signed int)INPUTSIZE);
		printf("Using default vals input filename: %s\n", params->valName);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "inputkey")) {
		getCmdLineArgumentString(argc, (const char **)argv, "inputkey", &(params->keyName));
	} else {
		params->keyName = new char[100];
		snprintf(params->keyName, 100, "mergesortKeyInput%i", (signed int)INPUTSIZE);
		printf("Using default keys input filename: %s\n", params->keyName);
	}
}

void readData(parameters_t *params, const uint numValues) 
{
	FILE *fgold, *finput;
	if (finput = fopen(params->valName, "rb")) {
		fread(params->h_SrcVal, params->size * sizeof(uint), 1, finput);
	} else if (params->generate) {		
		uint *newVals = (uint*) malloc(INPUTSIZE*sizeof(uint));
		fillValues(newVals, INPUTSIZE);

		if (finput = fopen(params->valName, "wb"))  {
			fwrite(newVals, INPUTSIZE * sizeof(uint), 1, finput);
		} else {
			printf("Could not write val input to file, proceeding anyway...\n");
		}

		memcpy(params->h_SrcVal, newVals, params->size * sizeof(uint));
		free(newVals);
	} else {
		fatal("Could not open val input. Use -generate");
	}
	fclose(finput);

	if (finput = fopen(params->keyName, "rb")) {
		fread(params->h_SrcKey, params->size * sizeof(uint), 1, finput);
	} else if (params->generate) {
		uint *newKeys = (uint*) malloc(INPUTSIZE*sizeof(uint));

		for (uint i = 0; i < INPUTSIZE; i++)
		{
		    newKeys[i] = rand() % numValues;
		}

		if (finput = fopen(params->keyName, "wb")) {
			fwrite(newKeys, INPUTSIZE * sizeof(uint), 1, finput);
		} else {
			printf("Could not write key input to file, proceeding anyway...\n");
		}

		memcpy(params->h_SrcKey, newKeys, params->size * sizeof(uint));
		free(newKeys);
	} else {
		fatal("Could not open key input. Use -generate");
	}
	fclose(finput);

	if (!(params->generate)) {
		if (fgold = fopen(params->goldName, "rb")) {
			fread(params->h_GoldVal, params->size * sizeof(uint), 1, fgold);
			fread(params->h_GoldKey, params->size * sizeof(uint), 1, fgold);
		} else {
			fatal("Could not open gold file. Use -generate");
		}
		fclose(fgold);
	}
}


////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	double timestamp, globaltimestamp;
	uint keysFlag, valuesFlag;

	srand( time(NULL) );

	parameters_t *params;
	params = (parameters_t*) malloc(sizeof(parameters_t));
    
    const uint DIR = 1;
    const uint numValues = 65536;//UINT_MAX-1;

    printf("%s Starting...\n\n", argv[0]);

    int dev = findCudaDevice(argc, (const char **) argv);

    if (dev == -1)
    {
        return EXIT_FAILURE;
    }

	getParams(argc, argv, params);

    printf("Allocating and initializing host arrays...\n\n");
    params->h_SrcKey = (uint *)malloc(params->size * sizeof(uint));
    params->h_SrcVal = (uint *)malloc(params->size * sizeof(uint));
    params->h_GoldKey = (uint *)malloc(params->size * sizeof(uint));
    params->h_GoldVal = (uint *)malloc(params->size * sizeof(uint));
    params->h_DstKey = (uint *)malloc(params->size * sizeof(uint));
    params->h_DstVal = (uint *)malloc(params->size * sizeof(uint));

	readData(params, numValues);

	for (int loop1 = 0; loop1 < params->iterations; loop1++)
	{
		globaltimestamp = mysecond();

		printf("Allocating and initializing CUDA arrays...\n\n");
		checkCudaErrors(cudaMalloc((void **)&(params->d_DstKey), params->size * sizeof(uint)));
		checkCudaErrors(cudaMalloc((void **)&(params->d_DstVal), params->size * sizeof(uint)));
		checkCudaErrors(cudaMalloc((void **)&(params->d_BufKey), params->size * sizeof(uint)));
		checkCudaErrors(cudaMalloc((void **)&(params->d_BufVal), params->size * sizeof(uint)));
		checkCudaErrors(cudaMalloc((void **)&(params->d_SrcKey), params->size * sizeof(uint)));
		checkCudaErrors(cudaMalloc((void **)&(params->d_SrcVal), params->size * sizeof(uint)));
		checkCudaErrors(cudaMemcpy(params->d_SrcKey, params->h_SrcKey, params->size * sizeof(uint), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(params->d_SrcVal, params->h_SrcVal, params->size * sizeof(uint), cudaMemcpyHostToDevice));

		printf("Initializing GPU merge sort...\n");
		initMergeSort();

		printf("Running GPU merge sort...\n");
		checkCudaErrors(cudaDeviceSynchronize());
		timestamp = mysecond();
		mergeSort(
		    params->d_DstKey,
		    params->d_DstVal,
		    params->d_BufKey,
		    params->d_BufVal,
		    params->d_SrcKey,
		    params->d_SrcVal,
		    params->size,
		    DIR
		);
		checkCudaErrors(cudaDeviceSynchronize());
		printf("Time: %.4fs\n", mysecond()-timestamp);

		printf("Reading back GPU merge sort results...\n");
		checkCudaErrors(cudaMemcpy(params->h_DstKey, params->d_DstKey, params->size * sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(params->h_DstVal, params->d_DstVal, params->size * sizeof(uint), cudaMemcpyDeviceToHost));

		printf("Inspecting the results...\n");fflush(stdout);
		for (int j=1; j<params->size; j++)
			if (params->h_DstKey[j-1]>params->h_DstKey[j]) printf("Key[%d]>Key[%d] (%d>%d)\n", j-1, j, params->h_DstKey[j-1], params->h_DstKey[j]);
		for (int j=1; j<params->size; j++)
			if (params->h_DstKey[j-1]>params->h_DstKey[j]) printf("Val[%d]>Val[%d] (%d>%d)\n", j-1, j, params->h_DstKey[j-1], params->h_DstKey[j]);
		keysFlag = validateSortedKeys(
		                    params->h_DstKey,
		                    params->h_SrcKey,
		                    1,
		                    params->size,
		                    numValues,
		                    DIR
		                );
printf("now values...\n"); fflush(stdout);
		valuesFlag = validateSortedValues(
		                      params->h_DstKey,
		                      params->h_DstVal,
		                      params->h_SrcKey,
		                      1,
		                      params->size
		                  );

		printf("Shutting down...\n"); fflush(stdout);
		closeMergeSort();
		checkCudaErrors(cudaFree(params->d_SrcVal));
		checkCudaErrors(cudaFree(params->d_SrcKey));
		checkCudaErrors(cudaFree(params->d_BufVal));
		checkCudaErrors(cudaFree(params->d_BufKey));
		checkCudaErrors(cudaFree(params->d_DstVal));
		checkCudaErrors(cudaFree(params->d_DstKey));
		printf("iteration time=%.4fs\n", mysecond()-globaltimestamp);
	}
    free(params->h_DstVal);
    free(params->h_DstKey);
    free(params->h_SrcVal);
    free(params->h_SrcKey);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    exit((keysFlag && valuesFlag) ? EXIT_SUCCESS : EXIT_FAILURE);
}
