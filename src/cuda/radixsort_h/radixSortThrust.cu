/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>
#include <omp.h>

#include <helper_cuda.h>
#include "helper_sorts.h"

#include <algorithm>
#include <time.h>
#include <limits.h>

#include <sys/time.h>

#define INPUTSIZE 134217728
#define RETRY_COUNT 3


extern "C" uint sortVerify(uint *d_DstKey, uint *d_DstVal, uint *d_SrcVal, int size);

int generate;

#ifdef LOGS
	#include "log_helper.h"
#endif

typedef struct parameters_s {
	int numElements;
	int numIterations;
	int verbose;
	int debug;
	int generate;
	int fault_injection;
	char *goldName, *inputName;
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
	printf("Syntax: %s -size=N [-generate] [-float] [-verbose] [-debug] [-inputkey=<path>] [-inputval=<path>] [-gold=<path>] [-iterations=N] [-keysonly] \n", argv[0]);
	exit(EXIT_FAILURE);
}

void getParams(int argc, char *argv[], parameters_t *params)
{
	params->numIterations = 100000000;
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
		params->numElements = getCmdLineArgumentInt(argc, (const char **)argv, "size");
		if (params->numElements > INPUTSIZE) {
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
		params->numIterations = 1;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "iterations")) {
		params->numIterations = getCmdLineArgumentInt(argc, (const char **)argv, "iterations");
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "debug")) {
		params->fault_injection = 1;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "gold")) {
		getCmdLineArgumentString(argc, (const char **)argv, "gold", &(params->goldName));
	} else {
		params->goldName = new char[100];
		snprintf(params->goldName, 100, "radixsort_Gold%s%i", "KeysVals", (signed int)params->numElements);
		printf("Using default gold filename: %s\n", params->goldName);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "input")) {
		getCmdLineArgumentString(argc, (const char **)argv, "input", &(params->inputName));
	} else {
		params->inputName = new char[100];
		snprintf(params->inputName, 100, "radixsort_Input%i", (signed int)INPUTSIZE);
		printf("Using default keys input filename: %s\n", params->inputName);
	}
}

void readData(parameters_t *params, uint *h_keys, uint *h_keysGold, uint *h_values, uint *h_valuesGold, int onlygold = 0)
{
    FILE *finput, *fgold;
    if (!onlygold)
    {
    if (finput = fopen(params->inputName, "rb")) {
		fread(h_keys, params->numElements * sizeof(uint), 1, finput);
	} else if (params->generate) {
        uint *new_keys = (uint *)malloc(sizeof(uint) * INPUTSIZE);
        // Fill up with some random data
        thrust::default_random_engine rng(clock());

        thrust::uniform_int_distribution<unsigned int> u(0, UINT_MAX);

        for (int i = 0; i < (int)INPUTSIZE; i++)
            new_keys[i] = u(rng);

		if (finput = fopen(params->inputName, "wb"))  {
			fwrite(new_keys, INPUTSIZE * sizeof(uint), 1, finput);
		} else {
			printf("Could not write key input to file, proceeding anyway...\n");
		}
		memcpy(h_keys, new_keys, params->numElements * sizeof(uint));
		free(new_keys);
	} else {
		fatal("Could not open key input. Use -generate");
	}
	fclose(finput);

	register uint *ptr = h_values;
	register uint i;
	#pragma omp parallel for
	for (i = 0; i<params->numElements; i++)
		ptr[i] = i;

	if (params->fault_injection) {
		h_keys[5] = rand() % UINT_MAX;
		printf(">>>>> Will inject an error: h_keys[5]=%d\n", h_keys[5]);
		h_values[12] = rand() % params->numElements;
		printf(">>>>> Will inject an error: h_values[12]=%d\n", h_values[12]);
	}
}
    if (!(params->generate)) {
        if (fgold = fopen(params->goldName, "rb")) {
            fread(h_keysGold, params->numElements * sizeof(uint), 1, fgold);
			fread(h_valuesGold, params->numElements * sizeof(uint), 1, fgold);
            fclose(fgold);
        } else {
            fatal("Could not open gold file. Use -generate");
        }
    }
}

void writeOutput(parameters_t *params, uint *h_keys, uint *h_values)
{
	FILE *fgold;
	if (fgold = fopen(params->goldName, "wb")) {
		fwrite(h_keys, params->numElements * sizeof(uint), 1, fgold);
		fwrite(h_values, params->numElements * sizeof(uint), 1, fgold);
		fclose(fgold);
	} else {
		printf("Error: could not open gold file in wb mode.\n");
	}
}

int checkKeys(parameters_t *params, uint *h_keys, uint *h_keysOut)
{ // Magicas que a semana anterior ao teste proporcionam
    register unsigned char *srcHist;
    register unsigned char *resHist;

	uint numValues = UINT_MAX;

    int flag = 1;
	int errors = 0;

	register uint index, range;
	long unsigned int control;
	range = ((numValues*sizeof(unsigned char) > (size_t)2*1024*1024*1024) ? (size_t)2*1024*1024*1024 : numValues); // Avoid more than 4GB of RAM alloc

	srcHist = (unsigned char *)malloc(range * sizeof(unsigned char));
	resHist = (unsigned char *)malloc(range * sizeof(unsigned char));

	if (!srcHist || !resHist) fatal("Could not alloc src or res histograms");

    for (index = 0, control = 0; control < numValues; index += range, control += range)
	{
		printf("index = %u range = %u alloc=%.2fMB\n", index, range, 2 * (double)range * sizeof(unsigned char) / 1000000);


        //Build histograms for keys arrays
        memset(srcHist, 0, range * sizeof(unsigned char));
        memset(resHist, 0, range * sizeof(unsigned char));

		register uint indexPLUSrange = index + range;
		register uint i;
		register uint *resKey = h_keysOut;
		register uint *srcKey = h_keys;
		#pragma omp parallel for
		for (i = 0; i < params->numElements; i++)
        {
			if ((srcKey[i] >= index) && (srcKey[i] < indexPLUSrange))
            {
				#pragma omp atomic
                srcHist[srcKey[i]-index]++;
            }
			if ((resKey[i] >= index) && (resKey[i] < indexPLUSrange))
            {
				#pragma omp atomic
                resHist[resKey[i]-index]++;
            }
        }

		#pragma omp parallel for
		for (i = 0; i < range; i++)
            if (srcHist[i] != resHist[i])
			#pragma omp critical
            {
				char error_detail[150];
                snprintf(error_detail, 150, "The histogram from element %d differs. srcHist=%d dstHist=%d\n", i+index, srcHist[i], resHist[i]);
                #ifdef LOGS
                    if (!(params->generate)) log_error_detail(error_detail);
                #endif
                printf("ERROR : %s\n", error_detail);
				errors++;
                flag = 0;
            }

	}
	free(resHist);
	free(srcHist);

	//Finally check the ordering
	register uint i;
	register uint *resKeys = h_keysOut;
	#pragma omp parallel for
	for (i = 0; i < params->numElements - 1; i++)
		if (resKeys[i] > resKeys[i + 1])
		#pragma omp critical
		{
			char error_detail[150];
			snprintf(error_detail, 150, "Elements not ordered. index=%d %d>%d", i, resKeys[i], resKeys[i + 1]);
			#ifdef LOGS
				if (!(params->generate)) log_error_detail(error_detail);
			#endif
			printf("ERROR: %s\n", error_detail);
			errors++;
			flag = 0;
		}

    if (flag) printf("OK\n");
    if (!flag) printf("Errors found.\n");

	return errors;
}

int checkVals(parameters_t *params, uint *h_keys, uint *h_keysOut, uint *h_valuesOut)
{
    int correctFlag = 1, stableFlag = 1;
	int errors = 0;

    printf("...inspecting keys and values array: "); fflush(stdout);

	register uint *resKey = h_keysOut;
	register uint *srcKey = h_keys;
	register uint *resVal = h_valuesOut;
	register uint j;
	#pragma omp parallel for
    for (j = 0; j < params->numElements; j++)
    {
        if (resKey[j] != srcKey[resVal[j]])
		#pragma omp critical
		{
			char error_detail[150];
			snprintf(error_detail, 150, "The link between Val and Key arrays in incorrect. index=%d wrong_key=%d val=%d correct_key_pointed_by_val=%d", j, resKey[j], resVal[j], srcKey[resVal[j]]);
			#ifdef LOGS
				if (!(params->generate)) log_error_detail(error_detail);
			#endif
			printf("ERROR: %s\n", error_detail);
			errors++;
            correctFlag = 0;
		}

        if ((j < params->numElements - 1) && (resKey[j] == resKey[j + 1]) && (resVal[j] > resVal[j + 1]))
		#pragma omp critical
		{
			char error_detail[150];
			snprintf(error_detail, 150, "Unstability detected at index=%d key=%d val[i]=%d val[i+1]=%d", j, resKey[j], resVal[j], resVal[j + 1]);
			#ifdef LOGS
				if (!(params->generate)) log_error_detail(error_detail);
			#endif
			printf("ERROR: %s\n", error_detail);
			errors++;
            correctFlag = 0;
		}
    }

    printf(correctFlag ? "OK\n" : "***corrupted!!!***\n");
    printf(stableFlag ? "...stability property: stable!\n" : "...stability property: NOT stable\n");

	return errors;
}

int compareGoldOutput(parameters_t *params, uint *h_keysOut, uint *h_valuesOut, uint *h_keysGold, uint *h_valuesGold)
{
	//return (memcmp(params->h_GoldKey, params->h_DstKey, params->numElements * sizeof(uint)) || memcmp(params->h_GoldVal, params->h_DstVal, params->numElements * sizeof(uint)));
	int flag = 0;
	register uint *h_kOut = h_keysOut;
	register uint *h_kGold = h_keysGold;
	register uint *h_vOut = h_valuesOut;
	register uint *h_vGold = h_valuesGold;
	#pragma omp parallel num_threads(2) shared(flag)
	{
		if (omp_get_thread_num() == 0) { // Thread 0
			register unsigned int i;
			#pragma omp parallel for
			for (i=0; i<params->numElements; i++)
			{
				if (h_kOut[i] != h_kGold[i]) flag=1;
			}
		} else { // Thread 1
			register unsigned int i;
			#pragma omp parallel for
			for (i=0; i<params->numElements; i++)
			{
				if (h_vOut[i] != h_vGold[i]) flag=1;
			}
		}
	}
	return flag;
}

void testSort(parameters_t *params)
{
    int keybits = 32;

	double itertimestamp, kernel_time, timestamp;
	int retries = 0, errNum=0;

    if (params->verbose)
        printf("\nSorting %d %d-bit int keys and values\n\n", params->numElements, keybits);

    int deviceID = -1;

    if (cudaSuccess == cudaGetDevice(&deviceID))
    {
        cudaDeviceProp devprop;
        cudaGetDeviceProperties(&devprop, deviceID);
        unsigned int totalMem = 4 * params->numElements * sizeof(uint);

        if (devprop.totalGlobalMem < totalMem)
        {
            printf("Error: insufficient amount of device memory to sort %d elements.\n", params->numElements);
            printf("%u bytes needed, %u bytes available\n", (int) totalMem, (int) devprop.totalGlobalMem);
            exit(EXIT_SUCCESS);
        }
    }

    uint *h_keys = (uint*) malloc(sizeof(uint) * params->numElements);
    uint *h_keysOut = (uint*) malloc(sizeof(uint) * params->numElements);
    uint *h_keysGold = (uint*) malloc(sizeof(uint) * params->numElements);
    uint *h_values, *h_valuesOut, *h_valuesGold;

	h_values = (uint*) malloc(sizeof(uint) * params->numElements);
	h_valuesOut = (uint*) malloc(sizeof(uint) * params->numElements);
	h_valuesGold = (uint*) malloc(sizeof(uint) * params->numElements);
	
	if (params->verbose) printf("Preparing setup data..."); fflush(stdout);
	timestamp = mysecond();

    readData(params, h_keys, h_keysGold, h_values, h_valuesGold);

	if (params->verbose) printf("Done in %.4fs\n", mysecond() - timestamp);

    // Copy data onto the GPU
    uint *d_keys, *d_values, *d_input_keys, *d_input_values;
	checkCudaErrors( cudaMalloc((void**)&d_keys, sizeof(uint) * params->numElements) );
	checkCudaErrors( cudaMalloc((void**)&d_input_keys, sizeof(uint) * params->numElements) );
	checkCudaErrors( cudaMalloc((void**)&d_values, sizeof(uint) * params->numElements) );
	checkCudaErrors( cudaMalloc((void**)&d_input_values, sizeof(uint) * params->numElements) );


	checkCudaErrors( cudaMemcpy(d_input_keys, h_keys, sizeof(uint) * params->numElements, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_input_values, h_values, sizeof(uint) * params->numElements, cudaMemcpyHostToDevice) );

    // run multiple iterations to compute an average sort time

    for (unsigned int loop1 = 0; loop1 < params->numIterations; loop1++)
    {
		itertimestamp = mysecond();
        if (params->verbose) printf("================== [Iteration #%i began]\n", loop1);
        
		retries = 0;
		do 
		{

			// reset data before sort
			/*checkCudaErrors( cudaMemcpy(d_keys, h_keys, sizeof(uint) * params->numElements, cudaMemcpyHostToDevice) );
			checkCudaErrors( cudaMemcpy(d_values, h_values, sizeof(uint) * params->numElements, cudaMemcpyHostToDevice) );
			*/
			// reset data before sort
			checkCudaErrors( cudaMemcpy(d_keys, d_input_keys, sizeof(uint) * params->numElements, cudaMemcpyDeviceToDevice) );
			checkCudaErrors( cudaMemcpy(d_values, d_input_values, sizeof(uint) * params->numElements, cudaMemcpyDeviceToDevice) );

			timestamp = mysecond();

			///////////////// Kernel
			checkCudaErrors(cudaDeviceSynchronize());
			#ifdef LOGS
				if (!(params->generate)) start_iteration();
			#endif

			thrust::sort_by_key(thrust::device_ptr<uint> (d_keys), thrust::device_ptr<uint> (d_keys+params->numElements), thrust::device_ptr<uint> (d_values));
			checkCudaErrors(cudaDeviceSynchronize());
			
			kernel_time = mysecond() - timestamp;
			
			errNum = sortVerify(d_keys, d_values, d_input_keys, params->numElements);
			checkCudaErrors(cudaDeviceSynchronize());
			
			#ifdef LOGS
				if (!(params->generate)) end_iteration();
			#endif
			/////////////////////////

			if (errNum) printf("GPU Verify Found ERROR! numErr=%d\n", errNum);
			if (params->verbose) printf("GPU Kernel time: %.4fs\n", kernel_time);
			if (params->verbose) printf("GPU Verify Kernel time: %.4fs\n", mysecond() - timestamp);
			retries++;
		} while ((retries<RETRY_COUNT) && (errNum != 0));


	    // Get results back to host for correctness checking
		checkCudaErrors( cudaMemcpy(h_keysOut, d_keys, sizeof(uint) * params->numElements, cudaMemcpyDeviceToHost) );
		checkCudaErrors( cudaMemcpy(h_valuesOut, d_values, sizeof(uint) * params->numElements, cudaMemcpyDeviceToHost) );

		timestamp = mysecond();
		int errors = 0;
		if (params->generate) {
			printf("Validating output...\n");

			errors += checkKeys(params, h_keys, h_keysOut);
			errors += checkVals(params, h_keys, h_keysOut, h_valuesOut);

			if (errors)
				printf("Errors ocurred when validating gold, this is bad. I will save it to file anyway.\n");

			writeOutput(params, h_keysOut, h_valuesOut);
		} else {
			if (compareGoldOutput(params, h_keysOut, h_valuesOut, h_keysGold, h_valuesGold)) {
				free(h_keysGold);
				free(h_valuesGold);

				printf("Warning! Gold file mismatch detected, proceeding to error analysis...\n");

				errors += checkKeys(params, h_keys, h_keysOut);
				errors += checkVals(params, h_keys, h_keysOut, h_valuesOut);

				h_keysGold = (uint*) malloc(sizeof(uint) * params->numElements);
				h_valuesGold = (uint*) malloc(sizeof(uint) * params->numElements);

			} else {
				errors = 0;
			}
			#ifdef LOGS
				if (!(params->generate)) log_error_count(errors);
			#endif
		}
        if (params->verbose) printf("Host gold check time: %.4fs\n", mysecond() - timestamp);

		// Display the time between event recordings
        if (params->verbose) printf("Perf: %.3fM elems/sec\n", 1.0e-6f * params->numElements / kernel_time);
        if (params->verbose) {
            printf("Iteration %d ended. Elapsed time: %.4fs\n", loop1, mysecond()-itertimestamp);
        } else {
            printf(".");
        }
        fflush(stdout);
    }

	return;
}

int main(int argc, char **argv)
{
	parameters_t *params = (parameters_t *) malloc(sizeof(parameters_t));
	getParams(argc, argv, params);
    // Start logs
    printf("%s Starting...\n\n", argv[0]);

    findCudaDevice(argc, (const char **)argv);

	#ifdef LOGS
		char test_info[90];
		snprintf(test_info, 90, "size:%d keysOnly:0", params->numElements);
		if (!params->generate) start_log_file("cudaRadixSort", test_info);
	#endif

    testSort(params);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    checkCudaErrors(cudaDeviceReset());
}
