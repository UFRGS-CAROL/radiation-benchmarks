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
	int noinputensurance;
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
	printf("Syntax: %s -size=N [-generate] [-verbose] [-debug] [-inputkey=<path>] [-inputval=<path>] [-gold=<path>] [-iterations=N] [-noinputensurance]\n", argv[0]);
	exit(EXIT_FAILURE);
}

void getParams(int argc, char *argv[], parameters_t *params)
{
	params->size = 10000;
	params->iterations = 100000000;
	params->verbose = 0;
	params->generate = 0;
	params->fault_injection = 0;
	params->noinputensurance = 0;
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

	if (checkCmdLineFlag(argc, (const char **)argv, "noinputensurance")) {
		params->noinputensurance = 1;
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
		snprintf(params->valName, 100, "mergesortInputVal%i", (signed int)INPUTSIZE);
		printf("Using default vals input filename: %s\n", params->valName);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "inputkey")) {
		getCmdLineArgumentString(argc, (const char **)argv, "inputkey", &(params->keyName));
	} else {
		params->keyName = new char[100];
		snprintf(params->keyName, 100, "mergesortInputKey%i", (signed int)INPUTSIZE);
		printf("Using default keys input filename: %s\n", params->keyName);
	}
}

void writeOutput(parameters_t *params)
{
	FILE *fgold;
	if (fgold = fopen(params->goldName, "wb")) {
		fwrite(params->h_DstVal, params->size * sizeof(uint), 1, fgold);
		fwrite(params->h_DstKey, params->size * sizeof(uint), 1, fgold);
		fclose(fgold);
	} else {
		printf("Error: could not open gold file in wb mode.\n");
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

		if (!(params->noinputensurance)) {
		  	printf("Warning: I will alloc %.2fMB of RAM, be ready to crash.", (double)numValues * sizeof(unsigned char) / 1000000);
			printf(" If this hangs, it can not ensure the input does not have more than 256 equal entries, which may");
			printf(" cause really rare issues under radiation tests. Use -noinputensurance in this case.\n");

			unsigned char *srcHist;
			srcHist = (unsigned char *)malloc(numValues * sizeof(unsigned char));
			if (!srcHist)
				fatal("Could not alloc RAM for histogram. Use -noinputensurance.");

			register unsigned char max = 0;
			register uint valor;
			for (uint i = 0; i < INPUTSIZE; i++) {
				do {
					valor = rand() % numValues;
				} while (srcHist[valor]==UCHAR_MAX);
				srcHist[valor]++;
				if (srcHist[valor]>max) max = srcHist[valor];
			    newKeys[i] = valor;
			}
			free(srcHist);
			printf("Maximum repeats of one single key: %d\n", max);
		} else {
			for (uint i = 0; i < INPUTSIZE; i++) {
			    newKeys[i] = rand() % numValues;
			}
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
			fclose(fgold);
		} else {
			fatal("Could not open gold file. Use -generate");
		}
	}

	if (params->fault_injection) {
		params->h_SrcVal[5] = rand() % params->size;
		printf(">>>>> Will inject an error: h_SrcVal[5]=%d\n", params->h_SrcVal[5]);
		params->h_SrcKey[12] = rand() % numValues;
		printf(">>>>> Will inject an error: h_SrcKey[12]=%d\n", params->h_SrcKey[12]);
	}
}

int checkKeys(parameters_t *params, uint numValues)
{ // Magicas que a semana anterior ao teste proporcionam
    unsigned char *srcHist;
    unsigned char *resHist;

    int flag = 1;
	int errors = 0;

	register uint index, range;
	long unsigned int control;
	range = ((2*numValues*sizeof(unsigned char) > 2048000000) ? 1024000000 : numValues); // Avoid more than 2GB of RAM alloc

	srcHist = (unsigned char *)malloc(range * sizeof(unsigned char));
	resHist = (unsigned char *)malloc(range * sizeof(unsigned char));

	if (!srcHist || !resHist) fatal("Could not alloc src or res");

    for (index = 0, control = 0; control < numValues; index += range, control += range)
	{
		printf("index = %u range = %u alloc=%.2fMB\n", index, range, 2 * (double)range * sizeof(unsigned char) / 1000000);


        //Build histograms for keys arrays
        memset(srcHist, 0, range * sizeof(unsigned char));
        memset(resHist, 0, range * sizeof(unsigned char));

		register uint indexPLUSrange = index + range;
		register uint *srcKey = params->h_SrcKey;
		register uint *resKey = params->h_DstKey;
		#pragma omp parallel for
		for (uint i = 0; i < params->size; i++)
        {
			//if (index!=0) printf("srcKey[%d]=%d resKey[%d]=%d index=%d indexPLUSrange=%d\n", i, srcKey[i], i, resKey[i], index, indexPLUSrange); fflush(stdout);
			if ((srcKey[i] >= index) && (srcKey[i] < indexPLUSrange) && (srcKey[i] < numValues))
            {
				#pragma omp atomic
                srcHist[srcKey[i]-index]++;
            }
			if ((resKey[i] >= index) && (resKey[i] < indexPLUSrange) && (resKey[i] < numValues))
            {
				#pragma omp atomic
                resHist[resKey[i]-index]++;
            }
        }
		#pragma omp parallel for
		for (uint i = 0; i < range; i++)
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
	register uint *resKey = params->h_DstKey;
	#pragma omp parallel for
	for (uint i = 0; i < params->size - 1; i++)
		if (resKey[i] > resKey[i + 1])
		#pragma omp critical
		{
			char error_detail[150];
			snprintf(error_detail, 150, "Elements not ordered. index=%d %d>%d", i, resKey[i], resKey[i + 1]);
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

int checkVals(parameters_t *params)
{
    int correctFlag = 1, stableFlag = 1;
	int errors = 0;

    printf("...inspecting keys and values array: "); fflush(stdout);

	register uint *resKey = params->h_DstKey;
	register uint *srcKey = params->h_SrcKey;
	register uint *resVal = params->h_DstVal;
	#pragma omp parallel for
    for (uint j = 0; j < params->size; j++)
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

        if ((j < params->size - 1) && (resKey[j] == resKey[j + 1]) && (resVal[j] > resVal[j + 1]))
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

int compareGoldOutput(parameters_t *params)
{
	//return (memcmp(params->h_GoldKey, params->h_DstKey, params->size * sizeof(uint)) || memcmp(params->h_GoldVal, params->h_DstVal, params->size * sizeof(uint)));
	register unsigned int i;
	register uint *ptr1 = params->h_GoldKey;
	register uint *ptr2 = params->h_DstKey;
	int flag = 0;
	#pragma omp parallel for
	for (i=0; i<params->size; i++)
	{
		if (ptr1[i] != ptr2[i]) flag=1;
	}
	return flag;
}
////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	double timestamp, globaltimestamp, kernel_time;

	srand( time(NULL) );

	parameters_t *params;
	params = (parameters_t*) malloc(sizeof(parameters_t));

    const uint DIR = 1;
    const uint numValues = UINT_MAX; // 65536

    printf("%s Starting...\n\n", argv[0]);

    int dev = findCudaDevice(argc, (const char **) argv);

    if (dev == -1)
    {
        return EXIT_FAILURE;
    }

	getParams(argc, argv, params);

	#ifdef LOGS
		char test_info[90];
		snprintf(test_info, 90, "size:%d", params->size);
		if (!params->generate) start_log_file("cudaMergeSort", test_info);
	#endif

    params->h_SrcKey = (uint *)malloc(params->size * sizeof(uint));
    params->h_SrcVal = (uint *)malloc(params->size * sizeof(uint));
    params->h_GoldKey = (uint *)malloc(params->size * sizeof(uint));
    params->h_GoldVal = (uint *)malloc(params->size * sizeof(uint));
    params->h_DstKey = (uint *)malloc(params->size * sizeof(uint));
    params->h_DstVal = (uint *)malloc(params->size * sizeof(uint));

	readData(params, numValues);

	checkCudaErrors(cudaMalloc((void **)&(params->d_DstKey), params->size * sizeof(uint)));
	checkCudaErrors(cudaMalloc((void **)&(params->d_DstVal), params->size * sizeof(uint)));
	checkCudaErrors(cudaMalloc((void **)&(params->d_BufKey), params->size * sizeof(uint)));
	checkCudaErrors(cudaMalloc((void **)&(params->d_BufVal), params->size * sizeof(uint)));
	checkCudaErrors(cudaMalloc((void **)&(params->d_SrcKey), params->size * sizeof(uint)));
	checkCudaErrors(cudaMalloc((void **)&(params->d_SrcVal), params->size * sizeof(uint)));

	for (int loop1 = 0; loop1 < params->iterations; loop1++)
	{
		globaltimestamp = mysecond();
        if (params->verbose) printf("================== [Iteration #%i began]\n", loop1);

		checkCudaErrors(cudaMemcpy(params->d_SrcKey, params->h_SrcKey, params->size * sizeof(uint), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(params->d_SrcVal, params->h_SrcVal, params->size * sizeof(uint), cudaMemcpyHostToDevice));

		initMergeSort();

		checkCudaErrors(cudaDeviceSynchronize());
		timestamp = mysecond();
	    #ifdef LOGS
	        if (!(params->generate)) start_iteration();
	    #endif
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
	    #ifdef LOGS
	        if (!(params->generate)) end_iteration();
	    #endif
		kernel_time = mysecond() - timestamp;

        if (params->verbose) printf("GPU Kernel time: %.4fs\n", kernel_time);

		timestamp = mysecond();

		checkCudaErrors(cudaMemcpy(params->h_DstKey, params->d_DstKey, params->size * sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(params->h_DstVal, params->d_DstVal, params->size * sizeof(uint), cudaMemcpyDeviceToHost));

		int errors = 0;

		if (params->generate)  {
			printf("Validating output...\n");

			errors += checkKeys(params, numValues);
			errors += checkVals(params);

			if (errors)
				printf("Errors ocurred when validating gold, this is bad. I will save it to file anyway.\n");

			writeOutput(params);
		} else {
			if (compareGoldOutput(params)) {

				printf("Warning! Gold file mismatch detected, proceeding to error analysis...\n");

				errors += checkKeys(params, numValues);
				errors += checkVals(params);
			} else {
				errors = 0;
			}
			#ifdef LOGS
				if (!(params->generate)) log_error_count(errors);
			#endif
		}

		if (params->verbose) printf("Gold check/generate time: %.4fs\n", mysecond() - timestamp);

		closeMergeSort(); // Dealloc some gpu data

        // Display the time between event recordings
        if (params->verbose) printf("Perf: %.3fk elems/sec\n",(float)params->size/(kernel_time*1000.0f));
        if (params->verbose) {
            printf("Iteration %d ended. Elapsed time: %.4fs\n", loop1, mysecond()-globaltimestamp);
        } else {
            printf(".");
        }
        fflush(stdout);
	}
	checkCudaErrors(cudaFree(params->d_SrcVal));
	checkCudaErrors(cudaFree(params->d_SrcKey));
	checkCudaErrors(cudaFree(params->d_BufVal));
	checkCudaErrors(cudaFree(params->d_BufKey));
	checkCudaErrors(cudaFree(params->d_DstVal));
	checkCudaErrors(cudaFree(params->d_DstKey));
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

	exit(EXIT_SUCCESS);
}
