#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "helper_sorts.h"

int checkKeys(uint *data, uint *outdata, int size)
{
    register unsigned char *srcHist;
    register unsigned char *resHist;

    uint numValues = UINT_MAX;

    int flag = 1;
	int errors = 0;

	register uint index, range;
	long unsigned int control;
	range = ((numValues*sizeof(unsigned char) > 2048000000) ? 2048000000 : numValues); // Avoid more than 1GB of RAM alloc

	srcHist = (unsigned char *)malloc(range * sizeof(unsigned char));
	resHist = (unsigned char *)malloc(range * sizeof(unsigned char));

	if (!srcHist || !resHist) {
		fprintf(stderr, "\nCould not alloc srcHist (%x) or resHist(%x).\n", srcHist, resHist);
		return -1;
	}

    for (index = 0, control = 0; control < numValues; index += range, control += range)
	{
		printf("index = %u range = %u alloc=%.2fMB\n", index, range, 2 * (double)range * sizeof(unsigned char) / 1000000);


        //Build histograms for keys arrays
        memset(srcHist, 0, range * sizeof(unsigned char));
        memset(resHist, 0, range * sizeof(unsigned char));

		register uint indexPLUSrange = index + range;
		register uint *srcKey = data;
		register uint *resKey = outdata;
		register uint i;
		#pragma omp parallel for
		for (i = 0; i < size; i++)
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
		for (i = 0; i < range; i++)
            if (srcHist[i] != resHist[i])
			#pragma omp critical
            {
				char error_detail[150];
                snprintf(error_detail, 150, "The histogram from element %d differs. srcHist=%d dstHist=%d\n", i+index, srcHist[i], resHist[i]);
                #ifdef LOGS
                    if (!(generate)) log_error_detail(error_detail);
                #endif
                printf("ERROR : %s\n", error_detail);
				errors++;
                flag = 0;
            }

	}
	free(resHist);
	free(srcHist);

	//Finally check the ordering
	register uint *resKey = outdata;
	register uint i;
	#pragma omp parallel for
	for (i = 0; i < size - 1; i++)
		if (resKey[i] > resKey[i + 1])
		#pragma omp critical
		{
			char error_detail[150];
			snprintf(error_detail, 150, "Elements not ordered. index=%d %d>%d", i, resKey[i], resKey[i + 1]);
			#ifdef LOGS
				if (!(generate)) log_error_detail(error_detail);
			#endif
			printf("ERROR: %s\n", error_detail);
			errors++;
			flag = 0;
		}

    if (flag) printf("OK\n");
    if (!flag) printf("Errors found.\n");

	return errors;
}
