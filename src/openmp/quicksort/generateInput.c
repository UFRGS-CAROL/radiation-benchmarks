#include <stdio.h>
#include <stdlib.h>

#define INPUTSIZE 134217728
int main(){

        unsigned *ndata = (unsigned*)malloc(INPUTSIZE*sizeof(unsigned));//new unsigned[INPUTSIZE];
        printf("Generating input, this will take a long time..."); fflush(stdout);

        //if (!(params->noinputensurance)) {
	//	  	printf("Warning: I will alloc %.2fMB of RAM, be ready to crash.", (double)UINT_MAX * sizeof(unsigned char) / 1000000);
	//		printf(" If this hangs, it can not ensure the input does not have more than 256 equal entries, which may");
	//		printf(" cause really rare issues under radiation tests. Use -noinputensurance in this case.\n");

	//		unsigned char *srcHist;
	//		srcHist = (unsigned char *)malloc(UINT_MAX * sizeof(unsigned char));
	//		if (!srcHist)
	//			fatal("Could not alloc RAM for histogram. Use -noinputensurance.");

	//		register unsigned char max = 0;
	//		register uint valor;
	//		for (uint i = 0; i < INPUTSIZE; i++) {
	//			do {
	//				valor = rand() % UINT_MAX;
	//			} while (srcHist[valor]==UCHAR_MAX);
	//			srcHist[valor]++;
	//			if (srcHist[valor]>max) max = srcHist[valor];
	//		    ndata[i] = valor;
	//		}
	//		free(srcHist);
	//		printf("Maximum repeats of one single key: %d\n", max);
	//	} else {
            for (unsigned int i=0; i<INPUTSIZE; i++)
            {
                // Build data 8 bits at a time
                ndata[i] = 0;
                char *ptr = (char *)&(ndata[i]);

                for (unsigned j=0; j<sizeof(unsigned); j++)
                {
                    *ptr++ = (char)(rand() & 255);
                }
            }
        //}
	char input_name[500];
        snprintf(input_name, 500, "inputsort_%d",INPUTSIZE);
	FILE *finput;
        if (!(finput = fopen(input_name, "wb"))) {
            printf("Warning! Couldn't write the input to file, proceeding anyway...\n");
        } else {
            fwrite(ndata, INPUTSIZE*sizeof(unsigned), 1 , finput);
            fclose(finput);
        }
        printf("Done.\n");
}
