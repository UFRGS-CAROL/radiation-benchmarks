#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

// Xeon Phi total cores = 57. 1 core probably runs de OS.
#define MIC_NUM_CORES 56000
#define ARRAY_SIZE 56000
#define MAX 32000
#define refword 0x55555555

//#define rotate(inout) ({ asm ("rol #1,%0" : "=d" (inout)); })


///=============================================================================
int main (int argc, char *argv[]) {
    uint64_t size=0;
    uint64_t repetitions=0;
    if(argc != 2) {
        printf("Please provide the number of repetitions.\n");
        exit(EXIT_FAILURE);
    }

    repetitions = atoi(argv[1]);

    printf("Repetitions:%"PRIu64"\n", repetitions);

    uint64_t i = 0;
    uint64_t j = 0;
    uint64_t error_count = 0;


    #pragma offload target(mic) reduction(+:error_count)
    {
        #pragma omp parallel for
        for(j = 0; j < MIC_NUM_CORES; j++)
        {
            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");
            uint64_t value=refword;
            for (i = 0; i < repetitions; i++) {
                //rotate(value);
                asm ("rol %0,#1" : "=d" (value));
// injecting one error
//		if(i == 1)
//			value = 1;
		if ( i % 32 == 0 && value != refword) {
//			error_count++;
			value = refword;
		}
            }
            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");
        }
    }

    printf("%"PRIu64"\n", error_count);
    exit(EXIT_SUCCESS);
}
