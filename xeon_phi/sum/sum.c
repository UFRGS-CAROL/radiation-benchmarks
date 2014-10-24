#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

// Xeon Phi total cores = 57. 1 core probably runs de OS.
#define MIC_NUM_CORES 56000
#define ARRAY_SIZE 56000
#define MAX 32000

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
    uint64_t arrayA[ARRAY_SIZE];
    uint64_t arrayB[ARRAY_SIZE];
    uint64_t error_count = 0;

    for (i = 0; i < ARRAY_SIZE; i++) {
        arrayA[i] = rand() % MAX;
	arrayB[i] = rand() % MAX;
    }

    #pragma offload target(mic) in(arrayA, arrayB) reduction(+:error_count)
    {
        #pragma omp parallel for
        for(j = 0; j < MIC_NUM_CORES; j++)
        {
            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");
            int value=arrayA[j];
            for (i = 0; i < repetitions; i++) {
                value += arrayB[j];
            }
// injecting one error
//		if(j == 1)
//			value = 1;
            if(arrayA[j]+(arrayB[j]*repetitions) != value){
                error_count++;
            }
            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");
        }
    }

    printf("%"PRIu64"\n", error_count);
    exit(EXIT_SUCCESS);
}
