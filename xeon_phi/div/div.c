#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

// Xeon Phi total cores = 57. 1 core probably runs de OS.
#define MIC_NUM_CORES 56000
#define ARRAY_SIZE 56000
#define MAX 32000

double fRand(double fmin, double fmax) {

    double f = (double) rand() / RAND_MAX;

    return (double )fmin + f * (fmax - fmin);
}

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
    double arrayA[ARRAY_SIZE];
    double arrayB[ARRAY_SIZE];
    uint64_t error_count = 0;

    for (i = 0; i < ARRAY_SIZE; i++) {
        arrayA[i] = fRand(-1590.35, 1987.59);
	arrayB[i] = fRand(-15.65, 15.68);
    }

    #pragma offload target(mic) in(arrayA, arrayB) reduction(+:error_count)
    {
        #pragma omp parallel for
        for(j = 0; j < MIC_NUM_CORES; j++)
        {
            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");
            double value=arrayA[j];
            for (i = 0; i < repetitions; i++) {
                value /= arrayB[j];
            }
// injecting one error
//		if(j == 1)
//			value = 1;
            double gold = arrayA[j] / pow(arrayB[j],repetitions);
            if((fabs((float)(value- gold )/value) > 0.00000001)||(fabs((float)(value-gold)/gold) > 0.00000001)){
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
