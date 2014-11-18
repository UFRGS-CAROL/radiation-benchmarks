#include "../../refword.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <omp.h>

//https://software.intel.com/en-us/articles/using-intel-avx-without-writing-avx

// Xeon Phi Configuration
#define MIC_NUM_CORES       1                      // Max. 56 Cores (+1 core runs de OS)
#define MIC_NUM_THREADS     4*MIC_NUM_CORES         // Max. 4 Threads per Core.
#define MAX_SIZE            512*1024*MIC_NUM_CORES  // Max. 512KB per L2

#define ELEMENTS 8     // 64 bytes (512bits) ZMM register / element size

extern double elapsedTime (void);

//======================================================================
int main(int argc, char *argv[]) {

    uint64_t repetitions = 0;
    uint32_t ref_word = 0;

    if(argc != 3) {
        printf("Please provide the number of <repetitions> and <refword option>.\n");
        print_refword();
        exit(EXIT_FAILURE);
    }

    repetitions = string_to_uint64(argv[1]);
    ref_word = get_refword(atoi(argv[2]));

    printf("Repetitions:%"PRIu64"\n",           repetitions);
    printf("Ref Word:0x%08x\n",                 ref_word);

    omp_set_num_threads(MIC_NUM_THREADS);
    printf("Threads:%"PRIu32"\n",               MIC_NUM_THREADS);

    //==================================================================
    // Benchmark variables
    double startTime,  duration;
    uint32_t th_id = 0;
    uint32_t i = 0;
    uint32_t error_count = 0;

    #pragma offload target(mic)
    {
        #pragma omp parallel for private(th_id, i) reduction(+:error_count)
        for(th_id = 0; th_id < MIC_NUM_THREADS; th_id++)
        {
            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");

            __declspec(aligned(64)) double a[ELEMENTS] ,b[ELEMENTS];

            for (i=0; i<ELEMENTS; i++) {
                b[i] = a[i] = (double)ref_word;
            }

            #pragma vector aligned (a,b)
            for(i = 1; i<repetitions; i++) {
                a[0:ELEMENTS] *= b[0:ELEMENTS];
            }

            for(i = 0; i<ELEMENTS; i++) {
                if (a[i] != pow(ref_word, repetitions)) {
                    error_count++;
                    printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, 0, th_id, a[i]); \
                }
            }

            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");
        }
    }

    printf("Errors: %"PRIu32"\n", error_count);
    exit(EXIT_SUCCESS);
}
