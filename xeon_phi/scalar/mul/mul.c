#include "../../refword.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

// Xeon Phi Configuration
#define MIC_NUM_CORES       (56)                      // Max. 56 Cores (+1 core runs de OS)
#define MIC_NUM_THREADS     (4*MIC_NUM_CORES)         // Max. 4 Threads per Core.

//======================================================================
#define LOOP_BLOCK {\
                    asm ("imul %1, %0" : "+r" (value) : "r" (ref_word));\
                    }

//======================================================================
int main (int argc, char *argv[]) {

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
    uint32_t th_id = 0;
    uint64_t i = 0;
    uint32_t error_count = 0;
    uint32_t final_ref_word = pow(ref_word, 32);

    #pragma offload target(mic)
    {
        #pragma omp parallel for private(th_id, i) reduction(+:error_count)
        for(th_id = 0; th_id < MIC_NUM_THREADS; th_id++)
        {
            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");

            uint32_t value = ref_word;

            for (i = 1; i <= repetitions; i++) {

                // DEBUG: injecting one error
                //if(th_id == 0 && i == 0)
                    //value = ~ref_word; // Bit-wise not

                LOOP_BLOCK
                LOOP_BLOCK
                LOOP_BLOCK
                LOOP_BLOCK

                LOOP_BLOCK
                LOOP_BLOCK
                LOOP_BLOCK
                LOOP_BLOCK

                LOOP_BLOCK
                LOOP_BLOCK
                LOOP_BLOCK
                LOOP_BLOCK

                LOOP_BLOCK
                LOOP_BLOCK
                LOOP_BLOCK
                LOOP_BLOCK

                LOOP_BLOCK
                LOOP_BLOCK
                LOOP_BLOCK
                LOOP_BLOCK

                LOOP_BLOCK
                LOOP_BLOCK
                LOOP_BLOCK
                LOOP_BLOCK

                LOOP_BLOCK
                LOOP_BLOCK
                LOOP_BLOCK
                LOOP_BLOCK

                LOOP_BLOCK
                LOOP_BLOCK
                LOOP_BLOCK
                LOOP_BLOCK

                if (value != final_ref_word) {
                    error_count++;
                    printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, 0, th_id, value); \
                }
                value = ref_word;
            }
            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");
        }
    }

    printf("Errors: %"PRIu32"\n", error_count);
    exit(EXIT_SUCCESS);
}
