#include "../../refword.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

// Xeon Phi Configuration
#define MIC_CORES       (56)                      // Max. 56 Cores (+1 core runs de OS)
#define MIC_THREADS     (4*MIC_CORES)         // Max. 4 Threads per Core.

//======================================================================
#define LOOP_BLOCK {\
                        asm ("or %1, %0" : "+r" (value) : "r" (refw));\
                    }

//======================================================================
int main (int argc, char *argv[]) {

    uint64_t repetitions = 0;
    uint32_t refw = 0;

    if(argc != 3) {
        fprintf(stderr,"Please provide the number of <repetitions> and <refword option>.\n");
        print_refword();
        exit(EXIT_FAILURE);
    }

    repetitions = string_to_uint64(argv[1]);
    refw = get_refword(atoi(argv[2]));

    fprintf(stderr,"Repetitions:%"PRIu64"\n",           repetitions);
    fprintf(stderr,"Ref Word:0x%08x\n",                 refw);

    omp_set_num_threads(MIC_THREADS);
    fprintf(stderr,"Threads:%"PRIu32"\n",               MIC_THREADS);

    //==================================================================
    // Benchmark variables
    uint32_t th_id = 0;
    uint64_t i = 0;
    uint32_t errors = 0;
    uint32_t final_refw = refw;

    #pragma offload target(mic)
    {
        #pragma omp parallel for private(th_id, i) reduction(+:errors)
        for(th_id = 0; th_id < MIC_THREADS; th_id++)
        {
            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");

            uint32_t value = refw;

            for (i = 1; i <= repetitions; i++) {

                // DEBUG: injecting one error
                //if(th_id == 0 && i == 0)
                    //value = ~refw; // Bit-wise not

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

                if (value != final_refw) {
                    errors++;
                    fprintf(stderr,"%d it, %d pos, %d thread, 0x%08x syndrome\n", i, 0, th_id, value ^ refw); \
                    value = refw;
                }
            }
            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");
        }
    }

    fprintf(stderr,"Errors: %"PRIu32"\n", errors);
    exit(EXIT_SUCCESS);
}
