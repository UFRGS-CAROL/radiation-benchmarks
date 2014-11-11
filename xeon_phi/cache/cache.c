#include "../refword.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

// Xeon Phi Configuration
#define MIC_NUM_CORES       (56)                      // Max. 56 Cores (+1 core runs de OS)
#define MIC_NUM_THREADS     (1*MIC_NUM_CORES)         // Max. 4 Threads per Core.
#define MAX_SIZE            (512*1024*MIC_NUM_CORES)  // Max. 512KB per L2

//======================================================================
#define LOOP_BLOCK {\
                        if ((ptr_vector[jump] ^ ref_word)) { \
                            error_count++; \
                            printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, jump, th_id, ptr_vector[jump] ^ ref_word); \
                            ptr_vector[jump] = ref_word; \
                        } \
                        jump++; \
                    }

//======================================================================
int main (int argc, char *argv[]) {

    uint32_t size = 0;
    uint32_t repetitions = 0;
    uint32_t ref_word = 0;

    if(argc != 4) {
        printf("Please provide the number of <repetitions> and <array size> and <refword option>.\n");
        print_refword();
        exit(EXIT_FAILURE);
    }

    repetitions = atoi(argv[1]);
    size = atoi(argv[2]);
    ref_word = get_refword(atoi(argv[3]));

    if (size % 32 != 0) {
        printf("The array size needs to be divisible by 32 (due to unrolling).\n");
        exit(EXIT_FAILURE);
    }

    if (size > MAX_SIZE) {
        printf("The array size needs to be equal or smaller than %d (due to L2 cache size).\n", MAX_SIZE);
        exit(EXIT_FAILURE);
    }

    if (size % (MIC_NUM_THREADS * 32 * sizeof(uint32_t)) != 0) {
        printf("The array size needs divisible by %d (due to num. threads, unrolling and element size).\n", MIC_NUM_THREADS * 32 * sizeof(uint32_t));
        exit(EXIT_FAILURE);
    }


    printf("Element size: %"PRIu32" bytes\n",   (uint32_t)sizeof(uint32_t));
    printf("Total elements: %"PRIu32"\n",     (uint32_t)(size / sizeof(uint32_t)));
    printf("Total size:%"PRIu32" KB (%"PRIu32" KB/Thread)\n",     size / 1024, (size / 1024) / MIC_NUM_THREADS);
    printf("Repetitions:%"PRIu32"\n",           repetitions);
    printf("Ref Word:0x%08x\n",                 ref_word);

    omp_set_num_threads(MIC_NUM_THREADS);
    printf("Threads:%"PRIu32"\n",               MIC_NUM_THREADS);

    //==================================================================
    // Benchmark variables
    uint32_t th_id = 0;
    uint32_t i = 0;
    uint32_t jump = 0;
    uint32_t slice = (size / sizeof(uint32_t)) / MIC_NUM_THREADS ;
    uint32_t error_count = 0;

    uint32_t *ptr_vector;
    ptr_vector = (uint32_t *)valloc(size);

    //==================================================================
    // Initialize the vector
    for (i = 0; i < (size / sizeof(uint32_t)); i++) {
        ptr_vector[i] = ref_word;
    }

    //==================================================================
    // Start the parallel region
    #pragma offload target(mic) in(ptr_vector:length(size / sizeof(uint32_t)))
    {
        #pragma omp parallel for private(th_id, i, jump) reduction(+:error_count)
        for(th_id = 0; th_id < MIC_NUM_THREADS; th_id++)
        {
            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");

            for (i = 0; i < repetitions; i++) {
                for (jump = slice * th_id; jump < slice * (th_id + 1); ) {

                    // DEBUG: injecting one error
                    //if(th_id == 0 && i == 0 && jump == 0)
                        //ptr_vector[jump] = ~ref_word; // Bit-wise not

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
