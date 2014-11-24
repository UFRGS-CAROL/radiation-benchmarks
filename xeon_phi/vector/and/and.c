#include "../../refword.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <omp.h>

//https://software.intel.com/en-us/articles/using-intel-avx-without-writing-avx

// Xeon Phi Configuration
#define MIC_CORES       1                      // Max. 56 Cores (+1 core runs de OS)
#define MIC_THREADS     4*MIC_CORES         // Max. 4 Threads per Core.
#define MAX_SIZE            512*1024*MIC_CORES  // Max. 512KB per L2

#define ITEMS 16     // 64 bytes (512bits) ZMM register / element size

extern double elapsedTime (void);

//======================================================================
int main(int argc, char *argv[]) {

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
    double startTime,  duration;
    uint32_t th_id = 0;
    uint64_t i = 0;
    uint32_t errors = 0;
    uint32_t ones;

    asm volatile("movl $0xFFFFFFFF, %0" : "=r" (ones));

    #pragma offload target(mic)
    {
        #pragma omp parallel for private(th_id, i) reduction(+:errors)
        for(th_id = 0; th_id < MIC_THREADS; th_id++)
        {
            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");

            __declspec(aligned(64)) uint32_t a[ITEMS] ,b[ITEMS];

            for (i=0; i<ITEMS; i++) {
                b[i] = a[i] = (uint32_t)refw;
            }

            #pragma vector aligned (a,b)
            for(i = 1; i < repetitions; i++) {
                a[0:ITEMS] = b[0:ITEMS] & ones;
                b[0:ITEMS] = a[0:ITEMS] & ones;
            }

            for(i = 0; i<ITEMS; i++) {
                if (a[i] != refw) {
                    errors++;
                    fprintf(stderr,"%d it, %d pos, %d thread, 0x%08x syndrome\n", i, 0, th_id, a[i]); \
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
