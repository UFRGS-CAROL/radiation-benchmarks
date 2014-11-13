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


#define SIZE 8
// 56 cores * 4 threads = 224
// ITER multiple of 224 to equally schedule threads among cores
#define ITER 22400000

extern double elapsedTime (void);

//======================================================================
int main(int argc, char *argv[]) {

    uint32_t repetitions=0;

    if(argc != 2) {
        printf("Please provide the number of <repetitions>.\n");
        exit(EXIT_FAILURE);
    }

    repetitions = atoi(argv[1]);
    printf("Repetitions:%"PRIu32"\n",           repetitions);

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

            __declspec(aligned(64)) double a[SIZE],b[SIZE],c[SIZE];

            for (i=0; i<SIZE;i++) {
                c[i]=b[i]=a[i]=(double)rand();
            }

            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");

            #pragma vector aligned (a,b,c)
            for(i=0; i<ITER;i++) {
                a[0:SIZE]=b[0:SIZE]*c[0:SIZE]+a[0:SIZE];
            }

            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");


        // need the gold output to chech errors
        // count++;???
        }
    }

    printf("Errors: %"PRIu32"\n", error_count);
    exit(EXIT_SUCCESS);
}
