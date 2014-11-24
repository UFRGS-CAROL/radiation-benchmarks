#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>     // uint32_t
#include <inttypes.h>   // %"PRIu32"
#include <unistd.h>     // Sleep
#include <time.h>       // Time
#include <omp.h>        // OpenMP

// Xeon Phi Configuration
#define MIC_CORES       (56)            // Max. 56 Cores (+1 core runs de OS)
#define MIC_THREADS     (4*MIC_CORES)   // Max. 4 Threads per Core.
#define MAX_ERROR       32              // Max. number of errors per repetition
#define LOG_SIZE        128             // Line size per error
#define BUSY            10000000         // Repetitions in the busy wait

#define ITEMS 16     // 64 bytes (512bits) ZMM register / element size

// =============================================================================
uint64_t string_to_uint64(char *string) {
    uint64_t result = 0;
    char c;

    for (  ; (c = *string ^ '0') <= 9 && c >= 0; ++string) {
        result = result * 10 + c;
    }
    return result;
}

//======================================================================
int main(int argc, char *argv[]) {

    uint64_t repetitions = 0;

    if(argc != 2) {
        fprintf(stderr,"Please provide the number of <repetitions>.\n");
        exit(EXIT_FAILURE);
    }

    repetitions = string_to_uint64(argv[1]);
    if (repetitions == 0)       repetitions -= 1;   // MAX UINT64_T = 18446744073709551615
    omp_set_num_threads(MIC_THREADS);

    fprintf(stderr,"#HEADER Repetitions:%"PRIu64" ",    repetitions);
    fprintf(stderr,"Threads:%"PRIu32"\n",               MIC_THREADS);

    //==================================================================
    // Time stamp
    {
        time_t     now = time(0);
        struct tm  tstruct = *localtime(&now);
        char       time[64];
        strftime(time, sizeof(time), "#BEGIN Y:%Y M:%m D:%d Time:%X", &tstruct);
        fprintf(stderr,"%s\n", time);
    }

    //==================================================================
    // Benchmark variables
    uint32_t th_id = 0;
    uint64_t i = 0;
    uint32_t errors = 0;

    uint32_t x;
    uint32_t y;
    char log[MIC_THREADS][MAX_ERROR][LOG_SIZE];

    //==================================================================
    // Benchmark
    for (i = 0; i <= repetitions; i++) {

        //======================================================================
        // Prepare the log
        for (x = 0; x < MIC_THREADS; x++)
            for (y = 0; y < MAX_ERROR; y++)
                log[x][y][0] = '\0';

        errors = 0;

        time_t     now = time(0);
        struct tm  tstruct = *localtime(&now);
        char       time[64];
        strftime(time, sizeof(time), "#ERROR Y:%Y M:%m D:%d Time:%X", &tstruct);

        //======================================================================P
        // Parallel region
        #pragma offload target(mic) inout(log)
        {
            #pragma omp parallel for private(th_id, i) reduction(+:errors)
            for(th_id = 0; th_id < MIC_THREADS; th_id++)
            {
                asm volatile ("nop");
                asm volatile ("nop");
                asm volatile ("nop");

                __declspec(aligned(64)) uint32_t a[ITEMS] ,b[ITEMS], c[ITEMS];

                for (i=0; i<ITEMS; i++) {
                    c[i] = b[i] = a[i] = (uint32_t)refw;
                }

                #pragma vector aligned (a,b,c)
                a[0:ITEMS] = b[0:ITEMS] * c[0:ITEMS] + a[0:ITEMS];

                asm volatile ("nop");
                asm volatile ("nop");
                asm volatile ("nop");
            }
        }
    }

    fprintf(stderr,"Errors: %"PRIu32"\n", errors);
    exit(EXIT_SUCCESS);
}
