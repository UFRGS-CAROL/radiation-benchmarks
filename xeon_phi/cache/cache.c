#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>     // uint32_t
#include <inttypes.h>   // %"PRIu32"
#include <unistd.h>     // Sleep
#include <time.h>       // Time
#include <omp.h>        // OpenMP

// Xeon Phi Configuration
#define MIC_CORES       (56)            // Max. 56 Cores (+1 core runs de OS)
#define MIC_THREADS     (1*MIC_CORES)   // Max. 4 Threads per Core.
#define MAX_ERROR       32              // Max. number of errors per repetition
#define LOG_SIZE        128             // Line size per error
#define BUSY            5000000         // Repetitions in the busy wait

// =============================================================================
uint64_t string_to_uint64(char *string) {
    uint64_t result = 0;
    char c;

    for (  ; (c = *string ^ '0') <= 9 && c >= 0; ++string) {
        result = result * 10 + c;
    }
    return result;
}

// =============================================================================
int main (int argc, char *argv[]) {

    uint32_t size = 0;
    uint64_t repetitions = 0;

    if(argc != 3) {
        fprintf(stderr,"Please provide the number of <repetitions> and <array size>.\n");
        exit(EXIT_FAILURE);
    }

    repetitions = string_to_uint64(argv[1]);
    size = atoi(argv[2]);

    if (size % (MIC_THREADS * sizeof(uint32_t)) != 0) {
        fprintf(stderr,"The array size needs divisible by %ld (#threads * element size).\n", MIC_THREADS * sizeof(uint32_t));
        exit(EXIT_FAILURE);
    }

    if (repetitions == 0)       repetitions -= 1;   // MAX UINT64_T = 18446744073709551615
    omp_set_num_threads(MIC_THREADS);

    fprintf(stderr,"#HEADER Elem.Size:%"PRIu32"B ", (uint32_t)sizeof(uint32_t));
    fprintf(stderr,"ITEMS:%"PRIu32" ",           (uint32_t)(size / sizeof(uint32_t)));
    fprintf(stderr,"ArraySize:%"PRIu32"KB ",        (uint32_t)(size / 1024));
    fprintf(stderr,"SizePerThread:%"PRIu32"KB ",    (uint32_t)(size / 1024) / MIC_THREADS);
    fprintf(stderr,"Repetitions:%"PRIu64" ",        repetitions);
    fprintf(stderr,"Threads:%"PRIu32"\n",           MIC_THREADS);

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
    uint32_t jump = 0;
    uint32_t slice = (size / sizeof(uint32_t)) / MIC_THREADS ;
    uint32_t errors = 0;

    uint32_t *ptr_vector;
    ptr_vector = (uint32_t *)valloc(size);

    uint32_t x;
    uint32_t y;
    char log[MIC_THREADS][MAX_ERROR][LOG_SIZE];

    //==================================================================
    // Benchmark
    for (i = 0; i < repetitions; i++) {

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
        #pragma offload target(mic) in(ptr_vector:length(size / sizeof(uint32_t)))  inout(log)
        {
            #pragma omp parallel for private(th_id, jump) reduction(+:errors)
            for(th_id = 0; th_id < MIC_THREADS; th_id++)
            {
                asm volatile ("nop");
                asm volatile ("nop");

                uint32_t refw = 0;

                //==============================================================
                // Initialize the variables with a new REFWORD
                if ((i % 3) == 0)
                    asm volatile("movl $0x0, %0" : "=r" (refw));
                else if ((i % 3) == 1)
                    asm volatile("movl $0xFFFFFFFF, %0" : "=r" (refw));
                else
                    asm volatile("movl $0x55555555, %0" : "=r" (refw));

                for (jump = slice * th_id; jump < slice * (th_id + 1); jump++) {
                    ptr_vector[jump] = refw;
                }

                //==============================================================
                // Busy wait
                for(j = (repetitions == 0); j < 5000000; j++) {
                    asm volatile ("nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;");
                    asm volatile ("nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;");
                    asm volatile ("nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;");
                    asm volatile ("nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;");
                    asm volatile ("nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;");
                }

                for (jump = slice * th_id; jump < slice * (th_id + 1); jump++) {

                    //==========================================================
                    // DEBUG: injecting one error (Bit-wise not RefWord)
                    //if(th_id == 0 && i == 0 && jump == 0)
                        //ptr_vector[jump] = ~refw;

                    if (ptr_vector[jump] != refw) {
                        snprintf(log[th_id][errors++], LOG_SIZE,
                                 "%s IT:%"PRIu64" POS:%d TH:%d, REF:0x%08x WAS:0x%08x\n", time, i, j, th_id, refw, ptr_vector[jump]);
                    }

                }

            }
            asm volatile ("nop");
            asm volatile ("nop");
        }

        //======================================================================
        // Write the log if exists
        for (x = 0; x < MIC_THREADS; x++)
            for (y = 0; y < MAX_ERROR; y++)
                if (log[x][y][0] != '\0')
                    fprintf(stderr,"%s", log[x][y]);

    }

    //==================================================================
    // Time stamp
    {
        time_t     now = time(0);
        struct tm  tstruct = *localtime(&now);
        char       time[64];
        strftime(time, sizeof(time), "#FINAL Y:%Y M:%m D:%d Time:%X", &tstruct);
        fprintf(stderr,"%s\n", time);
    }

    exit(EXIT_SUCCESS);
}
