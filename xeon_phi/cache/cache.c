#include "../../include/log_helper.h"
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
#define BUSY            20000000        // Repetitions in the busy wait

//#define ALL_DEBUG
#ifdef ALL_DEBUG
    #define DEBUG   if (i==0 && j==0 && errors==0) \
                        asm volatile("movl %1, %0" : "=r" (ptr_vector[j]) : "r" (~ptr_vector[j]));
#else
    #define DEBUG /*OFF*/
#endif

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

    char msg[LOG_SIZE];
    snprintf(msg, sizeof(msg),
            "Repetitions:%"PRIu64" Threads:%"PRIu32" Elem.Size:%"PRIu32"B ArraySize:%"PRIu32"KB SizePerThread:%"PRIu32"KB",
            repetitions,
            MIC_THREADS,
            (uint32_t)sizeof(uint32_t),
            (uint32_t)(size / sizeof(uint32_t)),
            (uint32_t)(size / 1024) / MIC_THREADS);
    start_log_file("cache_mem", msg);
    set_max_errors_iter(MAX_ERROR);


    //==================================================================
    // Benchmark variables
    uint32_t th_id = 0;
    uint64_t i = 0;
    uint32_t j = 0;
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

        start_iteration();
        //======================================================================P
        // Parallel region
        #pragma offload target(mic) in(ptr_vector:length(size / sizeof(uint32_t)))  inout(log)
        {
            #pragma omp parallel for private(th_id, j) reduction(+:errors)
            for(th_id = 0; th_id < MIC_THREADS; th_id++)
            {
                asm volatile ("nop");
                asm volatile ("nop");

                uint32_t ref_int = 0;

                //==============================================================
                // Initialize the variables with a new REFWORD
                if ((i % 3) == 0)
                    asm volatile("movl $0x0, %0" : "=r" (ref_int));
                else if ((i % 3) == 1)
                    asm volatile("movl $0xFFFFFFFF, %0" : "=r" (ref_int));
                else
                    asm volatile("movl $0x55555555, %0" : "=r" (ref_int));

                for (j = slice * th_id; j < slice * (th_id + 1); j++) {
                    ptr_vector[j] = ref_int;
                }

                //==============================================================
                // Busy wait
                for(j = (repetitions == 0); j < BUSY; j++) {
                    asm volatile ("nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;");
                    asm volatile ("nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;");
                    asm volatile ("nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;");
                    asm volatile ("nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;");
                    asm volatile ("nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;");
                }

                for (j = slice * th_id; j < slice * (th_id + 1); j++) {
                    DEBUG

                    if (ptr_vector[j] != ref_int) {
                        snprintf(log[th_id][errors++], LOG_SIZE,
                                 "IT:%"PRIu64" POS:%d TH:%d OP:MEM REF:0x%08x WAS:0x%08x", i, j, th_id, ref_int, ptr_vector[j]);
                    }

                }

            }
            asm volatile ("nop");
            asm volatile ("nop");
        }
        end_iteration();

        //======================================================================
        // Write the log if exists
        for (x = 0; x < MIC_THREADS; x++)
            for (y = 0; y < MAX_ERROR; y++)
                if (log[x][y][0] != '\0')
                    log_error_detail(log[x][y]);

        log_error_count(errors);

    }

    end_log_file();
    exit(EXIT_SUCCESS);
}
