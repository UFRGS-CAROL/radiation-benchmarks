#include "../../../../include/log_helper.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>     // uint32_t
#include <inttypes.h>   // %"PRIu32"
#include <unistd.h>     // Sleep
#include <time.h>       // Time
#include <omp.h>        // OpenMP
#include <sched.h>      // sched_getcpu
#include "offload.h"    // omp_set_num_threads_target


// Xeon Phi Configuration
#define MIC_CORES       (56)            // Max. 56 Cores (+1 core runs de OS)
#define MIC_THREADS     (4*MIC_CORES)   // Max. 4 Threads per Core.
#define MAX_ERROR       32              // Max. number of errors per repetition
#define LOG_SIZE        128             // Line size per error
#define BUSY            10000000         // Repetitions in the busy wait

//#define ALL_DEBUG
#ifdef ALL_DEBUG
    #define DEBUG   if (i==0 && errors==0) \
                        asm volatile("movl %1, %0" : "=r" (count0) : "r" (~count0));\
                    if (i == 10) while(1);
#else
    #define DEBUG /*OFF*/
#endif

#ifdef ALL_DEBUG
    __declspec(target(mic)) sched_getcpu();
#endif

//======================================================================
#define LOOP_BLOCK(V) {\
        DEBUG \
        if (count##V != ref_int) { \
            snprintf(log[th_id][errors++], LOG_SIZE, "IT:%"PRIu64" POS:%d TH:%d OP:REG REF:0x%08x WAS:0x%08x", i, j, th_id, ref_int, count##V); \
        } \
                    }

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
int main (int argc, char *argv[]) {

    uint64_t repetitions = 0;

    if(argc != 2) {
        fprintf(stderr,"Please provide the number of <repetitions> (0 for MAX).\n");
        exit(EXIT_FAILURE);
    }

    repetitions = string_to_uint64(argv[1]);
    if (repetitions == 0)       repetitions -= 1;   // MAX UINT64_T = 18446744073709551615
    omp_set_num_threads_target(TARGET_MIC, 0, MIC_THREADS);

    char msg[LOG_SIZE];
    snprintf(msg, sizeof(msg), "Loop:%"PRIu64" Threads:%"PRIu32"", repetitions, MIC_THREADS);
    if (start_log_file("scalar_reg", msg) != 0) {
        exit(EXIT_FAILURE);
    }
    set_max_errors_iter(MAX_ERROR);

    //==================================================================
    // Benchmark variables
    uint32_t th_id = 0;
    uint64_t i = 0;
    uint64_t j = 0;
    uint32_t errors = 0;

    uint32_t x;
    uint32_t y;
    char log[MIC_THREADS][MAX_ERROR][LOG_SIZE];

    #ifdef ALL_DEBUG
        printf("Before offload (local processor): Thread %d, on cpu %d.\n", omp_get_thread_num(), sched_getcpu());
    #endif

    //==================================================================
    // Benchmark
    for (i = 0; i <= repetitions; i++) {

        //======================================================================
        // Prepare the log
        for (x = 0; x < MIC_THREADS; x++)
            for (y = 0; y < MAX_ERROR; y++)
                log[x][y][0] = '\0';

        errors = 0;

        start_iteration();
        //======================================================================P
        // Parallel region
        #pragma offload target(mic) inout(log)
        {
            #pragma omp parallel for private(th_id, j) reduction(+:errors)
            for(th_id = 0; th_id < MIC_THREADS; th_id++)
            {
                asm volatile ("nop");
                asm volatile ("nop");

                #ifdef ALL_DEBUG
                    printf("After offload: Thread %d, on cpu %d.\n", omp_get_thread_num(), sched_getcpu());
                #endif

                uint32_t ref_int = 0;

                uint32_t count0  = 0 ;
                uint32_t count1  = 0 ;
                uint32_t count2  = 0 ;
                uint32_t count3  = 0 ;
                uint32_t count4  = 0 ;
                uint32_t count5  = 0 ;
                uint32_t count6  = 0 ;
                uint32_t count7  = 0 ;

                //==============================================================
                // Initialize the variables with a new REFWORD
                if ((i % 3) == 0)
                    asm volatile("movl $0x0, %0" : "=r" (ref_int));
                else if ((i % 3) == 1)
                    asm volatile("movl $0xFFFFFFFF, %0" : "=r" (ref_int));
                else
                    asm volatile("movl $0x55555555, %0" : "=r" (ref_int));

                asm volatile("mov %1, %0" : "=r" (count0) : "r" (ref_int) : );
                asm volatile("mov %1, %0" : "=r" (count1) : "r" (ref_int) : );
                asm volatile("mov %1, %0" : "=r" (count2) : "r" (ref_int) : );
                asm volatile("mov %1, %0" : "=r" (count3) : "r" (ref_int) : );
                asm volatile("mov %1, %0" : "=r" (count4) : "r" (ref_int) : );
                asm volatile("mov %1, %0" : "=r" (count5) : "r" (ref_int) : );
                asm volatile("mov %1, %0" : "=r" (count6) : "r" (ref_int) : );
                asm volatile("mov %1, %0" : "=r" (count7) : "r" (ref_int) : );

                //==============================================================
                // Busy wait
                for(j = (repetitions == 0); j < BUSY; j++) {
                    asm volatile ("nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;");
                    asm volatile ("nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;");
                    asm volatile ("nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;");
                    asm volatile ("nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;");
                    asm volatile ("nop;nop;nop;nop;nop;nop;nop;nop;nop;nop;");
                }

                //==========================================================
                // DEBUG: injecting one error (Bit-wise not RefWord)
                //if(th_id == 0 && i == 0)
                    //count0 = ~ref_int;

                LOOP_BLOCK(0)
                LOOP_BLOCK(1)
                LOOP_BLOCK(2)
                LOOP_BLOCK(3)
                LOOP_BLOCK(4)
                LOOP_BLOCK(5)
                LOOP_BLOCK(6)
                LOOP_BLOCK(7)

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
