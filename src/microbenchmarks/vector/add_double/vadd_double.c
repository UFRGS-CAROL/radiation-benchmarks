#include "../../../include/log_helper.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>     // uint32_t
#include <inttypes.h>   // %"PRIu32"
#include <unistd.h>     // Sleep
#include <time.h>       // Time
#include <math.h>       // pow
#include <omp.h>        // OpenMP
#include <sched.h>      // sched_getcpu

#define MAX_ERROR       32            // Max. number of errors per repetition
#define LOG_SIZE        128             // Line size per error
#define BUSY            1000000          // Repetitions in the busy wait

//#define ITEMS_INT           16              // 64 bytes (512bits) ZMM register / element size
#define ITEMS_INT           8              // 32 bytes (256bits) YMM register / element size
//#define ITEMS_FPD           8              // 64 bytes (512bits) ZMM register / element size
#define ITEMS_FPD           4              // 32 bytes (256bits) YMM register / element size

//#define ALL_DEBUG
#ifdef ALL_DEBUG
    #define DEBUG_INT   if (i==0 && j==0 && errors==0) vec_int[0] = ~vec_int[0];\
                    if (i == 10) while(1);
    #define DEBUG_FPD   if (i==0 && j==0 && errors==0) vec_fpd[0] = vec_fpd[0]-1;\
                    if (i == 10) while(1);
#else
    #define DEBUG_INT /*OFF*/
    #define DEBUG_FPD /*OFF*/
#endif



#define LOOP_ADD {\
        value_fpd = 0.0; \
        asm volatile("vbroadcastsd %0, %%ymm2" :  : "m" (ref_fpd1) : "ymm2"); \
        asm volatile("vbroadcastsd %0, %%ymm3" :  : "m" (value_fpd) : "ymm3"); \
        \
        asm volatile("vaddpd %%ymm3, %%ymm2, %%ymm3" : : : "ymm2", "ymm3");\
        asm volatile("vaddpd %%ymm3, %%ymm2, %%ymm3" : : : "ymm2", "ymm3");\
        asm volatile("vaddpd %%ymm3, %%ymm2, %%ymm3" : : : "ymm2", "ymm3");\
        asm volatile("vaddpd %%ymm3, %%ymm2, %%ymm3" : : : "ymm2", "ymm3");\
        asm volatile("vaddpd %%ymm3, %%ymm2, %%ymm3" : : : "ymm2", "ymm3");\
        asm volatile("vaddpd %%ymm3, %%ymm2, %%ymm3" : : : "ymm2", "ymm3");\
        asm volatile("vaddpd %%ymm3, %%ymm2, %%ymm3" : : : "ymm2", "ymm3");\
        asm volatile("vaddpd %%ymm3, %%ymm2, %%ymm3" : : : "ymm2", "ymm3");\
        \
        asm volatile("vaddpd %%ymm3, %%ymm2, %%ymm3" : : : "ymm2", "ymm3");\
        asm volatile("vaddpd %%ymm3, %%ymm2, %%ymm3" : : : "ymm2", "ymm3");\
        asm volatile("vaddpd %%ymm3, %%ymm2, %%ymm3" : : : "ymm2", "ymm3");\
        asm volatile("vaddpd %%ymm3, %%ymm2, %%ymm3" : : : "ymm2", "ymm3");\
        asm volatile("vaddpd %%ymm3, %%ymm2, %%ymm3" : : : "ymm2", "ymm3");\
        asm volatile("vaddpd %%ymm3, %%ymm2, %%ymm3" : : : "ymm2", "ymm3");\
        asm volatile("vaddpd %%ymm3, %%ymm2, %%ymm3" : : : "ymm2", "ymm3");\
        asm volatile("vaddpd %%ymm3, %%ymm2, %%ymm3" : : : "ymm2", "ymm3");\
        \
        asm volatile("vmovapd %%ymm3, %0" : "=m" (vec_fpd[0]) : : "ymm3"); \
        DEBUG_FPD \
        for(k = 0; k < ITEMS_FPD; k++) { \
            if (fabs(vec_fpd[k] - (ref_fpd1 * 16)) > 0.00000000000001) \
                snprintf(log[th_id][errors++], LOG_SIZE, "IT:%"PRIu64" POS:%d TH:%d OP:ADD REF:%.14f WAS:%.14f", i, k, th_id, (ref_fpd1 * 16), vec_fpd[k]); \
        } \
                }



//======================================================================
// Linear Feedback Shift Register using 32 bits and XNOR. Details at:
// http://www.xilinx.com/support/documentation/application_notes/xapp052.pdf
// http://www.ece.cmu.edu/~koopman/lfsr/index.html
void ref_word(uint32_t *ref_int1, double *ref_fpd1){
    static uint32_t counter = 0;

    counter++;
    if (counter == 1){
        *ref_int1  = 0xCCCCCCCC;   // 1100 1100 1100 1100 | 1100 1100 1100 1100 (3435973836)
        *ref_fpd1  = 1.33333333333333325931846502499E0;   // 00111111 11110101 01010101 01010101 01010101 01010101 01010101 01010101
        return;
    }
    else if (counter == 2){
        *ref_int1  = 0x66666666;   // 0110 0110 0110 0110 | 0110 0110 0110 0110
        *ref_fpd1  = 1.42857142857142838110462434997E0 ;   // 00111111 11110110 11011011 01101101 10110110 11011011 01101101 10110110
        return;
    }
    else if (counter == 3){
        *ref_int1  = 0x33333333;   // 0011 0011 0011 0011 | 0011 0011 0011 0011
        *ref_fpd1  = 1.571428571428571396850770725E0 ;   // 00111111 11111001 00100100 10010010 01001001 00100100 10010010 01001001

        return;
    }
    else if (counter == 4){
        *ref_int1  = 0xAAAAAAAA;   // 1010 1010 1010 1010 | 1010 1010 1010 1010
        *ref_fpd1  = 1.85714285714285698425385362498E0 ;   // 00111111 11111101 10110110 11011011 01101101 10110110 11011011 01101101

        return;
    }
    else if (counter == 5){
        *ref_int1  = 0x55555555;   // 0101 0101 0101 0101 | 0101 0101 0101 0101
        *ref_fpd1  = 1.99999999999999977795539507497E0 ;   // 00111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111

        return;
    }
    else if (counter == 6) {
        *ref_int1  = 0x99999999;   // 1001 1001 1001 1001 | 1001 1001 1001 1001
        *ref_fpd1  = -1.33333333333333325931846502499E0;   // 00111111 11110101 01010101 01010101 01010101 01010101 01010101 01010101
        return;
    }
    else if (counter == 7){
        *ref_int1  = 0x88888888;   // 1000 1000 1000 1000 | 1000 1000 1000 1000
        *ref_fpd1  = -1.42857142857142838110462434997E0 ;   // 00111111 11110110 11011011 01101101 10110110 11011011 01101101 10110110
        return;
    }
    else if (counter == 8){
        *ref_int1  = 0x44444444;   // 0100 0100 0100 0100 | 0100 0100 0100 0100
        *ref_fpd1  = -1.571428571428571396850770725E0 ;   // 00111111 11111001 00100100 10010010 01001001 00100100 10010010 01001001
        return;
    }
    else if (counter == 9){
        *ref_int1  = 0x22222222;   // 0010 0010 0010 0010 | 0010 0010 0010 0010
        *ref_fpd1  = -1.85714285714285698425385362498E0 ;   // 00111111 11111101 10110110 11011011 01101101 10110110 11011011 01101101
        return;
    }
    else {
        *ref_int1  = 0x11111111;  // 0001 0001 0001 0001 | 0001 0001 0001 0001
        *ref_fpd1  = -1.99999999999999977795539507497E0 ;   // 10111111 11111111 11111111 11111111 11111111 11111111 11111111 11111111
        counter = 0;
        return;
    }
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
    uint64_t num_threads = 0;

    if(argc != 3) {
        fprintf(stderr,"Please provide the number of <repetitions> (0 for MAX) and number of <threads>.\n");
        exit(EXIT_FAILURE);
    }

    repetitions = string_to_uint64(argv[1]);
    num_threads = string_to_uint64(argv[2]);
    if (repetitions == 0)       repetitions -= 1;   // MAX UINT64_T = 18446744073709551615
    //omp_set_num_threads_target(TARGET_MIC, 0, num_threads);
    omp_set_num_threads(num_threads);


    char msg[LOG_SIZE];
    snprintf(msg, sizeof(msg), "Loop:%"PRIu64" Threads:%"PRIu32"", repetitions, num_threads);
    if (start_log_file("vadd_double", msg) != 0) {
        exit(EXIT_FAILURE);
    }
    set_max_errors_iter(MAX_ERROR);
    set_iter_interval_print(10);

    //==================================================================
    // Benchmark variables
    uint32_t th_id = 0;
    uint64_t i = 0;
    uint64_t j = 0;
    uint64_t k = 0;
    uint32_t errors = 0;

    uint32_t x;
    uint32_t y;
    char log[num_threads][MAX_ERROR][LOG_SIZE];


    //==================================================================
    // Benchmark
    for (i = 0; i <= repetitions; i++) {

        //======================================================================
        // Prepare the log
        for (x = 0; x < num_threads; x++)
            for (y = 0; y < MAX_ERROR; y++)
                log[x][y][0] = '\0';

        errors = 0;

        //==============================================================
        // Initialize the variables with a new REFWORD
        uint32_t ref_int1;
        double ref_fpd1;
        ref_word(&ref_int1, &ref_fpd1);

        start_iteration();
        //======================================================================P
        // Parallel region
            #pragma omp parallel for private(th_id, j, k) firstprivate(ref_int1, ref_fpd1) reduction(+:errors)
            for(th_id = 0; th_id < num_threads; th_id++)
            {
                asm volatile ("nop");
                asm volatile ("nop");


                double value_fpd;
                double vec_fpd[ITEMS_FPD];

                    for(j = (repetitions == 0); j < BUSY; j++) {
                        LOOP_ADD
                        LOOP_ADD
                        LOOP_ADD
                        LOOP_ADD
                        LOOP_ADD
                        LOOP_ADD
                        LOOP_ADD
                        LOOP_ADD
                    }

            }
            asm volatile ("nop");
            asm volatile ("nop");
        end_iteration();

        //======================================================================
        // Write the log if exists
        for (x = 0; x < num_threads; x++)
            for (y = 0; y < MAX_ERROR; y++)
                if (log[x][y][0] != '\0')
                    log_error_detail(log[x][y]);

        log_error_count(errors);

    }

    end_log_file();
    exit(EXIT_SUCCESS);
}
