#include "../../../include/log_helper.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>     // uint32_t
#include <inttypes.h>   // %"PRIu32"
#include <unistd.h>     // Sleep
#include <time.h>       // Time
#include <omp.h>        // OpenMP
#include <math.h>       // pow

// Xeon Phi Configuration
#define MIC_CORES       (56)             // Max. 56 Cores (+1 core runs de OS)
#define MIC_THREADS     (4*MIC_CORES)   // Max. 4 Threads per Core.
#define MAX_ERROR       32            // Max. number of errors per repetition
#define LOG_SIZE        128             // Line size per error
#define BUSY            300000          // Repetitions in the busy wait

#define ITEMS_INT           16              // 64 bytes (512bits) ZMM register / element size

// ~ #define DEBUG_INT           if (i==0 && j==0 && errors==0) vec_int[0] = ~vec_int[0];
#define DEBUG_INT /*OFF*/

//======================================================================
#define LOOP_AND {\
        value_int = 0xFFFFFFFF; \
        asm volatile("vpbroadcastd %0, %%zmm0" :  : "m" (ref_int1) : "zmm0"); \
        asm volatile("vpbroadcastd %0, %%zmm1" :  : "m" (value_int) : "zmm1"); \
        \
        asm volatile("vpandd %%zmm1, %%zmm0, %%zmm0" : : : "zmm0", "zmm1");\
        asm volatile("vpandd %%zmm1, %%zmm0, %%zmm0" : : : "zmm0", "zmm1");\
        asm volatile("vpandd %%zmm1, %%zmm0, %%zmm0" : : : "zmm0", "zmm1");\
        asm volatile("vpandd %%zmm1, %%zmm0, %%zmm0" : : : "zmm0", "zmm1");\
        asm volatile("vpandd %%zmm1, %%zmm0, %%zmm0" : : : "zmm0", "zmm1");\
        asm volatile("vpandd %%zmm1, %%zmm0, %%zmm0" : : : "zmm0", "zmm1");\
        asm volatile("vpandd %%zmm1, %%zmm0, %%zmm0" : : : "zmm0", "zmm1");\
        asm volatile("vpandd %%zmm1, %%zmm0, %%zmm0" : : : "zmm0", "zmm1");\
        \
        asm volatile("vpandd %%zmm1, %%zmm0, %%zmm0" : : : "zmm0", "zmm1");\
        asm volatile("vpandd %%zmm1, %%zmm0, %%zmm0" : : : "zmm0", "zmm1");\
        asm volatile("vpandd %%zmm1, %%zmm0, %%zmm0" : : : "zmm0", "zmm1");\
        asm volatile("vpandd %%zmm1, %%zmm0, %%zmm0" : : : "zmm0", "zmm1");\
        asm volatile("vpandd %%zmm1, %%zmm0, %%zmm0" : : : "zmm0", "zmm1");\
        asm volatile("vpandd %%zmm1, %%zmm0, %%zmm0" : : : "zmm0", "zmm1");\
        asm volatile("vpandd %%zmm1, %%zmm0, %%zmm0" : : : "zmm0", "zmm1");\
        asm volatile("vpandd %%zmm1, %%zmm0, %%zmm0" : : : "zmm0", "zmm1");\
        \
        asm volatile("vmovdqa32 %%zmm0, %0" : "=m" (vec_int[0]) : : "zmm0"); \
        DEBUG_INT \
        for(k = 0; k < ITEMS_INT; k++) { \
            if (vec_int[k] != ref_int1) \
                snprintf(log[th_id][errors++], LOG_SIZE, "IT:%"PRIu64" POS:%d TH:%d OP:AND REF:0x%08x WAS:0x%08x", i, k, th_id, ref_int1, vec_int[k]); \
        } \
                }

#define LOOP_ADD {\
        value_int = 0; \
        asm volatile("vpbroadcastd %0, %%zmm2" :  : "m" (ref_int2) : "zmm2"); \
        asm volatile("vpbroadcastd %0, %%zmm3" :  : "m" (value_int) : "zmm3"); \
        \
        asm volatile("vpaddd %%zmm3, %%zmm2, %%zmm3" : : : "zmm2", "zmm3");\
        asm volatile("vpaddd %%zmm3, %%zmm2, %%zmm3" : : : "zmm2", "zmm3");\
        asm volatile("vpaddd %%zmm3, %%zmm2, %%zmm3" : : : "zmm2", "zmm3");\
        asm volatile("vpaddd %%zmm3, %%zmm2, %%zmm3" : : : "zmm2", "zmm3");\
        asm volatile("vpaddd %%zmm3, %%zmm2, %%zmm3" : : : "zmm2", "zmm3");\
        asm volatile("vpaddd %%zmm3, %%zmm2, %%zmm3" : : : "zmm2", "zmm3");\
        asm volatile("vpaddd %%zmm3, %%zmm2, %%zmm3" : : : "zmm2", "zmm3");\
        asm volatile("vpaddd %%zmm3, %%zmm2, %%zmm3" : : : "zmm2", "zmm3");\
        \
        asm volatile("vpaddd %%zmm3, %%zmm2, %%zmm3" : : : "zmm2", "zmm3");\
        asm volatile("vpaddd %%zmm3, %%zmm2, %%zmm3" : : : "zmm2", "zmm3");\
        asm volatile("vpaddd %%zmm3, %%zmm2, %%zmm3" : : : "zmm2", "zmm3");\
        asm volatile("vpaddd %%zmm3, %%zmm2, %%zmm3" : : : "zmm2", "zmm3");\
        asm volatile("vpaddd %%zmm3, %%zmm2, %%zmm3" : : : "zmm2", "zmm3");\
        asm volatile("vpaddd %%zmm3, %%zmm2, %%zmm3" : : : "zmm2", "zmm3");\
        asm volatile("vpaddd %%zmm3, %%zmm2, %%zmm3" : : : "zmm2", "zmm3");\
        asm volatile("vpaddd %%zmm3, %%zmm2, %%zmm3" : : : "zmm2", "zmm3");\
        \
        asm volatile("vmovdqa32 %%zmm3, %0" : "=m" (vec_int[0]) : : "zmm3"); \
        DEBUG_INT \
        for(k = 0; k < ITEMS_INT; k++) { \
            if (vec_int[k] != (ref_int2 << 4)) \
                snprintf(log[th_id][errors++], LOG_SIZE, "IT:%"PRIu64" POS:%d TH:%d OP:ADD REF:0x%08x WAS:0x%08x", i, k, th_id, (ref_int2 << 4), vec_int[k]); \
        } \
                }

#define LOOP_MUL {\
        value_int = 0x2; \
        asm volatile("vpbroadcastd %0, %%zmm4" :  : "m" (value_int) : "zmm4"); \
        asm volatile("vpbroadcastd %0, %%zmm5" :  : "m" (ref_int3) : "zmm5"); \
        \
        asm volatile("vpmulld %%zmm5, %%zmm4, %%zmm5" : : : "zmm4", "zmm5");\
        asm volatile("vpmulld %%zmm5, %%zmm4, %%zmm5" : : : "zmm4", "zmm5");\
        asm volatile("vpmulld %%zmm5, %%zmm4, %%zmm5" : : : "zmm4", "zmm5");\
        asm volatile("vpmulld %%zmm5, %%zmm4, %%zmm5" : : : "zmm4", "zmm5");\
        asm volatile("vpmulld %%zmm5, %%zmm4, %%zmm5" : : : "zmm4", "zmm5");\
        asm volatile("vpmulld %%zmm5, %%zmm4, %%zmm5" : : : "zmm4", "zmm5");\
        asm volatile("vpmulld %%zmm5, %%zmm4, %%zmm5" : : : "zmm4", "zmm5");\
        asm volatile("vpmulld %%zmm5, %%zmm4, %%zmm5" : : : "zmm4", "zmm5");\
        \
        asm volatile("vmovdqa32 %%zmm5, %0" : "=m" (vec_int[0]) : : "zmm5"); \
        DEBUG_INT \
        for(k = 0; k < ITEMS_INT; k++) { \
            if (vec_int[k] != (ref_int3 << 8)) \
                snprintf(log[th_id][errors++], LOG_SIZE, "IT:%"PRIu64" POS:%d TH:%d OP:MUL REF:0x%08x WAS:0x%08x", i, k, th_id, (ref_int3 << 8), vec_int[k]); \
        } \
                }


//======================================================================
// Linear Feedback Shift Register using 32 bits and XNOR. Details at:
// http://www.xilinx.com/support/documentation/application_notes/xapp052.pdf
// http://www.ece.cmu.edu/~koopman/lfsr/index.html
void ref_word(uint32_t *ref_int1, uint32_t *ref_int2, uint32_t *ref_int3){
    static uint32_t counter = 0;

    counter++;
    if (counter == 1){
        *ref_int1  = 0xCCCCCCCC;   // 1100 1100 1100 1100 | 1100 1100 1100 1100 (3435973836)
        *ref_int2  = 0x0CCCCCCC;   // 0000 1100 1100 1100 | 1100 1100 1100 1100
        *ref_int3  = 0x00CCCCCC;   // 0000 0000 1100 1100 | 1100 1100 1100 1100
        return;
    }
    else if (counter == 2){
        *ref_int1  = 0x66666666;   // 0110 0110 0110 0110 | 0110 0110 0110 0110
        *ref_int2  = 0x06666666;   // 0000 0110 0110 0110 | 0110 0110 0110 0110
        *ref_int3  = 0x06666666;   // 0000 0000 0110 0110 | 0110 0110 0110 0110
        return;
    }
    else if (counter == 3){
        *ref_int1  = 0x33333333;   // 0011 0011 0011 0011 | 0011 0011 0011 0011
        *ref_int2  = 0x03333333;   // 0000 0011 0011 0011 | 0011 0011 0011 0011
        *ref_int3  = 0x00333333;   // 0000 0000 0011 0011 | 0011 0011 0011 0011
        return;
    }
    else if (counter == 4){
        *ref_int1  = 0xAAAAAAAA;   // 1010 1010 1010 1010 | 1010 1010 1010 1010
        *ref_int2  = 0x0AAAAAAA;   // 0000 1010 1010 1010 | 1010 1010 1010 1010
        *ref_int3  = 0x00AAAAAA;   // 0000 0000 1010 1010 | 1010 1010 1010 1010
        return;
    }
    else if (counter == 5){
        *ref_int1  = 0x55555555;   // 0101 0101 0101 0101 | 0101 0101 0101 0101
        *ref_int2  = 0x05555555;   // 0000 0101 0101 0101 | 0101 0101 0101 0101
        *ref_int3  = 0x00555555;   // 0000 0000 0101 0101 | 0101 0101 0101 0101
        return;
    }
    else if (counter == 6) {
        *ref_int1  = 0x99999999;   // 1001 1001 1001 1001 | 1001 1001 1001 1001
        *ref_int2  = 0x09999999;   // 0000 1001 1001 1001 | 1001 1001 1001 1001
        *ref_int3  = 0x00999999;   // 0000 0000 1001 1001 | 1001 1001 1001 1001
        return;
    }
    else if (counter == 7){
        *ref_int1  = 0x88888888;   // 1000 1000 1000 1000 | 1000 1000 1000 1000
        *ref_int2  = 0x08888888;   // 0000 1000 1000 1000 | 1000 1000 1000 1000
        *ref_int3  = 0x00888888;   // 0000 0000 1000 1000 | 1000 1000 1000 1000
        return;
    }
    else if (counter == 8){
        *ref_int1  = 0x44444444;   // 0100 0100 0100 0100 | 0100 0100 0100 0100
        *ref_int2  = 0x04444444;   // 0000 0100 0100 0100 | 0100 0100 0100 0100
        *ref_int3  = 0x00444444;   // 0000 0000 0100 0100 | 0100 0100 0100 0100
        return;
    }
    else if (counter == 9){
        *ref_int1  = 0x22222222;   // 0010 0010 0010 0010 | 0010 0010 0010 0010
        *ref_int2  = 0x02222222;   // 0000 0010 0010 0010 | 0010 0010 0010 0010
        *ref_int3  = 0x00222222;   // 0000 0000 0010 0010 | 0010 0010 0010 0010
        return;
    }
    else {
        *ref_int1  = 0x11111111;  // 0001 0001 0001 0001 | 0001 0001 0001 0001
        *ref_int2  = 0x01111111;  // 0000 0001 0001 0001 | 0001 0001 0001 0001
        *ref_int3  = 0x00111111;  // 0000 0000 0001 0001 | 0001 0001 0001 0001
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

    if(argc != 2) {
        fprintf(stderr,"Please provide the number of <repetitions> (0 for MAX).\n");
        exit(EXIT_FAILURE);
    }

    repetitions = string_to_uint64(argv[1]);
    if (repetitions == 0)       repetitions -= 1;   // MAX UINT64_T = 18446744073709551615
    omp_set_num_threads(MIC_THREADS);

    char msg[LOG_SIZE];
    snprintf(msg, sizeof(msg), "Repetitions:%"PRIu64" Threads:%"PRIu32"", repetitions, MIC_THREADS);
    start_log_file("vector_and_int", msg);
    set_max_errors_iter(MAX_ERROR);

    //==================================================================
    // Benchmark variables
    uint32_t th_id = 0;
    uint64_t i = 0;
    uint64_t j = 0;
    uint64_t k = 0;
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

        //==============================================================
        // Initialize the variables with a new REFWORD
        uint32_t ref_int1, ref_int2, ref_int3;
        ref_word(&ref_int1, &ref_int2, &ref_int3);

        start_iteration();
        //======================================================================P
        // Parallel region
        #pragma offload target(mic) inout(log)
        {
            #pragma omp parallel for private(th_id, j, k) firstprivate(ref_int1, ref_int2, ref_int3) reduction(+:errors)
            for(th_id = 0; th_id < MIC_THREADS; th_id++)
            {
                asm volatile ("nop");
                asm volatile ("nop");


                // Portion of memory with 512 bits
                uint32_t value_int;
                __declspec(aligned(64)) uint32_t vec_int[ITEMS_INT];


                //==============================================================
                // AND
                if (th_id % 3 == 0) {
                    for(j = (repetitions == 0); j < BUSY; j++) {
                        LOOP_AND
                        LOOP_AND
                        LOOP_AND
                        LOOP_AND
                        LOOP_AND
                        LOOP_AND
                        LOOP_AND
                        LOOP_AND
                    }
                }


                //==============================================================
                // ADD
                else if (th_id % 3 == 1) {
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

                //==============================================================
                // MUL
                else if (th_id % 3 == 2) {
                    for(j = (repetitions == 0); j < BUSY; j++) {
                        LOOP_MUL
                        LOOP_MUL
                        LOOP_MUL
                        LOOP_MUL
                        LOOP_MUL
                        LOOP_MUL
                        LOOP_MUL
                        LOOP_MUL
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
