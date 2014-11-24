#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>     // uint32_t
#include <inttypes.h>   // %"PRIu32"
#include <unistd.h>     // Sleep
#include <time.h>       // Time
#include <omp.h>        // OpenMP
#include <math.h>       // pow

// Xeon Phi Configuration
#define MIC_CORES       (56)            // Max. 56 Cores (+1 core runs de OS)
#define MIC_THREADS     (4*MIC_CORES)   // Max. 4 Threads per Core.
#define MAX_ERROR       32              // Max. number of errors per repetition
#define LOG_SIZE        128             // Line size per error
#define BUSY            300000          // Repetitions in the busy wait

// ~ #define DEBUG           if (i==0 && j==0 && errors==0) asm volatile("movl %1, %0" : "=r" (value) : "r" (~value));
#define DEBUG /*OFF*/

//======================================================================
#define LOOP_AND {\
        value = refw1; \
        asm volatile("and $0xFFFFFFFF, %0" : "+r" (value));\
        asm volatile("and $0xFFFFFFFF, %0" : "+r" (value));\
        asm volatile("and $0xFFFFFFFF, %0" : "+r" (value));\
        asm volatile("and $0xFFFFFFFF, %0" : "+r" (value));\
        asm volatile("and $0xFFFFFFFF, %0" : "+r" (value));\
        asm volatile("and $0xFFFFFFFF, %0" : "+r" (value));\
        asm volatile("and $0xFFFFFFFF, %0" : "+r" (value));\
        asm volatile("and $0xFFFFFFFF, %0" : "+r" (value));\
        \
        asm volatile("and $0xFFFFFFFF, %0" : "+r" (value));\
        asm volatile("and $0xFFFFFFFF, %0" : "+r" (value));\
        asm volatile("and $0xFFFFFFFF, %0" : "+r" (value));\
        asm volatile("and $0xFFFFFFFF, %0" : "+r" (value));\
        asm volatile("and $0xFFFFFFFF, %0" : "+r" (value));\
        asm volatile("and $0xFFFFFFFF, %0" : "+r" (value));\
        asm volatile("and $0xFFFFFFFF, %0" : "+r" (value));\
        asm volatile("and $0xFFFFFFFF, %0" : "+r" (value));\
        \
        DEBUG \
        if (value != refw1) \
            snprintf(log[th_id][errors++], LOG_SIZE, "%s IT:%"PRIu64" POS:%d TH:%d REF:0x%08x WAS:0x%08x\n", time, i, j, th_id, refw1, value); \
                }

#define LOOP_ADD {\
        value = 0; \
        asm volatile("addl %1, %0" : "+r" (value) : "r" (refw2));\
        asm volatile("addl %1, %0" : "+r" (value) : "r" (refw2));\
        asm volatile("addl %1, %0" : "+r" (value) : "r" (refw2));\
        asm volatile("addl %1, %0" : "+r" (value) : "r" (refw2));\
        asm volatile("addl %1, %0" : "+r" (value) : "r" (refw2));\
        asm volatile("addl %1, %0" : "+r" (value) : "r" (refw2));\
        asm volatile("addl %1, %0" : "+r" (value) : "r" (refw2));\
        asm volatile("addl %1, %0" : "+r" (value) : "r" (refw2));\
        \
        asm volatile("addl %1, %0" : "+r" (value) : "r" (refw2));\
        asm volatile("addl %1, %0" : "+r" (value) : "r" (refw2));\
        asm volatile("addl %1, %0" : "+r" (value) : "r" (refw2));\
        asm volatile("addl %1, %0" : "+r" (value) : "r" (refw2));\
        asm volatile("addl %1, %0" : "+r" (value) : "r" (refw2));\
        asm volatile("addl %1, %0" : "+r" (value) : "r" (refw2));\
        asm volatile("addl %1, %0" : "+r" (value) : "r" (refw2));\
        asm volatile("addl %1, %0" : "+r" (value) : "r" (refw2));\
        \
        DEBUG \
        if (value != (refw2 << 4)) \
            snprintf(log[th_id][errors++], LOG_SIZE, "%s IT:%"PRIu64" POS:%d TH:%d REF:0x%08x WAS:0x%08x\n", time, i, j, th_id, (refw2 << 4), value); \
                }

#define LOOP_MUL {\
        value = refw3; \
        asm volatile("imul $0x2, %0" : "+r" (value));\
        asm volatile("imul $0x2, %0" : "+r" (value));\
        asm volatile("imul $0x2, %0" : "+r" (value));\
        asm volatile("imul $0x2, %0" : "+r" (value));\
        asm volatile("imul $0x2, %0" : "+r" (value));\
        asm volatile("imul $0x2, %0" : "+r" (value));\
        asm volatile("imul $0x2, %0" : "+r" (value));\
        asm volatile("imul $0x2, %0" : "+r" (value));\
        \
        DEBUG \
        if (value != (refw3 << 8)) \
            snprintf(log[th_id][errors++], LOG_SIZE, "%s IT:%"PRIu64" POS:%d TH:%d REF:0x%08x WAS:0x%08x\n", time, i, j, th_id, (refw3 << 8), value); \
                }

#define LOOP_DIV {\
        value = refw1; \
        /* Copy the operands to perform the division */ \
        asm volatile("movl $0x0, %%edx" : : : ); \
        asm volatile("movl %0, %%eax" : : "r" (value) : "eax"); \
        asm volatile("movl $0x2, %%ebx" : : : ); \
        /* Perform the division */ \
        asm volatile("divl %%ebx" : : : "eax", "edx", "ebx"); \
        asm volatile("movl $0x0, %%edx" : : : ); \
        asm volatile("divl %%ebx" : : : "eax", "edx", "ebx"); \
        asm volatile("movl $0x0, %%edx" : : : ); \
        asm volatile("divl %%ebx" : : : "eax", "edx", "ebx"); \
        asm volatile("movl $0x0, %%edx" : : : ); \
        asm volatile("divl %%ebx" : : : "eax", "edx", "ebx"); \
        /* Copy back the operands to check the division */ \
        asm volatile("movl %%eax, %0" : "=r" (value) : : "eax"); \
        \
        DEBUG \
        if (value != (refw1 >> 4)) \
            snprintf(log[th_id][errors++], LOG_SIZE, "%s IT:%"PRIu64" POS:%d TH:%d REF:0x%08x WAS:0x%08x\n", time, i, j, th_id, (refw1 >> 4), value); \
                }

//======================================================================
// Linear Feedback Shift Register using 32 bits and XNOR. Details at:
// http://www.xilinx.com/support/documentation/application_notes/xapp052.pdf
// http://www.ece.cmu.edu/~koopman/lfsr/index.html
void refword(uint32_t *refw1, uint32_t *refw2, uint32_t *refw3){
    static uint32_t counter = 0;

    counter++;
    if (counter == 1){
        *refw1  = 0xCCCCCCCC;   // 1100 1100 1100 1100 | 1100 1100 1100 1100 (3435973836)
        *refw2  = 0x0CCCCCCC;   // 0000 1100 1100 1100 | 1100 1100 1100 1100
        *refw3  = 0x00CCCCCC;   // 0000 0000 1100 1100 | 1100 1100 1100 1100
        return;
    }
    else if (counter == 2){
        *refw1  = 0x66666666;   // 0110 0110 0110 0110 | 0110 0110 0110 0110
        *refw2  = 0x06666666;   // 0000 0110 0110 0110 | 0110 0110 0110 0110
        *refw3  = 0x06666666;   // 0000 0000 0110 0110 | 0110 0110 0110 0110
        return;
    }
    else if (counter == 3){
        *refw1  = 0x33333333;   // 0011 0011 0011 0011 | 0011 0011 0011 0011
        *refw2  = 0x03333333;   // 0000 0011 0011 0011 | 0011 0011 0011 0011
        *refw3  = 0x00333333;   // 0000 0000 0011 0011 | 0011 0011 0011 0011
        return;
    }
    else if (counter == 4){
        *refw1  = 0xAAAAAAAA;   // 1010 1010 1010 1010 | 1010 1010 1010 1010
        *refw2  = 0x0AAAAAAA;   // 0000 1010 1010 1010 | 1010 1010 1010 1010
        *refw3  = 0x00AAAAAA;   // 0000 0000 1010 1010 | 1010 1010 1010 1010
        return;
    }
    else if (counter == 5){
        *refw1  = 0x55555555;   // 0101 0101 0101 0101 | 0101 0101 0101 0101
        *refw2  = 0x05555555;   // 0000 0101 0101 0101 | 0101 0101 0101 0101
        *refw3  = 0x00555555;   // 0000 0000 0101 0101 | 0101 0101 0101 0101
        return;
    }
    else if (counter == 6) {
        *refw1  = 0x99999999;   // 1001 1001 1001 1001 | 1001 1001 1001 1001
        *refw2  = 0x09999999;   // 0000 1001 1001 1001 | 1001 1001 1001 1001
        *refw3  = 0x00999999;   // 0000 0000 1001 1001 | 1001 1001 1001 1001
        return;
    }
    else if (counter == 7){
        *refw1  = 0x88888888;   // 1000 1000 1000 1000 | 1000 1000 1000 1000
        *refw2  = 0x08888888;   // 0000 1000 1000 1000 | 1000 1000 1000 1000
        *refw3  = 0x00888888;   // 0000 0000 1000 1000 | 1000 1000 1000 1000
        return;
    }
    else if (counter == 8){
        *refw1  = 0x44444444;   // 0100 0100 0100 0100 | 0100 0100 0100 0100
        *refw2  = 0x04444444;   // 0000 0100 0100 0100 | 0100 0100 0100 0100
        *refw3  = 0x00444444;   // 0000 0000 0100 0100 | 0100 0100 0100 0100
        return;
    }
    else if (counter == 9){
        *refw1  = 0x22222222;   // 0010 0010 0010 0010 | 0010 0010 0010 0010
        *refw2  = 0x02222222;   // 0000 0010 0010 0010 | 0010 0010 0010 0010
        *refw3  = 0x00222222;   // 0000 0000 0010 0010 | 0010 0010 0010 0010
        return;
    }
    else {
        *refw1  = 0x11111111;  // 0001 0001 0001 0001 | 0001 0001 0001 0001
        *refw2  = 0x01111111;  // 0000 0001 0001 0001 | 0001 0001 0001 0001
        *refw3  = 0x00111111;  // 0000 0000 0001 0001 | 0001 0001 0001 0001
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
    uint64_t j = 0;
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

        //==============================================================
        // Initialize the variables with a new REFWORD
        uint32_t refw1, refw2, refw3;
        refword(&refw1, &refw2, &refw3);

        //======================================================================P
        // Parallel region
        #pragma offload target(mic) inout(log)
        {
            #pragma omp parallel for private(th_id, j) firstprivate(refw1, refw2, refw3) reduction(+:errors)
            for(th_id = 0; th_id < MIC_THREADS; th_id++)
            {
                asm volatile ("nop");
                asm volatile ("nop");
                asm volatile ("nop");

                uint32_t value = 0;
                //==============================================================
                // AND
                if (th_id % 4 == 0) {
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
                else if (th_id % 4 == 1) {
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
                else if (th_id % 4 == 2) {
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

                //==============================================================
                // DIV
                else if (th_id % 4 == 3) {
                    for(j = (repetitions == 0); j < BUSY; j++) {
                        LOOP_DIV
                        LOOP_DIV
                        LOOP_DIV
                        LOOP_DIV
                        LOOP_DIV
                        LOOP_DIV
                        LOOP_DIV
                        LOOP_DIV
                    }
                }

            }
            asm volatile ("nop");
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
