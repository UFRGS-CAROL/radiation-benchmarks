#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>     // uint32_t
#include <inttypes.h>   // %"PRIu32"
#include <unistd.h>     // Sleep
#include <time.h>       // Time
#include <omp.h>        // OpenMP

// Xeon Phi Configuration
#define MIC_NUM_CORES       (56)                      // Max. 56 Cores (+1 core runs de OS)
#define MIC_NUM_THREADS     (4*MIC_NUM_CORES)         // Max. 4 Threads per Core.

//======================================================================
#define LOOP_BLOCK(V) {\
                        if (count##V != ref_word) { \
                            /* Time stamp */ \
                            time_t     now = time(0); \
                            struct tm  tstruct = *localtime(&now); \
                            char       buffer[100]; \
                            strftime(buffer, sizeof(buffer), "#ERROR Y:%Y M:%m D:%d Time:%X ", &tstruct); \
                            printf("%s", buffer); \
                            /* Error log */ \
                            error_count++; \
                            printf("IT:%"PRIu64" POS:%d THREAD:%d, REF:0x%08x FOUND:0x%08x\n", i, 0, th_id, ref_word, count##V); \
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
        printf("Please provide the number of <repetitions>.\n");
        exit(EXIT_FAILURE);
    }

    repetitions = string_to_uint64(argv[1]);
    if (repetitions == 0)       repetitions -= 1;   // MAX UINT64_T = 18446744073709551615
    omp_set_num_threads(MIC_NUM_THREADS);

    printf("#HEADER Repetitions:%"PRIu64" ",        repetitions);
    printf("Threads:%"PRIu32"\n",           MIC_NUM_THREADS);

    //==================================================================
    // Time stamp
    {
        time_t     now = time(0);
        struct tm  tstruct = *localtime(&now);
        char       buffer[100];
        strftime(buffer, sizeof(buffer), "#BEGIN Y:%Y M:%m D:%d Time:%X\n", &tstruct);
        printf("%s", buffer);
    }

    //==================================================================
    // Benchmark variables
    uint32_t th_id = 0;
    uint64_t i = 0;
    uint32_t error_count = 0;

    #pragma offload target(mic)
    {
        #pragma omp parallel for private(th_id, i) reduction(+:error_count)
        for(th_id = 0; th_id < MIC_NUM_THREADS; th_id++)
        {
            asm volatile ("nop");
            asm volatile ("nop");

            uint32_t ref_word = 0;

            uint32_t count0  = 0 ;
            uint32_t count1  = 0 ;
            uint32_t count2  = 0 ;
            uint32_t count3  = 0 ;
            uint32_t count4  = 0 ;
            uint32_t count5  = 0 ;
            uint32_t count6  = 0 ;
            uint32_t count7  = 0 ;

            for (i = 1; i <= repetitions; i++) {

                //==============================================================
                // Initialize the variables with a new REFWORD
                if ((i % 3) == 0)
                    asm volatile("movl $0x0, %0" : "=r" (ref_word));
                else if ((i % 3) == 1)
                    asm volatile("movl $0xFFFFFFFF, %0" : "=r" (ref_word));
                else
                    asm volatile("movl $0x55555555, %0" : "=r" (ref_word));

                asm volatile("mov %1, %0" : "=r" (count0) : "r" (ref_word) : );
                asm volatile("mov %1, %0" : "=r" (count1) : "r" (ref_word) : );
                asm volatile("mov %1, %0" : "=r" (count2) : "r" (ref_word) : );
                asm volatile("mov %1, %0" : "=r" (count3) : "r" (ref_word) : );
                asm volatile("mov %1, %0" : "=r" (count4) : "r" (ref_word) : );
                asm volatile("mov %1, %0" : "=r" (count5) : "r" (ref_word) : );
                asm volatile("mov %1, %0" : "=r" (count6) : "r" (ref_word) : );
                asm volatile("mov %1, %0" : "=r" (count7) : "r" (ref_word) : );

                //==============================================================
                // Real work
                sleep(1);

                //==========================================================
                // DEBUG: injecting one error (Bit-wise not RefWord)
                //if(th_id == 0 && i == 0)
                    //count0 = ~ref_word; // Bit-wise not

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
    }

    //==================================================================
    // Time stamp
    {
        time_t     now = time(0);
        struct tm  tstruct = *localtime(&now);
        char       buffer[100];
        strftime(buffer, sizeof(buffer), "#FINAL Y:%Y M:%m D:%d Time:%X ", &tstruct);
        printf("%s", buffer);
        printf("TotalErrors:%"PRIu32"\n", error_count);
    }
    exit(EXIT_SUCCESS);
}
