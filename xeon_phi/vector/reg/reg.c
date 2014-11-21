#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>     // uint32_t
#include <inttypes.h>   // %"PRIu32"
#include <unistd.h>     // Sleep
#include <time.h>       // Time
#include <omp.h>        // OpenMP

//https://software.intel.com/en-us/articles/using-intel-avx-without-writing-avx

// Xeon Phi Configuration
#define MIC_NUM_CORES       (56)                      // Max. 56 Cores (+1 core runs de OS)
#define MIC_NUM_THREADS     (4*MIC_NUM_CORES)         // Max. 4 Threads per Core.

#define ELEMENTS 16     // 64 bytes (512bits) ZMM register / element size

#define LOOP_BLOCK(X,Y) {\
                            for(j = 0; j < ELEMENTS; j++) { \
                                if (X##Y[j] != ref_word) { \
                                    /* Time stamp */ \
                                    time_t     now = time(0);\
                                    struct tm  tstruct = *localtime(&now); \
                                    char       buffer[100]; \
                                    strftime(buffer, sizeof(buffer), "#ERROR Y:%Y M:%m D:%d Time:%X ", &tstruct); \
                                    printf("%s", buffer); \
                                    /* Error log */ \
                                    error_count++; \
                                    printf("IT:%"PRIu64" POS:%d THREAD:%d, REF:0x%08x FOUND:0x%08x\n", i, 0, th_id, ref_word, X##Y[j]); \
                                } \
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
int main(int argc, char *argv[]) {

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
    double startTime,  duration;
    uint32_t th_id = 0;
    uint64_t i = 0;
    uint64_t j = 0;
    uint32_t error_count = 0;

    #pragma offload target(mic)
    {
        #pragma omp parallel for private(th_id, i, j) reduction(+:error_count)
        for(th_id = 0; th_id < MIC_NUM_THREADS; th_id++)
        {
            asm volatile ("nop");
            asm volatile ("nop");

            uint32_t ref_word = 0;

            // 32 register of 512 bits
            __declspec(aligned(64)) uint32_t a1[ELEMENTS],
                                             a2[ELEMENTS],
                                             a3[ELEMENTS],
                                             a4[ELEMENTS],
                                             a5[ELEMENTS],
                                             a6[ELEMENTS],
                                             a7[ELEMENTS],
                                             a8[ELEMENTS];

            __declspec(aligned(64)) uint32_t b1[ELEMENTS],
                                             b2[ELEMENTS],
                                             b3[ELEMENTS],
                                             b4[ELEMENTS],
                                             b5[ELEMENTS],
                                             b6[ELEMENTS],
                                             b7[ELEMENTS],
                                             b8[ELEMENTS];

            __declspec(aligned(64)) uint32_t c1[ELEMENTS],
                                             c2[ELEMENTS],
                                             c3[ELEMENTS],
                                             c4[ELEMENTS],
                                             c5[ELEMENTS],
                                             c6[ELEMENTS],
                                             c7[ELEMENTS],
                                             c8[ELEMENTS];

            __declspec(aligned(64)) uint32_t d1[ELEMENTS],
                                             d2[ELEMENTS],
                                             d3[ELEMENTS],
                                             d4[ELEMENTS],
                                             d5[ELEMENTS],
                                             d6[ELEMENTS],
                                             d7[ELEMENTS],
                                             d8[ELEMENTS];

            #pragma vector aligned (a1,a2,a3,a4,a5,a6,a7,a8, b1,b2,b3,b4,b5,b6,b7,b8, c1,c2,c3,c4,c5,c6,c7,c8, d1,d2,d3,d4,d5,d6,d7,d8)
            for(i = 1; i < repetitions; i++) {

                //==============================================================
                // Initialize the variables with a new REFWORD
                if ((i % 3) == 0)
                    asm volatile("movl $0x0, %0" : "=r" (ref_word));
                else if ((i % 3) == 1)
                    asm volatile("movl $0xFFFFFFFF, %0" : "=r" (ref_word));
                else
                    asm volatile("movl $0x55555555, %0" : "=r" (ref_word));

                for (j = 0; j < ELEMENTS; j++) {
                    a1[i] = a2[i] = a3[i] = a4[i] = a5[i] = a6[i] = a7[i] = a8[i] = (uint32_t)ref_word;
                    b1[i] = b2[i] = b3[i] = b4[i] = b5[i] = b6[i] = b7[i] = b8[i] = (uint32_t)ref_word;
                    c1[i] = c2[i] = c3[i] = c4[i] = c5[i] = c6[i] = c7[i] = c8[i] = (uint32_t)ref_word;
                    d1[i] = d2[i] = d3[i] = d4[i] = d5[i] = d6[i] = d7[i] = d8[i] = (uint32_t)ref_word;
                }

                //==============================================================
                // Real work
                sleep(1);

                //==========================================================
                // DEBUG: injecting one error (Bit-wise not RefWord)
                //if(th_id == 0 && i == 0)
                    //a1[0] = ~ref_word; // Bit-wise not

                LOOP_BLOCK(a,1)
                LOOP_BLOCK(a,2)
                LOOP_BLOCK(a,3)
                LOOP_BLOCK(a,4)
                LOOP_BLOCK(a,5)
                LOOP_BLOCK(a,6)
                LOOP_BLOCK(a,7)
                LOOP_BLOCK(a,8)

                LOOP_BLOCK(b,1)
                LOOP_BLOCK(b,2)
                LOOP_BLOCK(b,3)
                LOOP_BLOCK(b,4)
                LOOP_BLOCK(b,5)
                LOOP_BLOCK(b,6)
                LOOP_BLOCK(b,7)
                LOOP_BLOCK(b,8)

                LOOP_BLOCK(c,1)
                LOOP_BLOCK(c,2)
                LOOP_BLOCK(c,3)
                LOOP_BLOCK(c,4)
                LOOP_BLOCK(c,5)
                LOOP_BLOCK(c,6)
                LOOP_BLOCK(c,7)
                LOOP_BLOCK(c,8)

                LOOP_BLOCK(d,1)
                LOOP_BLOCK(d,2)
                LOOP_BLOCK(d,3)
                LOOP_BLOCK(d,4)
                LOOP_BLOCK(d,5)
                LOOP_BLOCK(d,6)
                LOOP_BLOCK(d,7)
                LOOP_BLOCK(d,8)
            }

            asm volatile ("nop");
            asm volatile ("nop");
        }
    }

    printf("Errors: %"PRIu32"\n", error_count);
    exit(EXIT_SUCCESS);
}
