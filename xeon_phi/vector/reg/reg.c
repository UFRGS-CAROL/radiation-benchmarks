#include "../../refword.h"
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

#define ELEMENTS 16     // 64 bytes (512bits) ZMM register / element size

#define LOOP_BLOCK(V) {\
                            if (a##V[i] == ref2) {  \
                                error_count++;  \
                                printf("%d it, %d pos, %d thread, 0x%08x syndrome\n", i, 0, th_id, a##V[i]);    \
                                a##V[i] = (uint32_t)ref1;   \
                                b##V[i] = (uint32_t)ref2;   \
                                c##V[i] = (uint32_t)ref3;   \
                                d##V[i] = (uint32_t)ref_word;   \
                            }   \
                    }

extern double elapsedTime (void);

//======================================================================
int main(int argc, char *argv[]) {

    uint64_t repetitions = 0;
    uint32_t ref_word = 0;

    if(argc != 3) {
        printf("Please provide the number of <repetitions> and <refword option>.\n");
        print_refword();
        exit(EXIT_FAILURE);
    }

    repetitions = string_to_uint64(argv[1]);
    ref_word = get_refword(atoi(argv[2]));

    printf("Repetitions:%"PRIu64"\n",           repetitions);
    printf("Ref Word:0x%08x\n",                 ref_word);

    omp_set_num_threads(MIC_NUM_THREADS);
    printf("Threads:%"PRIu32"\n",               MIC_NUM_THREADS);

    //==================================================================
    // Benchmark variables
    double startTime,  duration;
    uint32_t th_id = 0;
    uint64_t i = 0;
    uint32_t error_count = 0;
    uint32_t ref1;
    uint32_t ref2;
    uint32_t ref3;

    // This will avoid simplification by the compiler
    asm volatile("movl %1, %0" : "=r" (ref1) : "r" (ref_word));
    asm volatile("movl %1, %0" : "=r" (ref2) : "r" (ref_word));
    asm volatile("movl %1, %0" : "=r" (ref3) : "r" (ref_word));

    #pragma offload target(mic)
    {
        #pragma omp parallel for private(th_id, i) reduction(+:error_count)
        for(th_id = 0; th_id < MIC_NUM_THREADS; th_id++)
        {
            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");

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


            for (i=0; i<ELEMENTS; i++) {
                a1[i] = a2[i] = a3[i] = a4[i] = a5[i] = a6[i] = a7[i] = a8[i] = (uint32_t)ref1;
                b1[i] = b2[i] = b3[i] = b4[i] = b5[i] = b6[i] = b7[i] = b8[i] = (uint32_t)ref2;
                c1[i] = c2[i] = c3[i] = c4[i] = c5[i] = c6[i] = c7[i] = c8[i] = (uint32_t)ref3;
                d1[i] = d2[i] = d3[i] = d4[i] = d5[i] = d6[i] = d7[i] = d8[i] = (uint32_t)ref_word;
            }

            #pragma vector aligned (a1,a2,a3,a4,a5,a6,a7,a8, b1,b2,b3,b4,b5,b6,b7,b8, c1,c2,c3,c4,c5,c6,c7,c8, d1,d2,d3,d4,d5,d6,d7,d8)
            for(i = 1; i < repetitions; i++) {

                a1[0:ELEMENTS] &= b1[0:ELEMENTS] == c1[0:ELEMENTS] == d1[0:ELEMENTS];
                a2[0:ELEMENTS] &= b2[0:ELEMENTS] == c2[0:ELEMENTS] == d2[0:ELEMENTS];
                a3[0:ELEMENTS] &= b3[0:ELEMENTS] == c3[0:ELEMENTS] == d3[0:ELEMENTS];
                a4[0:ELEMENTS] &= b4[0:ELEMENTS] == c4[0:ELEMENTS] == d4[0:ELEMENTS];
                a5[0:ELEMENTS] &= b5[0:ELEMENTS] == c5[0:ELEMENTS] == d5[0:ELEMENTS];
                a6[0:ELEMENTS] &= b6[0:ELEMENTS] == c6[0:ELEMENTS] == d6[0:ELEMENTS];
                a7[0:ELEMENTS] &= b7[0:ELEMENTS] == c7[0:ELEMENTS] == d7[0:ELEMENTS];
                a8[0:ELEMENTS] &= b8[0:ELEMENTS] == c8[0:ELEMENTS] == d8[0:ELEMENTS];


                for(i = 0; i<ELEMENTS; i++) {
                    LOOP_BLOCK(1)
                    LOOP_BLOCK(2)
                    LOOP_BLOCK(3)
                    LOOP_BLOCK(4)

                    LOOP_BLOCK(5)
                    LOOP_BLOCK(6)
                    LOOP_BLOCK(7)
                    LOOP_BLOCK(8)
                }
            }

            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");
        }
    }

    printf("Errors: %"PRIu32"\n", error_count);
    exit(EXIT_SUCCESS);
}
