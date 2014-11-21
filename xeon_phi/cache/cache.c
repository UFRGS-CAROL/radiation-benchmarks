#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>     // uint32_t
#include <inttypes.h>   // %"PRIu32"
#include <unistd.h>     // Sleep
#include <time.h>       // Time
#include <omp.h>        // OpenMP

// Xeon Phi Configuration
#define MIC_NUM_CORES       (56)                       // Max. 56 Cores (+1 core runs de OS)
#define MIC_NUM_THREADS     (1*MIC_NUM_CORES)         // Max. 4 Threads per Core.

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
        printf("Please provide the number of <repetitions> and <array size>.\n");
        exit(EXIT_FAILURE);
    }

    repetitions = string_to_uint64(argv[1]);
    size = atoi(argv[2]);

    if (size % (MIC_NUM_THREADS * sizeof(uint32_t)) != 0) {
        printf("The array size needs divisible by %ld (#threads * element size).\n", MIC_NUM_THREADS * sizeof(uint32_t));
        exit(EXIT_FAILURE);
    }

    if (repetitions == 0)       repetitions -= 1;   // MAX UINT64_T = 18446744073709551615
    omp_set_num_threads(MIC_NUM_THREADS);

    printf("#HEADER Elem.Size:%"PRIu32"B ", (uint32_t)sizeof(uint32_t));
    printf("Elements:%"PRIu32" ",           (uint32_t)(size / sizeof(uint32_t)));
    printf("ArraySize:%"PRIu32"KB ",        (uint32_t)(size / 1024));
    printf("SizePerThread:%"PRIu32"KB ",    (uint32_t)(size / 1024) / MIC_NUM_THREADS);
    printf("Repetitions:%"PRIu64" ",        repetitions);
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
    uint32_t jump = 0;
    uint32_t slice = (size / sizeof(uint32_t)) / MIC_NUM_THREADS ;
    uint32_t error_count = 0;

    uint32_t *ptr_vector;
    ptr_vector = (uint32_t *)valloc(size);

    //==================================================================
    // Start the parallel region
    #pragma offload target(mic) in(ptr_vector:length(size / sizeof(uint32_t)))
    {
        #pragma omp parallel for private(th_id, i, jump) reduction(+:error_count)
        for(th_id = 0; th_id < MIC_NUM_THREADS; th_id++)
        {
            asm volatile ("nop");
            asm volatile ("nop");

            uint32_t ref_word = 0;

            for (i = 0; i < repetitions; i++) {

                //==============================================================
                // Initialize the vector with a new REFWORD
                if ((i % 3) == 0)
                    asm volatile("movl $0x0, %0" : "=r" (ref_word));
                else if ((i % 3) == 1)
                    asm volatile("movl $0xFFFFFFFF, %0" : "=r" (ref_word));
                else
                    asm volatile("movl $0x55555555, %0" : "=r" (ref_word));

                for (jump = slice * th_id; jump < slice * (th_id + 1); jump++) {
                    ptr_vector[jump] = ref_word;
                }

                //==============================================================
                // Real work
                sleep(1);

                for (jump = slice * th_id; jump < slice * (th_id + 1); jump++) {

                    //==========================================================
                    // DEBUG: injecting one error (Bit-wise not RefWord)
                    //if(th_id == 0 && i == 0 && jump == 0)
                        //ptr_vector[jump] = ~ref_word;

                    if (ptr_vector[jump] != ref_word) {
                        //======================================================
                        // Time stamp
                        time_t     now = time(0);
                        struct tm  tstruct = *localtime(&now);
                        char       buffer[100];
                        strftime(buffer, sizeof(buffer), "#ERROR Y:%Y M:%m D:%d Time:%X ", &tstruct);
                        printf("%s", buffer);

                        //======================================================
                        // Error log
                        error_count++;
                        printf("IT:%"PRIu64" POS:%d THREAD:%d, REF:0x%08x FOUND:0x%08x\n", i, jump, th_id, ref_word, ptr_vector[jump]);
                    }

                }

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
