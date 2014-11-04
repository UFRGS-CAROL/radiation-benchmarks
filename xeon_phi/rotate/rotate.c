#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

// Xeon Phi total cores = 57. 1 core probably runs de OS.
#define MIC_NUM_THREADS 1
#define ARRAY_SIZE 56000
#define MAX 32000
#define REFWORD 1 //  0x55555555

///=============================================================================
int main (int argc, char *argv[]) {
    uint32_t size=0;
    uint32_t repetitions=0;
    if(argc != 2) {
        printf("Please provide the number of repetitions.\n");
        exit(EXIT_FAILURE);
    }

    repetitions = atoi(argv[1]);
    printf("Repetitions:%"PRIu32"\n", repetitions);

    omp_set_num_threads(MIC_NUM_THREADS);
    printf("Threads:%"PRIu32"\n", MIC_NUM_THREADS);

    uint32_t i = 0;
    uint32_t j = 0;
    uint32_t error_count = 0;

    #pragma offload target(mic) reduction(+:error_count)
    {
        #pragma omp parallel for private(j)
        for(i = 0; i < MIC_NUM_THREADS; i++)
        {
            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");
            uint32_t value = REFWORD;
            for (j = 1; j <= repetitions; j++) {
                
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );
	    	asm ("roll %0" : "+r" (value) : "0" (value) );

		// DEBUG
		// printf("%"PRIu64" => %d\n", j, value);

               // DEBUG: Injecting one error
               // if(i == 1)
               // 	value = 1;

		if ( value != REFWORD) {
			// DEBUG
                	// printf("Error found!\n");
			error_count++;
			value = REFWORD;
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
