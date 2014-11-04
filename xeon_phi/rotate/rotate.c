#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

// Xeon Phi total cores = 57. 1 core probably runs de OS.
#define MIC_NUM_CORES 1
#define ARRAY_SIZE 56000
#define MAX 32000
#define refword 1 //  0x55555555

//#define rotate(inout) ({ asm ("rol #1,%0" : "=d" (inout)); })


///=============================================================================
int main (int argc, char *argv[]) {
    uint64_t size=0;
    uint64_t repetitions=0;
    if(argc != 2) {
        printf("Please provide the number of repetitions.\n");
        exit(EXIT_FAILURE);
    }

    repetitions = atoi(argv[1]);

    printf("Repetitions:%"PRIu64"\n", repetitions);

 
    uint32_t i = 0;
    uint32_t error_count = 0;

    #pragma offload target(mic) reduction(+:error_count)
    {
        #pragma omp parallel for
        for(i = 0; i < MIC_NUM_CORES; i++)
        {
            asm volatile ("nop");
            asm volatile ("nop");
            asm volatile ("nop");
            uint32_t value=refword;
            uint32_t j;
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

                printf("%"PRIu64" => %d\n", j, value);

               // Injecting one error
               // if(i == 1)
               // 	value = 1;

		if ( value != refword) {
                	// printf("Error found!\n");
			error_count++;
			value = refword;
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
