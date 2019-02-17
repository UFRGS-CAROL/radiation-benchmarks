
/***************************************************************************
//                  MicroBenchmmark [ ADD ] Developed for radiation benchmarks. 
//                        Gabriel Piscoya Dávila - 00246031
//                               January 2019
***************************************************************************/
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//*****************************************  LOG  *************************//
#ifdef LOGS
#include "log_helper.h"
#endif
//************************************************************************//

int main(int argc, char **argv) {


	if(argc!= 3){
		printf("Dude, please use the program correctly !.\n");
		printf("1. Total ammount of SUMs to be performed (Inner loop)\n");
		printf("2. Number of repetitions (Outer loop)\n");	
		return 0;
	}
	//printf("%s\n",argv[1]);
	//printf("%s\n",argv[2]);
	unsigned long int sums = strtoul(argv[1],NULL,0);  //1000000000;
	unsigned long int rep  = strtoul(argv[2],NULL,0);  //1000000000;
	unsigned long int ins = 10;				// Quantity of sums inside inner loop	
#ifdef INT			
	unsigned long int gold = 0;		
#elif FLOAT
    float gold =2.3437499476131052e-04 ;    						// This number considers rounding error. 
#endif
	unsigned long int i = 0;				// Loop iteration variable
	unsigned long int j = 0;				// Loop iteration variable	
#ifdef INT	
	 unsigned int re = 0;				// ACC var
	printf("Gold is: %lu\n",gold);
#elif FLOAT
    float re = 0.0;
	printf("Gold is: %1.16e\n",gold);
#endif
	int error = 0 ;
	int ref_int2 = 1525;
	float ref_float2 = 0.1;
#ifdef LOGS
    set_iter_interval_print(10);
    char test_info[300];
    snprintf(test_info, 300,"%lu,%lu,%lu",rep,sums,ins);
#ifdef INT    
    start_log_file("MicroBenchmark_ADD_INT_CPU", test_info);
#elif FLOAT
    start_log_file("MicroBenchmark_ADD_FLOAT_CPU", test_info);
#endif    
#endif


for(i=0;i<rep;i++){

#ifdef LOGS
	start_iteration();
#endif

#pragma omp parallel for reduction(+:re) private(i,j)
	for(j=0;j<sums;j++){
#ifdef INT    
// This Code works with unsigned long int variable (64 bits). Please modify "re" variable to unsigned long int
// "addq" is used here !
/*      
        re ++;re ++;re ++;re ++;re ++;re ++;re ++;re ++;re ++;re ++;
*/
// This code works with int variables (32 bits). Please modify "re" variable to int.
        asm volatile("addl %1, %0;"
                     "addl %1, %0;"
                     "addl %1, %0;"
                     "addl %1, %0;"
                     "addl %1, %0;"
                     "subl %1, %0;"
                     "subl %1, %0;"
                     "subl %1, %0;"
                     "subl %1, %0;"
                     "subl %1, %0;" : "+r" (re) : "r" (ref_int2));        
#elif FLOAT  
    // Tried to make assembly code for FPU, but all efforts were in vain.
    // Reference: https://cs.fit.edu/~mmahoney/cse3101/float.html
    // TODO At this moment rounding error needs to be fixed !! 
         re = re + 1525.74;
         re = re + 1525.74;
         re = re + 1525.74;
         re = re + 1525.74;
         re = re + 1525.74;
         re = re + (-1525.74);
         re = re + (-1525.74);
         re = re + (-1525.74);
         re = re + (-1525.74);
         re = re + (-1525.74);
         printf("%1.16e \n",re);
#endif

	}
#ifdef LOGS
	end_iteration();
#endif
	error = 0;
#ifdef INT
	if(re != gold){
		error = 1;
#ifdef LOGS
		char error_detail[200];
		sprintf(error_detail,"i=%lu,j=%lu, E=%lu ,R=%lu",i,j,gold,re);
		log_error_detail(error_detail);
#endif				    
	}
#elif FLOAT
    //double delta = fabs(gold-re)/gold;
	if( re - gold >= 1e-8 ){
		error = 1;
#ifdef LOGS
		char error_detail[200];
		sprintf(error_detail,"i=%lu,j=%lu, E=%1.16e ,R=%1.16e",i,j,gold,re);
		log_error_detail(error_detail);
#endif				    
	}
#endif
	else
		printf(".");
		fflush(stdout);
#ifdef LOGS
    log_error_count(error); 	// Always just one error.
#endif
	re = 0;
}

}
