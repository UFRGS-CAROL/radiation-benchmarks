
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
#include <stdint.h>  
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
	int ref_int2 = 3;
	float ref_float2 = 0.1;
	unsigned long int sums = strtoul(argv[1],NULL,0);  //1000000000;
	unsigned long int rep  = strtoul(argv[2],NULL,0);  //1000000000;
	unsigned long int ins = 10;				// Quantity of multiplications inside inner loop	
#ifdef INT			
	unsigned long int gold = pow(ref_int2, ins*sums) ; 		
#elif FLOAT
    float gold = 1.0;   
#endif
	unsigned long int i = 0;				// Loop iteration variable
	unsigned long int j = 0;				// Loop iteration variable	
#ifdef INT	
	  unsigned long int re = 1;				// ACC var
	printf("Gold is: %lu\n",gold);
#elif FLOAT
    float re = 1.0;
	printf("Gold is: %1.16e\n",gold);
#endif
	int error = 0 ;

#ifdef LOGS
    set_iter_interval_print(10);
    char test_info[300];
    snprintf(test_info, 300,"%lu,%lu,%lu",rep,sums,ins);
#ifdef INT    
    start_log_file("MicroBenchmark_MUL_INT_CPU", test_info);
#elif FLOAT
    start_log_file("MicroBenchmark_MUL_FLOAT_CPU", test_info);
#endif    
#endif



for(i=0;i<rep;i++){

#ifdef LOGS
	start_iteration();
#endif

//#pragma omp parallel for reduction(+:re) private(i)
	for(j=0;j<sums;j++){
#ifdef INT    
 
 
        /*asm volatile("imul $0x3, %0;"
                     "imul $0x3, %0;"
                     "imul $0x3, %0;"
                     "imul $0x3, %0;"
                     "imul $0x3, %0;"
                     "imul $0x3, %0;"
                     "imul $0x3, %0;"
                     "imul $0x3, %0;"
                     "imul $0x3, %0;"
                     "imul $0x3, %0;": "+r" (re) ); 
        */
       // asm volatile("movl $0x0, %%edx" : : : ); 
        asm volatile("movl %0, %%eax" : : "r"(re) : "eax"); 
        //asm volatile("movl $0x2, %%ebx" : : : ); 
        //asm volatile("divl %%ebx" : : : "eax", "edx", "ebx"); \                   
                       
                     printf("Re:%d\n",re);     
#elif FLOAT  
    // Reference: https://cs.fit.edu/~mmahoney/cse3101/float.html
                re = re *3;
                re = re *3;                
                re = re *3;                
                re = re *3;                
                re = re *3;                
                re = re *3;                
                re = re *3;                
                re = re *3;                
                re = re *3;                
                re = re *3;  
                re = re/59049; // To avoid Overflow               
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
    double delta = fabs(gold-re)/gold;
    printf("Re:%1.16e\n",re);
	if(delta >= 1e-8 ){
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
	re = 1;
}

}
