
/***************************************************************************
//                  MicroBenchmmark [ ADD ] Developed for radiation benchmarks. 
//                        Gabriel Piscoya Dávila - 00246031
//                               January 2019
***************************************************************************/
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
	printf("%s\n",argv[1]);
	printf("%s\n",argv[2]);
	unsigned long int sums = strtoul(argv[1],NULL,0);  //1000000000;
	unsigned long int rep  = strtoul(argv[2],NULL,0);  //1000000000;
	unsigned long int ins = 10;				// Quantity of sums inside inner loop				
	unsigned long int gold = ins * sums; 		
	unsigned long int i = 0;				// Loop iteration variable
	unsigned long int j = 0;				// Loop iteration variable	
	unsigned long int re = 0;				// ACC var
	int error = 0 ;
	printf("Gold is: %lu\n",gold);

#ifdef LOGS
    set_iter_interval_print(10);
    char test_info[300];
    snprintf(test_info, 300,"%lu,%lu,%lu",rep,sums,ins);
    start_log_file("MicroBenchmark_ADD", test_info);
#endif


for(i=0;i<rep;i++){

#ifdef LOGS
	start_iteration();
#endif

#pragma omp parallel for reduction(+:re) private(i)
	for(j=0;j<sums;j++){
		re ++;
		re ++;
		re ++;
		re ++;
		re ++;
		re ++;
		re ++;
		re ++;
		re ++;
		re ++;
	}
#ifdef LOGS
	end_iteration();
#endif
	error = 0;
	if(re != gold){
		error = 1;
#ifdef LOGS
		char error_detail[200];
		sprintf(error_detail,"i=%lu,j=%lu, E=%lu ,R=%lu",i,j,gold,re);
		log_error_detail(error_detail);
#endif				    
	}
	else
		printf(".");
#ifdef LOGS
    log_error_count(error); 	// Always just one error.
#endif
	re = 0;
}

// Aqui podemos adicionar as coisas para divisão, multiplicação, shift
}
