
/***************************************************************************
//                  Memory Test Developed for radiation benchmarks. 
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
	unsigned long long int MAX = 1000000000;
	int i = 0;
	unsigned long long int re = 0;
	unsigned long long int gold = 5000000000;

#pragma omp parallel for reduction(+:re) private(i)
	for(i=0;i<MAX;i++){

		re ++;
		re ++;
		re ++;
		re ++;
		re ++;
    __asm__ ( "movl $10, %eax;"
                "movl $20, %ebx;"
                "addl %ebx, %eax;" );

	}
	if(re != gold)
		printf("Deu ruim");	
	else
		printf("%llu\n",re);

// Aqui podemos adicionar as coisas para divisão, multiplicação, shift
}
