
#include <string.h>
#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>

#include <string.h>

int main(int argc, char **argv) {
	int mult = 1;
	unsigned long long sys_mem = 671088640*mult;	// Tamanho da memoria dividido 4
	int* vetor;
	unsigned long long size = sizeof(int)* sys_mem * mult;
	int* gold;
	unsigned long long cont = 0;
for(cont = 0;cont< 100;cont ++){
	printf("********************************************************************\n");
	gold = (int*)malloc(sizeof(int));
	printf("Vetor de:%llu bytes \n",size);
	vetor = (int*)malloc(size);
	int i = 0;
	int k = 0;
	unsigned long long contador = 0;
	gold[0] = -1;
	printf("Gold:%d\n",gold[0]);
	memset(vetor,0xFF,size);

	for(k=0;k<50;k++){			
	#pragma omp parallel for reduction(+:contador) private(i)
		for(i = 0 ; i< sys_mem * mult; i++ ){
			//printf("Vetor[%d]:%d\n",i,vetor[i]);
			if(vetor[i] != gold[0] ){
				printf("It:%d, Pos[%d] :Sou um erro de Memoria E= %d, R= %d \n",k,i,gold[0],vetor[i]);
				vetor[i] = gold[0]; // Colocamos o valor certo na pos de memoria
				contador++; // Conta a ocorrencia de erros por iteracao			
			}
		}	
		printf("Contador -1: %llu\n",contador);	
	}
printf("***************Acabei com o 1111 *******************\n");
	gold[0] = 0;
	
	printf("Gold:%d\n",gold[0]);
	fflush(stdout);
	memset(vetor,0x00,size);
	for(k=0;k<50;k++){	
	#pragma omp parallel for reduction(+:contador) private(i)
		for(i = 0 ; i< sys_mem* mult; i++ ){
				//printf("Vetor[%d]:%d\n",i,vetor[i]);
			if(vetor[i] != gold[0] ){
				printf("%d: Sou um erro de Memoria E= %d, R= %d \n",i,gold[0],vetor[i]);
				vetor[i] = gold[0];
				contador++;			
			}
		}
		printf("Contador 0:%llu\n",contador);	
	}	
	free(vetor);
	
	}
	
}
