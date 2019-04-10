
/***************************************************************************
//                  Memory Test Developed for radiation benchmarks. 
//                        Gabriel Piscoya Dávila - 00246031
//                               January 2019
***************************************************************************/

#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
//*****************************************  LOG  *************************//
#ifdef LOGS
#include "log_helper.h"
#endif
//************************************************************************//

char time_now[200] = "";

// Debug params: -s 104857600 -e 2 -i 5 -w 2 
// Params ------------------------------------------------------------------
struct Params {

    unsigned long int          mem_size;
    unsigned long int          external_it;
    unsigned long int          internal_it;
    int         wait_time;
    int         verbose;    // Not implemented yet

    Params(int argc, char **argv) {
        // Default Values
        mem_size    =  3758096384; // 3,5 GB
        external_it =  100;
        internal_it =   50;
        verbose     =    0;
        wait_time   =    0;             
        int opt;
        while((opt = getopt(argc, argv, "hs:e:i:v:w:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;

            case 's': mem_size      = strtoul(optarg, NULL, 0); break;
            case 'e': external_it   = atoi(optarg); break;
            case 'i': internal_it   = atoi(optarg); break;
            case 'v': verbose       = atoi(optarg); break;
            case 'w': wait_time     = atoi(optarg); break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
    }
    void usage(){
        fprintf(stderr,
                "\nMemory_Test  Usage:  ./mem_test [options]"
                "\n"
                "\nGeneral options:"
                "\n -s <S> Memory Size in Bytes"
                "\n -e <E> Malloc (External) Iterations"
                "\n -i <I> Verification (Internal, same for 0x00 and 0xFF) Iterations"
                "\n -v <V> Verbose Mode"
                "\n -w <W> Wait Time between Write and Read in Seconds"
                "\n");
    }    
};

long long get_time() {
	struct timeval tv;

	gettimeofday(&tv, NULL);

	return (tv.tv_sec * 1000000) + tv.tv_usec;
}

void show_time(){
    time_t file_time;
    struct tm *ptm;
    char day[10], month[10], year[15], hour[10], second[10], minute[10];
    file_time = time(NULL);
    ptm = gmtime(&file_time);

    snprintf(day, sizeof(day), "%02d", ptm->tm_mday);
    snprintf(month, sizeof(month), "%02d", ptm->tm_mon + 1);
    snprintf(year, sizeof(year), "%04d", ptm->tm_year + 1900);
    snprintf(hour, sizeof(hour), "%02d", ptm->tm_hour);
    snprintf(minute, sizeof(minute), "%02d", ptm->tm_min);
    snprintf(second, sizeof(second), "%02d", ptm->tm_sec);

    	
	snprintf(time_now,sizeof(char)*200, "Y:%s M:%s D:%s Time:%s:%s:%s\n", year, month, day,
			hour, minute, second);

    printf("Y:%s M:%s D:%s Time:%s:%s:%s\n", year, month, day,hour, minute, second);    
    
}


int main(int argc, char **argv) {

    const Params p(argc, argv);
    printf("%lu, %lu, %lu ,%d ,%d\n",p.mem_size,p.external_it,p.internal_it,p.wait_time,p.verbose);
    
    long long init_time;
    double error_time = 0;
    
    init_time = get_time();
    
    show_time();
    
#ifdef LOGS
    disable_double_error_kill(); // Disable Double Kill Error 
    set_iter_interval_print(1);
    char test_info[300];
    snprintf(test_info, 300, "-s %lu, -e %lu, -i %lu, -w %d, -v %d\n",p.mem_size,p.external_it,p.internal_it,p.wait_time,p.verbose);
    start_log_file("memory_test", test_info);
#endif

	unsigned long int  sys_mem = p.mem_size/4;	// Tamanho da memoria dividido 4
	int* vetor;
	unsigned long int size = sizeof(int)* sys_mem;
	int* gold;
	unsigned long int  cont = 0;
    for(cont = 0;cont< p.external_it ;cont++){

#ifdef LOGS
        start_iteration();
#endif    
    
	    printf("********************************************************************\n");
	    gold = (int*)malloc(sizeof(int));
	    printf("Vetor de:%lu bytes \n",size);
	    vetor = (int*)malloc(size);

	    // TODO Verify if swap is used -> Exit and error message	    
	    //system("cat /proc/swaps");
	    
	    int i = 0;
	    int k = 0;
	    unsigned long int contador = 0;
	    gold[0] = -1;
	    printf("Gold:%d\n",gold[0]);
	    fflush(stdout);
	    memset(vetor,0xFF,size);

        sleep(p.wait_time); // Sleep Between Write/Read iteration for error accumulation in DDR
#ifdef LOGS
        end_iteration();
#endif

    	for(k = 0; k< p.internal_it; k++){			
        #pragma omp parallel for reduction(+:contador) private(i) 
            for(i = 0 ; i< sys_mem; i++ ){
                if(p.verbose == 1){
                    vetor[5]=12;                
                }

			    //printf("Vetor[%d]:%d\n",i,vetor[i]);
			    if(vetor[i] != gold[0] ){
			        error_time = (double) (get_time() - init_time) / 1000000;
			        printf("[%f] Gold: 1, It_Externa: %lu, It_Interna: %d, Pos[%d]:Sou um erro de Memoria E= %d, R= %d \n",error_time,cont, k,i,gold[0],vetor[i]);
#ifdef LOGS
		            char error_detail[200];
            		sprintf(error_detail,"[%f] Gold: 1, It_Externa: %lu, It_Interna: %d, Pos[%d]:Sou um erro de Memoria E= %d, R= %d \n",error_time,cont, k,i,gold[0],vetor[i]);
           			log_error_detail(error_detail);
#endif				    
				    vetor[i] = gold[0];         // Colocamos o valor certo na pos de memoria
				    contador++;                 // Conta a ocorrencia de erros por iteracao			
			    }
		    }	
		    printf("Contador -1: %lu\n",contador);
		    update_timestamp();    
	    }
#ifdef LOGS
        log_error_count(contador);                
#endif	
        printf("***************Acabei com o 1111 *******************\n");

#ifdef LOGS
        start_iteration();
#endif 
        contador = 0;
	    gold[0] = 0;
	
	    printf("Gold:%d\n",gold[0]);
	    fflush(stdout);
	    memset(vetor,0x00,size);
	    
        sleep(p.wait_time); // Sleep Between Write/Read iteration for error accumulation in DDR
#ifdef LOGS
        end_iteration();
#endif	  

	    for(k=0;k<p.internal_it;k++){	
	    #pragma omp parallel for reduction(+:contador) private(i)
		    for(i = 0 ; i< sys_mem; i++ ){
                //vetor[5]=12;
				//printf("Vetor[%d]:%d\n",i,vetor[i]);
			    if(vetor[i] != gold[0] ){
    			    error_time = (double) (get_time() - init_time) / 1000000;
                    printf("[%f] Gold: 0, It_Externa: %lu, It_Interna: %d, Pos[%d]:Sou um erro de Memoria E= %d, R= %d \n",error_time,cont, k,i,gold[0],vetor[i]);
#ifdef LOGS
		            char error_detail[200];
            		sprintf(error_detail,"[%f] Gold: 0, It_Externa: %lu, It_Interna: %d, Pos[%d]:Sou um erro de Memoria E= %d, R= %d \n",error_time,cont, k,i,gold[0],vetor[i]);
           			log_error_detail(error_detail);
#endif				    			        
				    vetor[i] = gold[0];
				    contador++;			
			    }
		    }
		    printf("Contador 0:%lu\n",contador);	
		    update_timestamp();	    	
	    }
#ifdef LOGS
        log_error_count(contador);
#endif
	    free(vetor);
	}
    show_time();
#ifdef LOGS
    log_info_detail(time_now);
    end_log_file();
#endif	

}
