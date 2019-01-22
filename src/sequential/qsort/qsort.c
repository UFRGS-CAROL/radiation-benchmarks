#define _GNU_SOURCE
#include <stdio.h>
#include <math.h>
#include<arpa/inet.h>
#include<sys/socket.h>
#include <stdlib.h>

#define UNLIMIT
#define MAXARRAY 300000 /* this number, if too large, will cause a seg. fault!! */


#define MOD 1000

#define NUM_EXEC 100

#define US_TO_S 0.000001
#define US_TO_MS 0.001

#define APP_SUCCESS            0xAA000000
#define APP_SDC            	   0xDD000000 //SDC
int s;
struct sockaddr_in server;
unsigned int buffer[4];

// TIMER instance
//XTmrCtr timer_dev;

void setup_socket(char* ip_addr, int port){
	s=socket(PF_INET, SOCK_DGRAM, 0);
	//memset(&server, 0, sizeof(struct sockaddr_in));
	//printf("port: %d",port);
	//printf("ip: %s", ip_addr);
	server.sin_family = AF_INET;
	server.sin_port = htons(port);
	server.sin_addr.s_addr = inet_addr(ip_addr);

}

void send_message(size_t size){
    //printf("message sent\n");
	sendto(s,buffer,4*size,0,(struct sockaddr *)&server,sizeof(server));
}


void qsort(void *base, size_t nitems, size_t size, int (*compar)(const void *, const void*));
int compare(const void *elem1, const void *elem2)
{
  /* D = [(x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2]^(1/2) */
  /* sort based on distances from the origin... */

  return (*((double*)elem1) > *((double*)elem2)) ? 1 : ((*((double*)elem1) == *((double*)elem2)) ? 0 : -1);
}



//---------------------------------------------------------------------------
int main(int argc, char **argv)
{
	int Status = 0;
    unsigned int port = atoi(argv[2]);
    setup_socket(argv[1],port);
    int status_app=0;


    double array[MAXARRAY];
    //long long temp_gold[MAXARRAY];
    double gold;
    int i;
    int ex=0;
    uint64_t aux;
    int endexec=0;
    int count=0;
    FILE* fp;
		FILE* f_gold;

		int cont=0;

    while(1)
    {
			fp = fopen(argv[3],"r");
			f_gold= fopen(argv[4],"r");
    	//printf("0\n");

    	status_app    = 0;
    	//########### control_dut ###########




        while((fscanf(fp, "%lf", &array[count]) == 1) &&  (count < MAXARRAY)) {

    	 		count++;
        }
        fclose(fp);
				//printf("comecou qsort\n");
	    	qsort(array,MAXARRAY,sizeof(double),compare);
				//printf("acabou qsort\n");
    	//XTime_GetTime(&tEnd);
    	//printf("%.8f us\n",1.0*(tEnd - tStart)/(COUNTS_PER_SECOND/1000000));


        //to test faults
    	//if (count == 100){
        //    array[30].distance = 15.21;array[32].distance = 40.85;array[31].distance = 20.21;array[33].distance = 17.21; count=0;
        //}
        // check for errors

				int num_error=0;

        for (i=0;i<MAXARRAY;i++)
        {

            //printf("%d\n",i);
						fscanf(f_gold, "%lf", &gold);
						//printf("lol\n",i);
            if (array[i] != gold)
              {
								printf("%d %lf %lf\n",i,array[i],gold);
									status_app=1;
									num_error++;


              }
        	//printf("a");
        }
				//printf("terminou teste\n");
				fclose(f_gold);

        //printf("end");
        //while(1);
    	//########### control_dut ###########
    	if (status_app == 0) // sem erros
    	{
            //printf("ok\n");
    		buffer[0] = APP_SUCCESS; //sem erros
    		send_message(1);
    		ex=0;

      }else{
				//printf("erro: %d",num_error);
				buffer[0] = 0xDD000000; // SDC
				buffer[1] = *((uint32_t*)&num_error);
				send_message(2);
			}


				//return 0;

        //printf("1\n");
        ex++;
				cont++;
    }

    return 0;
}
