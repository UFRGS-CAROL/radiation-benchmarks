#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<arpa/inet.h>
#include<sys/socket.h>
#include "fourier.h"
#define  DDC_PI  (3.14159265358979323846)
#define CHECKPOINTER(p)  CheckPointer(p,#p)
void fft_float (
    unsigned  NumSamples,
    int       InverseTransform,
    float    *RealIn,
    float    *ImagIn,
    float    *RealOut,
    float    *ImagOut );
static void CheckPointer ( void *p, char *name );
#define TRUE  1
#define FALSE 0
#define NUM_EXEC 100
#define BITS_PER_WORD   (sizeof(unsigned) * 8)

int s;
struct sockaddr_in server;
unsigned int buffer[4];

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
#define MAXSIZE 262144
#define MAXWAVES 8

int main(int argc, char *argv[]) {
	unsigned i,j;
    int ex;
	float *RealIn;
	float *ImagIn;
	float *RealOut;
	float *ImagOut;

	int invfft=0;
  float goldRealOut;
  float goldImagOut;
  FILE *f_golden_real;
  FILE *f_golden_img;
  FILE* fin;
    int status_app=0;
    unsigned int port = atoi(argv[2]);
    setup_socket(argv[1],port);

    while(1){

            fin=fopen(argv[3], "r");
            if(!fin){
              printf("error at opening golden file real\n");
            }
            status_app=0;
            srand(1);

            RealIn=(float*)malloc(sizeof(float)*MAXSIZE);
            ImagIn=(float*)malloc(sizeof(float)*MAXSIZE);
            RealOut=(float*)malloc(sizeof(float)*MAXSIZE);
            ImagOut=(float*)malloc(sizeof(float)*MAXSIZE);

            for(i=0;i<MAXSIZE;i++)
            {
                  /*   RealIn[i]=rand();*/
                  fscanf(fin,"%f",&RealIn[i]);

                 	 ImagIn[i]=0;

                // printf("%.22f ",RealIn[i]);
            }

            /* regular*/
            fft_float (MAXSIZE,invfft,RealIn,ImagIn,RealOut,ImagOut);
            f_golden_real=fopen(argv[4], "r");
            if(!f_golden_real){
              printf("error at opening golden file real\n");
            }


            for(i=0; i<MAXSIZE; i++)
            {

              fscanf(f_golden_real,"%f %f",&goldRealOut,&goldImagOut);
              //printf("%.22f %.22f ", RealOut[i],ImagOut[i]);
              //printf("%u\n", *(unsigned int*)&ImagOut[i]);
		          if((RealOut[i] != goldRealOut) || (ImagOut[i] != goldImagOut))
                  {
                    	if(status_app == 0)
                    	{
                          		//xil_printf ("SDC  ");
                          		//printf("error at index: %i\n\r(%f != %f) || (%f != %f)\n\r", i, RealOut[i], goldRealOut, ImagOut[i], goldImagOut[i]);
                          		buffer[0] = 0xDD000000;
                          		buffer[1] = *((uint32_t*)&i);
                          		buffer[2] = *((uint32_t*)&RealOut[i]);
                          		buffer[3] = *((uint32_t*)&ImagOut[i]); // u32, float has 32 bits
                          		send_message(4);

                    	}
                    	else
              				{
                          		//xil_printf ("SDC  ");
                          		//printf("error at index: %i\n\r(%f != %f) || (%f != %f)\n\r", i, RealOut[i], goldRealOut[i], ImagOut[i], goldImagOut[i]);
                          		buffer[0] = 0xCC000000;
                          		buffer[1] = *((uint32_t*)&i);
                          		buffer[2] = *((uint32_t*)&RealOut[i]);
                          		buffer[3] = *((uint32_t*)&ImagOut[i]); // u32, float has 32 bits
                          		send_message(4);

              				}
                      status_app=1;
                  }
            }
          //  printf("ended\n");

        free(RealIn);
        free(ImagIn);
        free(RealOut);
        free(ImagOut);

        //printf("ok\n");
        fclose(f_golden_real);

        //printf("fff\n");
        //return 0;

    if(status_app==0){
        //printf("ok\n");
        buffer[0] = 0xAA000000; //sem erros
        send_message(1);
    }
//return 0;
    }

    exit(0);
}
