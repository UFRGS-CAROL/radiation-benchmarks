#include<arpa/inet.h>
#include<sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include<sys/socket.h>

#ifdef LOGS
#include <log_helper.h>
#endif

#define MOD 1000

#define NUM_EXEC 1
#define PI 3.1415926535897932384626433


#ifndef LOGS

#define MATRIX_SIZE 500 // matrix size
float mA[MATRIX_SIZE * MATRIX_SIZE];
float mB[MATRIX_SIZE * MATRIX_SIZE];
float mCS0[MATRIX_SIZE * MATRIX_SIZE];
float golden[MATRIX_SIZE * MATRIX_SIZE];
#else
float *mA;
float *mB;
float *mCS0;
float *golden;
int MATRIX_SIZE;
#endif

int s;
struct sockaddr_in server;
unsigned int buffer[4];

#define US_TO_S 0.000001
#define US_TO_MS 0.001

#define APP_SUCCESS            0xAA000000

#define APP_SDC            	   0xDD000000 //SDC

// 1 if using control_dut
int control_dut_inuse = 1;

#define US_TO_MS 0.001

void setup_socket(char* ip_addr, int port) {
	s = socket(PF_INET, SOCK_DGRAM, 0);
	//memset(&server, 0, sizeof(struct sockaddr_in));
	//printf("port: %d",port);
	//printf("ip: %s", ip_addr);
	server.sin_family = AF_INET;
	server.sin_port = htons(port);
	server.sin_addr.s_addr = inet_addr(ip_addr);

}

void send_message(size_t size) {
	//printf("message sent\n");
	sendto(s, buffer, 4 * size, 0, (struct sockaddr *) &server, sizeof(server));
}

int compare(long iteration) {
	//########### control_dut ###########
	//XTime_GetTime(&endexec);
	//if (count == 5)
	//{mCS0[30][47] = 2.35; count=0;}
	// check for errors
	//mCS0[10][20]--;
	//mCS0[30][20]--;
	int i, j;

	int status_app = 0x00000000;
	int errors = 0;
	for (i = 0; i < MATRIX_SIZE; i++) {
		for (j = 0; j < MATRIX_SIZE; j++) {
			
			//printf("%.22f\n",mCS0[i][j]);
			if (mCS0[i * MATRIX_SIZE + j] != golden[i * MATRIX_SIZE + j]) {
#ifndef LOGS
				if (status_app == 0) {
					buffer[0] = 0xDD000000;
				} else {
					buffer[0] = 0xCC000000;
				}
				
				status_app = 1;
				//printf("\ni=%d j=%d \n %20.18f vs %20.18f\n",i,j,mCS0[i * MATRIX_SIZE + j],float_golden[i * MATRIX_SIZE + j]);
			
				buffer[1] = *((uint32_t*) &i);
				buffer[2] = *((uint32_t*) &j);
				buffer[3] = *((uint32_t*) &mCS0[i * MATRIX_SIZE + j]); // u32, float has 32 bits
				send_message(4);
#else
				char error_detail[150];
				//printf("oops %f %f\n",mCS0[i * MATRIX_SIZE + j],golden[i * MATRIX_SIZE + j]);
				snprintf(error_detail, 150,
						"p: [%d, %d], r: %1.20e, e: %1.20e",
						i, j, mCS0[i * MATRIX_SIZE + j], golden[i * MATRIX_SIZE + j]);
				//printf("%s\n", error_detail);
				log_error_detail(error_detail);
				errors++;

#endif
			}
		}
		//printf("a");
	}
	//printf("end");
	//########### control_dut ###########

#ifdef LOGS
	log_error_count(errors);
	printf("Iteration %ld errors %d\n", iteration, errors);
#else
	printf("STATUS: %d\n", status_app);
#endif
	
	return status_app;
}

void generate_gold_input(char *gold,char *mat_a,char *mat_b) {
	int i, j,k;
	FILE* f_golden = fopen(gold, "wb");
	FILE* fa = fopen(mat_a, "wb");
	FILE* fb = fopen(mat_b, "wb");
	printf("passou aqui\n");
	if (f_golden && fa && fb) {
		for (i = 0; i < MATRIX_SIZE; i++) {
			for (j = 0; j < MATRIX_SIZE; j++) {
				mA[i * MATRIX_SIZE + j] = (float)rand()/(float)(RAND_MAX/1000.00);
				mB[i * MATRIX_SIZE + j] = (float)rand()/(float)(RAND_MAX/(i+1));
	//				fprintf(f_golden, "%f ", mCS0[i * MATRIX_SIZE + j]);
			}
		}
		fwrite(mA, sizeof(float), MATRIX_SIZE * MATRIX_SIZE, fa);
		fwrite(mB, sizeof(float), MATRIX_SIZE * MATRIX_SIZE, fb);
		fclose(fa);
		fclose(fb);
		printf("PAssou doois\n");
		for (i = 0; i < MATRIX_SIZE; i++) {
			for (j = 0; j < MATRIX_SIZE; j++) {
				mCS0[i * MATRIX_SIZE + j] = 0.0;
				for (k = 0; k < MATRIX_SIZE; k++){
					mCS0[i * MATRIX_SIZE + j] += mA[i * MATRIX_SIZE +k] * mB[k* MATRIX_SIZE + j];
				}
			}

		}
		fwrite( mCS0, sizeof(float), MATRIX_SIZE * MATRIX_SIZE,f_golden);
		fclose(f_golden);
	}
}

 void read_input(char *inputa, char *inputb, char *gold){
	FILE *fina = fopen(inputa, "rb");
	FILE *finb = fopen(inputb, "rb");
	FILE* f_golden = fopen(gold, "rb");

	fread(mA, sizeof(float), MATRIX_SIZE * MATRIX_SIZE, fina);
	fread(mB, sizeof(float), MATRIX_SIZE * MATRIX_SIZE, finb);
	fread(golden, sizeof(float), MATRIX_SIZE * MATRIX_SIZE, f_golden);
	fclose(fina);
	fclose(finb);
	fclose(f_golden);
}


//---------------------------------------------------------------------------
int main(int argc, char **argv) {
	int Status = 0;


	unsigned int port = atoi(argv[2]);
	setup_socket(argv[1], port);
	char *inputa = argv[3];
	char *inputb = argv[4];
	char *gold = argv[5];
	int generate = atoi(argv[6]);
		int mt_siz;

#ifdef LOGS
	MATRIX_SIZE = atoi(argv[7]);
	if(!generate) {
		char *benchmark_name = "sequential_mxm";
		char test_info[100];
		snprintf(test_info, 100, "size:%d type:sequential_float", MATRIX_SIZE);

		start_log_file(benchmark_name, test_info);
	}

	if(generate) {
		printf("Generating for %s with size %d\n", argv[0], MATRIX_SIZE);
	}
	mA = malloc(sizeof(float*) * MATRIX_SIZE * MATRIX_SIZE);
	mB = malloc(sizeof(float*) * MATRIX_SIZE  * MATRIX_SIZE);
	mCS0 = malloc(sizeof(float*) * MATRIX_SIZE * MATRIX_SIZE);
	golden = malloc(sizeof(float*) * MATRIX_SIZE * MATRIX_SIZE);

	/*for(mt_siz = 0; mt_siz < MATRIX_SIZE; mt_siz++) {
		mA[mt_siz] = malloc(sizeof(float) * MATRIX_SIZE);
		mB[mt_siz] = malloc(sizeof(float) * MATRIX_SIZE);
		mCS0[mt_siz] = malloc(sizeof(float) * MATRIX_SIZE);
		golden[mt_siz] = malloc(sizeof(float) * MATRIX_SIZE);
	}*/

#endif

	int i = 0;
	int j = 0;
	int k = 0;
	int p = 0;
	int status_app;
	float a;
	float b;
	FILE *fin;

	printf("%d %s %s %s %d\n", generate, inputa, inputb, gold, MATRIX_SIZE);

	if(generate){
		generate_gold_input(gold,inputa,inputb);
#ifdef LOGS
		/*for(mt_siz = 0; mt_siz < MATRIX_SIZE; mt_siz++) {
			free(mA[mt_siz]);
			free(mB[mt_siz]);
			free(mCS0[mt_siz]);
		}*/
	
		free(mA);
		free(mB);
		free(mCS0);
#endif
		return 0;
	}

	long iteration = 0;


	//for (i = 0; i < MATRIX_SIZE; i++) {

		
	//}
	read_input(inputa, inputb, gold);

	while (1) {

#ifdef LOGS
		start_iteration();
#endif
		for (i = 0; i < MATRIX_SIZE; i++) {
			for (j = 0; j < MATRIX_SIZE; j++) {
				mCS0[i * MATRIX_SIZE + j] = 0.0;
				for (k = 0; k < MATRIX_SIZE; k++)
					mCS0[i * MATRIX_SIZE + j] += mA[i * MATRIX_SIZE + k] * mB[k * MATRIX_SIZE + j];
			}
		}
#ifdef LOGS
		end_iteration();
#endif
		//XTime tStart, tEnd, endexec;
		int cont = 0;

		//XTime_GetTime(&tStart);
		//XTime tStart, tEnd;
		//XTime_GetTime(&tStart);
		//printf("0\n");
		//########### control_dut ###########

		status_app = 0x00000000;
		//########### control_dut ###########

		//XTime_GetTime(&endexec);
		//if (count == 5)
		//{mCS0[30][47] = 2.35; count=0;}

		// check for errors
		//mCS0[10][20]--;
		//mCS0[30][20]--;
		status_app = compare(iteration++);

		if (status_app == 0x00000000) { // sem erros
			buffer[0] = APP_SUCCESS; //sem erros
			send_message(1);
		}else{
			read_input(inputa, inputb, gold);
		}

	}

#ifdef LOGS
	end_log_file();
	/*for(mt_siz = 0; mt_siz < MATRIX_SIZE; mt_siz++) {
		free(mA[mt_siz]);
		free(mB[mt_siz]);
		free(mCS0[mt_siz]);
	}*/
	free(mA);
	free(mB);
	free(mCS0);
#endif

	return 0;
}
