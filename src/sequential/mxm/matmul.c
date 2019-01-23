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

#define SIZE ((MATRIX_SIZE)*(MATRIX_SIZE))

#ifndef LOGS

#define MATRIX_SIZE 500 // matrix size
float mA[MATRIX_SIZE][MATRIX_SIZE];
float mB[MATRIX_SIZE][MATRIX_SIZE];
float mCS0[MATRIX_SIZE][MATRIX_SIZE];

#else
float **mA;
float **mB;
float **mCS0;
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

int compare(char* gold, long iteration) {
	//########### control_dut ###########
	//XTime_GetTime(&endexec);
	//if (count == 5)
	//{mCS0[30][47] = 2.35; count=0;}
	// check for errors
	//mCS0[10][20]--;
	//mCS0[30][20]--;
	int i, j;
	FILE* f_golden = fopen(gold, "r");
	float golden;
	int status_app = 0x00000000;
	int errors = 0;
	for (i = 0; i < MATRIX_SIZE; i++) {
		for (j = 0; j < MATRIX_SIZE; j++) {
			//printf("%f ",mCS0[i][j]);
			fscanf(f_golden, "%f", &golden);
			//printf("%.22f\n",mCS0[i][j]);
			if (mCS0[i][j] != golden) {
#ifndef LOGS
				if (status_app == 0) {
					buffer[0] = 0xDD000000;
				} else {
					buffer[0] = 0xCC000000;
				}
				//printf("%f\n",mCS0[i][j]);
				status_app = 1;
				//printf("\ni=%d j=%d \n %20.18f vs %20.18f\n",i,j,mCS0[i][j],float_golden[i][j]);

				buffer[1] = *((uint32_t*) &i);
				buffer[2] = *((uint32_t*) &j);
				buffer[3] = *((uint32_t*) &mCS0[i][j]); // u32, float has 32 bits
				send_message(4);
#else
				char error_detail[150];
				snprintf(error_detail, 150,
						"p: [%d, %d], r: %1.20e, e: %1.20e",
						i, j, mCS0[i][j], golden);
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
#endif
	fclose(f_golden);
	return status_app;
}

void generate_gold(char *gold) {
	int i, j;
	FILE* f_golden = fopen(gold, "w");
	if (f_golden) {
		for (i = 0; i < MATRIX_SIZE; i++) {
			for (j = 0; j < MATRIX_SIZE; j++) {
				fprintf(f_golden, "%f ", mCS0[i][j]);
			}
		}
		fclose(f_golden);
	}
}

//---------------------------------------------------------------------------
int main(int argc, char **argv) {
	int Status = 0;
	char *input, *gold;
	int generate = 0;
#ifndef LOGS
	unsigned int port = atoi(argv[2]);

	setup_socket(argv[1], port);
	input = argv[3];
	gold = argv[4];
#else
	input = argv[1];
	gold = argv[2];
	generate = atoi(argv[3]);
	MATRIX_SIZE = atoi(argv[4]);

	if(!generate) {
		char *benchmark_name = "sequential_mxm";
		char test_info[100];
		snprintf(test_info, 100, "size:%d type:sequential_float", MATRIX_SIZE);

		start_log_file(benchmark_name, test_info);
	}

	if(generate) {
		printf("Generating for %s with size %d\n", input, MATRIX_SIZE);
	}
	mA = malloc(sizeof(float*) * MATRIX_SIZE);
	mB = malloc(sizeof(float*) * MATRIX_SIZE);
	mCS0 = malloc(sizeof(float*) * MATRIX_SIZE);
	int mt_siz;
	for(mt_siz = 0; mt_siz < MATRIX_SIZE; mt_siz++) {
		mA[mt_siz] = malloc(sizeof(float) * MATRIX_SIZE);
		mB[mt_siz] = malloc(sizeof(float) * MATRIX_SIZE);
		mCS0[mt_siz] = malloc(sizeof(float) * MATRIX_SIZE);
	}

#endif

	int i = 0;
	int j = 0;
	int k = 0;
	int p = 0;
	int status_app;
	float golden;
	float a;
	float b;
	FILE *fin;

	long iteration = 0;
	while (1) {
		fin = fopen(input, "r");
		for (i = 0; i < MATRIX_SIZE; i++) {
			fscanf(fin, "%f %f", &a, &b);
			//printf("%.22f %.22f\n", a,b);
			for (j = 0; j < MATRIX_SIZE; j++) {
				mA[i][j] = a;
				mB[i][j] = b;

			}
		}
		fclose(fin);

#ifdef LOGS
		if(!generate) {
			start_iteration();
		}
#endif
		for (i = 0; i < MATRIX_SIZE; i++) {
			for (j = 0; j < MATRIX_SIZE; j++) {
				mCS0[i][j] = 0.0;
				for (k = 0; k < MATRIX_SIZE; k++)
					mCS0[i][j] += mA[i][k] * mB[k][j];
			}
		}
#ifdef LOGS
		if(!generate) {
			end_iteration();
		}
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
#ifdef LOGS
		if(!generate) {
#endif
		status_app = compare(gold, iteration++);
#ifdef LOGS
	}
#endif
		if (status_app == 0x00000000) { // sem erros
			//printf("ok\n");
			buffer[0] = APP_SUCCESS; //sem erros
			//send_message(1);
		}

#ifdef LOGS
		if (generate) {
			generate_gold(gold);
			break;
		}
#endif
	}

#ifdef LOGS
	if(!generate) {
		end_log_file();
	}
	for(mt_siz = 0; mt_siz < MATRIX_SIZE; mt_siz++) {
		free(mA[mt_siz]);
		free(mB[mt_siz]);
		free(mCS0[mt_siz]);
	}
	free(mA);
	free(mB);
	free(mCS0);
#endif

	return 0;
}
