#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<arpa/inet.h>
#include<sys/socket.h>
#include "fourier.h"
#define  DDC_PI  (3.14159265358979323846)
#define CHECKPOINTER(p)  CheckPointer(p,#p)
void fft_float(unsigned NumSamples, int InverseTransform, float *RealIn,
		float *ImagIn, float *RealOut, float *ImagOut);
static void CheckPointer(void *p, char *name);
#define TRUE  1
#define FALSE 0
#define NUM_EXEC 100
#define BITS_PER_WORD   (sizeof(unsigned) * 8)

int MAXSIZE;
int MAXWAVES;

#ifdef LOGS
#include "log_helper.h"
//#else
//#define MAXSIZE 262144
//#define MAXWAVES 8
#endif

int s;
struct sockaddr_in server;
unsigned int buffer[4];

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

void create_input(int max_size, int max_waves, char *fin_path) {
	float* real_in = (float*) malloc(sizeof(float) * max_size);
//	imag_in = (float*) malloc(sizeof(float) * max_size);
//	real_out = (float*) malloc(sizeof(float) * max_size);
//	imag_out = (float*) malloc(sizeof(float) * max_size);
	float *coeff = (float*) malloc(sizeof(float) * max_waves);
	float *amp = (float*) malloc(sizeof(float) * max_waves);
	int i, j;
	for (i = 0; i < max_waves; i++) {

		coeff[i] = rand() % 1000;

		amp[i] = rand() % 1000;

	}

	for (i = 0; i < max_size; i++) {
		/*   RealIn[i]=rand();*/
		real_in[i] = 0;
		for (j = 0; j < max_waves; j++) {
			/* randomly select sin or cos */
			if (rand() % 2) {

				real_in[i] += coeff[j] * cos(amp[j] * i);

			} else {
				real_in[i] += coeff[j] * sin(amp[j] * i);
			}
//			imag_in[i] = 0;
		}
	}

	FILE* fin = fopen(fin_path, "w");
	for (i = 0; i < MAXSIZE; i++) {
		fprintf(fin, "%f\n", real_in[i]);
	}
	fclose(fin);
	free(coeff);
	free(amp);
	free(real_in);
}

int main(int argc, char *argv[]) {
	unsigned i, j;
	int ex;
	float *RealIn;
	float *ImagIn;
	float *RealOut;
	float *ImagOut;

	int invfft = 0;
	float goldRealOut;
	float goldImagOut;
	FILE *f_golden_real;
	FILE *f_golden_img;
	FILE* fin;
	int status_app = 0;
	unsigned int port = atoi(argv[2]);
	setup_socket(argv[1], port);
	char *fin_path = argv[3];
	char *f_golden_real_path = argv[4];

	MAXSIZE = atoi(argv[5]);
	MAXWAVES = atoi(argv[6]);
#ifdef LOGS
	int generate = atoi(argv[7]);

	printf("Executing for infile %s golden file %s generate %d maxsize %d maxwaves %d\n", fin_path, f_golden_real_path, generate, MAXSIZE, MAXWAVES);

	if(generate) {
		create_input(MAXSIZE, MAXWAVES, fin_path);
	} else {
		char *benchmark_name = "SequentialFFT";
		char test_info[100];
		snprintf(test_info, 100, "size:%d waves_size:%d", MAXSIZE, MAXWAVES);

		start_log_file(benchmark_name, test_info);
	}

	unsigned long long iterations = 0;

#endif

	while (1) {
		fin = fopen(fin_path, "r");
		if (!fin) {
			printf("error at opening golden file real\n");
		}
		status_app = 0;
		srand(1);

		RealIn = (float*) malloc(sizeof(float) * MAXSIZE);
		ImagIn = (float*) malloc(sizeof(float) * MAXSIZE);
		RealOut = (float*) malloc(sizeof(float) * MAXSIZE);
		ImagOut = (float*) malloc(sizeof(float) * MAXSIZE);

		for (i = 0; i < MAXSIZE; i++) {
			/*   RealIn[i]=rand();*/
			fscanf(fin, "%f", &RealIn[i]);

			ImagIn[i] = 0;

			// printf("%.22f ",RealIn[i]);
		}

		/* regular*/
#ifdef LOGS
		if(!generate) {
			start_iteration();
		}
#endif
		fft_float(MAXSIZE, invfft, RealIn, ImagIn, RealOut, ImagOut);
#ifdef LOGS
		if(!generate) {
			end_iteration();
		}
#endif

#ifndef LOGS
		f_golden_real = fopen(f_golden_real_path, "r");

		if (!f_golden_real) {
			printf("error at opening golden file real\n");
		}

		for (i = 0; i < MAXSIZE; i++) {

			fscanf(f_golden_real, "%f %f", &goldRealOut, &goldImagOut);
			//printf("%.22f %.22f ", RealOut[i],ImagOut[i]);
			//printf("%u\n", *(unsigned int*)&ImagOut[i]);
			if ((RealOut[i] != goldRealOut) || (ImagOut[i] != goldImagOut)) {
				if (status_app == 0) {
					//xil_printf ("SDC  ");
					//printf("error at index: %i\n\r(%f != %f) || (%f != %f)\n\r", i, RealOut[i], goldRealOut, ImagOut[i], goldImagOut[i]);
					buffer[0] = 0xDD000000;
					buffer[1] = *((uint32_t*) &i);
					buffer[2] = *((uint32_t*) &RealOut[i]);
					buffer[3] = *((uint32_t*) &ImagOut[i]); // u32, float has 32 bits
					send_message(4);

				} else {
					//xil_printf ("SDC  ");
					//printf("error at index: %i\n\r(%f != %f) || (%f != %f)\n\r", i, RealOut[i], goldRealOut[i], ImagOut[i], goldImagOut[i]);
					buffer[0] = 0xCC000000;
					buffer[1] = *((uint32_t*) &i);
					buffer[2] = *((uint32_t*) &RealOut[i]);
					buffer[3] = *((uint32_t*) &ImagOut[i]); // u32, float has 32 bits
					send_message(4);

				}
				status_app = 1;
			}

		}

		if (status_app == 0) {
			//printf("ok\n");
			buffer[0] = 0xAA000000; //sem erros
			send_message(1);
		}

#else
		if(generate) {
			f_golden_real = fopen(f_golden_real_path, "w");
		} else {
			f_golden_real = fopen(f_golden_real_path, "r");
		}
		unsigned int errors = 0;
		for (i = 0; i < MAXSIZE; i++) {
			if(generate) {
				fprintf(f_golden_real, "%f %f\n", RealOut[i], ImagOut[i]);

			} else {
				fscanf(f_golden_real, "%f %f", &goldRealOut, &goldImagOut);
				//printf("%.22f %.22f ", RealOut[i],ImagOut[i]);
				//printf("%u\n", *(unsigned int*)&ImagOut[i]);
				if ((RealOut[i] != goldRealOut) || (ImagOut[i] != goldImagOut)) {
					errors++;

					char error_detail[300];
					snprintf(error_detail, 300,
							"p: [%d], realOut_e: %1.20e, realOut_r: %1.20e imagOut_e: %1.20e imagOut_r: %1.20e",
							i, goldRealOut, RealOut[i], goldImagOut, ImagOut[i]);
					log_error_detail(error_detail);

				}

			}

		}
		if(errors) {
			log_error_count(errors);
		}
		printf("ITERATION: %lld errors %d\n", iterations++, errors);
		//  printf("ended\n");

		if(generate) {
			exit(0);
		}

#endif
		free(RealIn);
		free(ImagIn);
		free(RealOut);
		free(ImagOut);

		//printf("ok\n");
		fclose(f_golden_real);
		fclose(fin);
		//printf("fff\n");
		//return 0;

//return 0;
	}

#ifdef LOGS
	if(!generate) {
		end_log_file();
	}
#endif
	exit(0);
}
