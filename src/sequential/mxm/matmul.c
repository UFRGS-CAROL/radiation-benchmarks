


#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <log_helper.h>

#define MOD 1000

#define NUM_EXEC 1
#define PI 3.1415926535897932384626433


float *mA;
float *mB;
float *mCS0;
float *golden;


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
	for (i = 0; i < matrix_dim; i++) {
		for (j = 0; j < matrix_dim; j++) {
			
			//printf("%.22f\n",mCS0[i][j]);
			if (mCS0[i * matrix_dim + j] != golden[i * matrix_dim + j]) {
				char error_detail[150];
				//printf("oops %f %f\n",mCS0[i * matrix_dim + j],golden[i * matrix_dim + j]);
				snprintf(error_detail, 150,
						"p: [%d, %d], r: %1.20e, e: %1.20e",
						i, j, mCS0[i * matrix_dim + j], golden[i * matrix_dim + j]);
				//printf("%s\n", error_detail);
				log_error_detail(error_detail);
				errors++;


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

void generate_gold_input(char *gold,char *input,int matrix_dim) {
	int i, j,k;
	FILE* f_golden = fopen(gold, "wb");
	FILE* fa = fopen(mat_a, "wb");

	printf("passou aqui\n");
	if (f_golden && fa && fb) {
		for (i = 0; i < matrix_dim; i++) {
			for (j = 0; j < matrix_dim; j++) {
				mA[i * matrix_dim + j] = (float)rand()/(float)(RAND_MAX/1000.00);
				mB[i * matrix_dim + j] = (float)rand()/(float)(RAND_MAX/(i+1));
	//				fprintf(f_golden, "%f ", mCS0[i * matrix_dim + j]);
			}
		}
		fwrite(mA, sizeof(float), matrix_dim * matrix_dim, fa);
		fwrite(mB, sizeof(float), matrix_dim * matrix_dim, fa);
		fclose(fa);
		fclose(fb);
		printf("PAssou doois\n");
		for (i = 0; i < matrix_dim; i++) {
			for (j = 0; j < matrix_dim; j++) {
				mCS0[i * matrix_dim + j] = 0.0;
				for (k = 0; k < matrix_dim; k++){
					mCS0[i * matrix_dim + j] += mA[i * matrix_dim +k] * mB[k* matrix_dim + j];
				}
			}

		}
		fwrite( mCS0, sizeof(float), matrix_dim * matrix_dim,f_golden);
		fclose(f_golden);
	}
}

 void read_input(char *inputa, char *inputb, char *gold, int matrix_dim){
	FILE *fin = fopen(inputa, "rb");

	FILE* f_golden = fopen(gold, "rb");

	fread(mA, sizeof(float), matrix_dim * matrix_dim, fin);
	fread(mB, sizeof(float), matrix_dim * matrix_dim, fin);
	fread(golden, sizeof(float), matrix_dim * matrix_dim, f_golden);
	fclose(fina);
	fclose(finb);
	fclose(f_golden);
}


static struct option long_options[] = {
    /* name, has_arg, flag, val */
    {"input", 1, NULL, 'i'},
    {"gold", 1, NULL, 'g'},
    {"size", 1, NULL, 's'},
    {0,0,0,0}
};

//---------------------------------------------------------------------------
int main(int argc, char **argv) {
		int matrix_dim = 0; /* default size */
	    int opt, option_index=0;
	    func_ret_t ret;
	    long long iteractions;
	    const char *input_file = NULL;
	    const char *gold_file = NULL;
    FP *m, *gold;
	while ((opt = getopt_long(argc, argv, "::s:n:i:g:l:",
                              long_options, &option_index)) != -1 ) {
        switch(opt) {
        case 'i':
            input_file = optarg;
            break;
        case 'l':
            iteractions = atoi(optarg);
            if(iteractions <=0) {
                printf("Error, invalid number of iteractions\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'g':
            gold_file = optarg;
            break; 
        case 's':
            matrix_dim = atoi(optarg);
            break;
        case '?':
            fprintf(stderr, "invalid option\n");
            break;
        case ':':
            fprintf(stderr, "missing argument\n");
            break;
        default:
            fprintf(stderr, "Usage: %s [-v] [-s matrix_dim|-i input_file]\n",
                    argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if ( (optind < argc) || (optind == 1)) {
        fprintf(stderr, "Usage: %s [-n no. of threads] [-s matrix_dim] [-i input_file] [-g gold_file] [-l #iterations]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
	int mt_siz;



	if(!generate) {
		char *benchmark_name = "sequential_mxm";
		char test_info[100];
		snprintf(test_info, 100, "size:%d type:sequential_float", matrix_dim);

		start_log_file(benchmark_name, test_info);
	}

	if(generate) {
		printf("Generating for %s with size %d\n", argv[0], matrix_dim);
	}
	mA = malloc(sizeof(float*) * matrix_dim * matrix_dim);
	mB = malloc(sizeof(float*) * matrix_dim  * matrix_dim);
	mCS0 = malloc(sizeof(float*) * matrix_dim * matrix_dim);
	golden = malloc(sizeof(float*) * matrix_dim * matrix_dim);

	/*for(mt_siz = 0; mt_siz < matrix_dim; mt_siz++) {
		mA[mt_siz] = malloc(sizeof(float) * matrix_dim);
		mB[mt_siz] = malloc(sizeof(float) * matrix_dim);
		mCS0[mt_siz] = malloc(sizeof(float) * matrix_dim);
		golden[mt_siz] = malloc(sizeof(float) * matrix_dim);
	}*/



	int i = 0;
	int j = 0;
	int k = 0;
	int p = 0;
	int status_app;
	float a;
	float b;
	FILE *fin;

	//printf("%d %s %s %s %d\n", generate, inputa, inputb, gold, matrix_dim);

	if(generate){
		generate_gold_input(gold,inputa,inputb);

		/*for(mt_siz = 0; mt_siz < matrix_dim; mt_siz++) {
			free(mA[mt_siz]);
			free(mB[mt_siz]);
			free(mCS0[mt_siz]);
		}*/
	
		free(mA);
		free(mB);
		free(mCS0);

		return 0;
	}

	


	//for (i = 0; i < matrix_dim; i++) {

		
	//}
	read_input(inputa, inputb, gold, matrix_dim);

	while (iteractions>0) {

		start_iteration();

		for (i = 0; i < matrix_dim; i++) {
			for (j = 0; j < matrix_dim; j++) {
				mCS0[i * matrix_dim + j] = 0.0;
				for (k = 0; k < matrix_dim; k++)
					mCS0[i * matrix_dim + j] += mA[i * matrix_dim + k] * mB[k * matrix_dim + j];
			}
		}

		end_iteration();
		
		status_app = compare(iteractions--);
	}

	end_log_file();
	/*for(mt_siz = 0; mt_siz < matrix_dim; mt_siz++) {
		free(mA[mt_siz]);
		free(mB[mt_siz]);
		free(mCS0[mt_siz]);
	}*/
	free(mA);
	free(mB);
	free(mCS0);


	return 0;
}
