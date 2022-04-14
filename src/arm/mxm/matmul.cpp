
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <cstring>
#include <chrono>
#include <getopt.h>
#include <cblas.h>
#include "log_helper.hpp"
#define MOD 1000

#define NUM_EXEC 1
#define PI 3.1415926535897932384626433


float *mA;
float *mB;
float *mCS0;
float *golden;


#define US_TO_S 0.000001
#define US_TO_MS 0.001

#define US_TO_MS 0.001

int compare(long iteration,int matrix_dim) {
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
				log_helper::log_error_detail(error_detail);
				errors++;


			}
		}
		//printf("a");
	}
	//printf("end");
	//########### control_dut ###########


	log_helper::log_error_count(errors);
	//printf("Iteration %ld errors %d\n", iteration, errors);

	
	return status_app;
}

void generate_gold_input(char *gold,char *input,int matrix_dim) {
	int i, j,k;
	//std::cout << "kek" << std::endl;
	FILE* f_golden = fopen(gold, "wb");
	FILE* fa = fopen(input, "wb");

	//std::cout << "lol 2.1" << std::endl;
	if (f_golden && fa) {
		for (i = 0; i < matrix_dim; i++) {
			for (j = 0; j < matrix_dim; j++) {
				mA[i * matrix_dim + j] = (float)rand()/(float)(RAND_MAX/1000.00);
				mB[i * matrix_dim + j] = (float)rand()/(float)(RAND_MAX/(i+1));
	//				floatrintf(f_golden, "%f ", mCS0[i * matrix_dim + j]);
			}
		}
		fwrite(mA, sizeof(float), matrix_dim * matrix_dim, fa);
		fwrite(mB, sizeof(float), matrix_dim * matrix_dim, fa);
		fclose(fa);
		
		//printf("PAssou doois\n");
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

 void read_input(char *inputa, char *gold, int matrix_dim){
	FILE *fin = fopen(inputa, "rb");

	FILE* f_golden = fopen(gold, "rb");

	fread(mA, sizeof(float), matrix_dim * matrix_dim, fin);
	fread(mB, sizeof(float), matrix_dim * matrix_dim, fin);
	fread(golden, sizeof(float), matrix_dim * matrix_dim, f_golden);
	fclose(fin);
	fclose(f_golden);
}


static struct option long_options[] = {
    /* name, has_arg, flag, val */
    {"input", 1, NULL, 'i'},
    {"gold", 1, NULL, 'g'},
    {"size", 1, NULL, 's'},
    {"generate", 0, NULL, 'c'},
    {"iterations", 1, NULL, 'l'},
    {0,0,0,0}
};

//---------------------------------------------------------------------------
int main(int argc, char **argv) {
		int matrix_dim = 0; /* default size */
	    int opt, option_index=0;
	    bool generate=false;
	    long long iteractions;
	    char *input_file = NULL;
	    char *gold_file = NULL;
    float *m, *gold;
	while ((opt = getopt_long(argc, argv, "i:g:l:s:c",
                              long_options, &option_index)) != -1 ) {
        switch(opt) {
        case 'i':
            input_file = optarg;
            break;
        case 'l':
            iteractions = atoll(optarg);
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
        case 'c':
            generate = true;
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
    log_helper::set_iter_interval_print(1);
    if ( (optind < argc) || (optind == 1)) {
        fprintf(stderr, "Usage: %s [-n no. of threads] [-s matrix_dim] [-i input_file] [-g gold_file] [-l #iterations]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

	if(generate) {
		std::cout << "Generating for " <<  argv[0] << " with size " << matrix_dim << std::endl;
	}else{
		char *benchmark_name = "sequential_mxm";
		char test_info[100];
		snprintf(test_info, 100, "MxM size:%d type:sequential_float", matrix_dim);
		log_helper::start_log_file(benchmark_name, test_info);
	}

	mA = (float*)malloc(sizeof(float*) * matrix_dim * matrix_dim);
	mB = (float*)malloc(sizeof(float*) * matrix_dim  * matrix_dim);
	mCS0 = (float*)malloc(sizeof(float*) * matrix_dim * matrix_dim);
	golden = (float*)malloc(sizeof(float*) * matrix_dim * matrix_dim);

	/*for(mt_siz = 0; mt_siz < matrix_dim; mt_siz++) {
		mA[mt_siz] = malloc(sizeof(float) * matrix_dim);
		mB[mt_siz] = malloc(sizeof(float) * matrix_dim);
		mCS0[mt_siz] = malloc(sizeof(float) * matrix_dim);
		golden[mt_siz] = malloc(sizeof(float) * matrix_dim);
	}*/

	//std::cout << "lol" <<generate<<" " <<input_file << " "<<gold_file << std::endl;

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
		generate_gold_input(gold_file,input_file,matrix_dim);
		//std::cout << "lol1.1" << std::endl;
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
	read_input(input_file, gold_file, matrix_dim);
	int iter = 0;
	while (iter < iteractions) {

		log_helper::start_iteration();

		cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,matrix_dim,matrix_dim,matrix_dim,1,mA,matrix_dim,mB,matrix_dim,0,mCS0,matrix_dim);

		log_helper::end_iteration();
		
		status_app = compare(iter++, matrix_dim);
	}

	log_helper::end_log_file();
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
