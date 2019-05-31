/*
 * =====================================================================================
 *
 *       Filename:  suite.c
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>

#include "common.h"
#include "../../include/log_helper.h"

#ifdef TIMING
long long timing_get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

long long setup_start, setup_end;
long long loop_start, loop_end;
long long kernel_start, kernel_end;
long long check_start, check_end;
#endif


static int do_verify = 0;
int omp_num_threads = 228;

static struct option long_options[] = {
    /* name, has_arg, flag, val */
    {"input", 1, NULL, 'i'},
    {"gold", 1, NULL, 'g'},
    {"size", 1, NULL, 's'},
    {0,0,0,0}
};

extern void
lud_omp(float *m, int matrix_dim);

int iteractions = 100000;
int
main ( int argc, char *argv[] )
{

#ifdef TIMING
    setup_start = timing_get_time();
#endif
    int matrix_dim = 0; /* default size */
    int opt, option_index=0;
    func_ret_t ret;
    const char *input_file = NULL;
    const char *gold_file = NULL;
    float *m, *gold;
    stopwatch sw;


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
        case 'v':
            do_verify = 1;
            break;
        case 'n':
            omp_num_threads = atoi(optarg);
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
            fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
                    argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if ( (optind < argc) || (optind == 1)) {
        fprintf(stderr, "Usage: %s [-n no. of threads] [-s matrix_size] [-i input_file] [-g gold_file] [-l #iterations]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    if (input_file && gold_file && matrix_dim>0) {
        int n = matrix_dim;
        FILE *f_a, *f_gold;
        f_a = fopen(input_file, "rb");
        f_gold = fopen(gold_file, "rb");

        if ((f_a == NULL) || (f_gold == NULL)) {
            printf("Error opening files\n");
            exit(EXIT_FAILURE);
        }

        m = (float*) malloc(sizeof(float)*n*n);
        gold = (float*) malloc(sizeof(float)*n*n);
        //std::cout << "read...";
        fread(m, sizeof(float) * n * n, 1, f_a);
        fread(gold, sizeof(float) * n * n, 1, f_gold);
        fclose(f_a);
        fclose(f_gold);
    }
    else {
        printf("No input, gold, or matrix_dim specified!\n");
        exit(EXIT_FAILURE);
    }

#ifdef LOGS
    char test_info[200];
    snprintf(test_info, 200, "matrix_dim:%d threads:%d", matrix_dim, omp_num_threads);
    start_log_file("openmpLUD", test_info);
#endif
#ifdef TIMING
    setup_end = timing_get_time();
#endif
    int loop;
    for(loop=0; loop<iteractions; loop++) {
#ifdef TIMING
        loop_start = timing_get_time();
#endif
#ifdef ERR_INJ
        if(loop == 2) {
            printf("injecting error, changing input!\n");
            m[0] = 102012;
            m[10] = 102012;
            m[55] = 102012;
        } else if (loop == 3) {
            printf("get ready, infinite loop...\n");
            fflush(stdout);
            while(1) {
                sleep(100);
            }
        }
#endif


#ifdef TIMING
        kernel_start = timing_get_time();
#endif
#ifdef LOGS
        start_iteration();
#endif
        lud_omp(m, matrix_dim);
#ifdef LOGS
        end_iteration();
#endif
#ifdef TIMING
        kernel_end = timing_get_time();
#endif

#ifdef TIMING
        check_start = timing_get_time();
#endif
        int i, host_errors = 0;
        #pragma omp parallel for reduction(+:host_errors)
        for ( i = 0; i < matrix_dim; i++) {
            int j;
            for ( j = 0; j < matrix_dim; j++) {
                if ((fabs((m[i + matrix_dim * j] - gold[i + matrix_dim * j]) / m[i + matrix_dim * j]) > 0.0000000001) || (fabs((m[i + matrix_dim * j] - gold[i + matrix_dim * j]) / gold[i + matrix_dim * j]) > 0.0000000001)) {
                //if (m[i + matrix_dim * j] != gold[i + matrix_dim * j]) {
                    char error_detail[200];
                    sprintf(error_detail," p: [%d, %d], r: %1.16e, e: %1.16e", i, j, m[i + matrix_dim * j], gold[i + matrix_dim * j]);
                    host_errors++;
#ifdef LOGS
                    log_error_detail(error_detail);
#endif

                }
            }
        }
#ifdef TIMING
        check_end = timing_get_time();
#endif
        if (host_errors > 0 ) {
            printf("Errors: %d\n",host_errors);
        } else {
            printf(".");
        }

#ifdef LOGS
        log_error_count(host_errors);
#endif
        // read inputs again
        {
            int n = matrix_dim;
            FILE *f_a, *f_gold;
            f_a = fopen(input_file, "rb");
            f_gold = fopen(gold_file, "rb");

            if ((f_a == NULL) || (f_gold == NULL)) {
                printf("Error opening files\n");
                exit(EXIT_FAILURE);
            }
            fread(m, sizeof(float) * n * n, 1, f_a);
            fread(gold, sizeof(float) * n * n, 1, f_gold);
            fclose(f_a);
            fclose(f_gold);
        }
#ifdef TIMING
        loop_end = timing_get_time();
        double setup_timing = (double) (setup_end - setup_start) / 1000000;
        double loop_timing = (double) (loop_end - loop_start) / 1000000;
        double kernel_timing = (double) (kernel_end - kernel_start) / 1000000;
        double check_timing = (double) (check_end - check_start) / 1000000;
        printf("\n\tTIMING:\n");
        printf("setup: %f\n",setup_timing);
        printf("loop: %f\n",loop_timing);
        printf("kernel: %f\n",kernel_timing);
        printf("check: %f\n",check_timing);
#endif

    }


#ifdef LOGS
    end_log_file();
#endif

    free(m);
    free(gold);

    return EXIT_SUCCESS;
}				/* ----------  end of function main  ---------- */
