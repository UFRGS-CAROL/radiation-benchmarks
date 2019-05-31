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

#include "common.h"

static int do_verify = 0;
int omp_num_threads = 228;

static struct option long_options[] = {
    /* name, has_arg, flag, val */
    {"size", 1, NULL, 's'},
    {0,0,0,0}
};

extern void
lud_omp(float *m, int matrix_dim);

int
main ( int argc, char *argv[] )
{
    int matrix_dim = 32; /* default size */
    int opt, option_index=0;
    func_ret_t ret;
    const char *input_file = NULL;
    float *m, *mm;
    stopwatch sw;


    while ((opt = getopt_long(argc, argv, "::s:n:",
                              long_options, &option_index)) != -1 ) {
        switch(opt) {
        case 'n':
            omp_num_threads = atoi(optarg);
            break;
        case 's':
            matrix_dim = atoi(optarg);
            printf("Generate input matrix internally, size =%d\n", matrix_dim);
            break;
        case '?':
            fprintf(stderr, "invalid option\n");
            break;
        case ':':
            fprintf(stderr, "missing argument\n");
            break;
        default:
            fprintf(stderr, "Usage: %s [-s matrix_size] [-n no. of threads]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if ( (optind < argc) || (optind == 1)) {
        fprintf(stderr, "Usage: %s [-s matrix_size] [-n no. of threads]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    if (input_file) {
        printf("Reading matrix from file %s\n", input_file);
        ret = create_matrix_from_file(&m, input_file, &matrix_dim);
        if (ret != RET_SUCCESS) {
            m = NULL;
            fprintf(stderr, "error create matrix from file %s\n", input_file);
            exit(EXIT_FAILURE);
        }
    }
    else if (matrix_dim) {
        printf("Creating matrix internally size=%d\n", matrix_dim);
        ret = create_matrix(&m, matrix_dim);
        if (ret != RET_SUCCESS) {
            m = NULL;
            fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
            exit(EXIT_FAILURE);
        }
    }

    else {
        printf("No input file specified!\n");
        exit(EXIT_FAILURE);
    }

    FILE *fpo;
    char input_name[500];
    snprintf(input_name, 500, "input_%d_th_%d",matrix_dim, omp_num_threads);
    fpo = fopen(input_name, "wb");
    fwrite(m, sizeof(float) * matrix_dim * matrix_dim, 1, fpo);
    fclose(fpo);

    stopwatch_start(&sw);
    lud_omp(m, matrix_dim);
    stopwatch_stop(&sw);
    printf("Time consumed(ms): %lf\n", 1000*get_interval_by_sec(&sw));

    char output_name[500];
    snprintf(output_name, 500, "gold_%d_th_%d",matrix_dim, omp_num_threads);
    fpo = fopen(output_name, "wb");
    fwrite(m, sizeof(float) * matrix_dim * matrix_dim, 1, fpo);
    fclose(fpo);
    int i, zeros=0;
    #pragma omp parallel for reduction(+:zeros)
    for(i = 0; i< matrix_dim*matrix_dim; i++) {
        if(m[i] == 0)
            zeros++;
    }
    printf("# of zeros: %d\n",zeros);

    free(m);

    return EXIT_SUCCESS;
}				/* ----------  end of function main  ---------- */
