#define LIMIT -999
//#define TRACE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <unistd.h>

#ifdef LOGS
#include "../../include/log_helper.h"
#endif

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

#define OPENMP

#define BLOCK_SIZE 16

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);


void ReadArrayFromFile(int* input_itemsets, int max_rows, char * filenameinput) {
    int n = max_rows;

    FILE *f_a;
    f_a = fopen(filenameinput, "rb");

    if (f_a == NULL) {
        printf("Error opening INPUT files\n");
        exit(-3);
    }

    fread(input_itemsets, sizeof(int) * n * n, 1, f_a);
    fclose(f_a);
}

void ReadGoldFromFile(int* gold_itemsets, int max_rows, char * filenamegold) {
    int n = max_rows;

    FILE *f_gold;
    f_gold = fopen(filenamegold, "rb");

    if (f_gold == NULL) {
        printf("Error opening GOLD file\n");
        exit(-3);
    }

    fread(gold_itemsets, sizeof(int) * n * n, 1, f_gold);
    fclose(f_gold);
}

#ifdef OMP_OFFLOAD
#pragma omp declare target
#endif
int maximum( int a,
             int b,
             int c) {

    int k;
    if( a <= b )
        k = b;
    else
        k = a;

    if( k <=c )
        return(c);
    else
        return(k);
}
#ifdef OMP_OFFLOAD
#pragma omp end declare target
#endif


int blosum62[24][24] = {
    { 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
    {-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
    {-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
    {-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
    { 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
    {-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
    {-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
    { 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
    {-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
    {-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
    {-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
    {-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
    {-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
    {-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
    {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
    { 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
    { 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
    {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
    {-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
    { 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
    {-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
    {-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
    { 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
    {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv)
{
    runTest( argc, argv);

    return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
    fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> <num_threads> <input_array> <gold_array> <#iteractions>\n", argv[0]);
    fprintf(stderr, "\t<dimension>      - x and y dimensions\n");
    fprintf(stderr, "\t<penalty>        - penalty(positive integer)\n");
    fprintf(stderr, "\t<num_threads>    - no. of threads\n");
    exit(1);
}

void nw_optimized(int *input_itemsets, int *referrence,
                  int max_rows, int max_cols, int penalty)
{
#ifdef OMP_OFFLOAD
    int transfer_size = max_rows * max_cols;
    #pragma omp target data map(to: max_cols, penalty, referrence[0:transfer_size]) map(input_itemsets[0:transfer_size])
    {

        #pragma omp target
#endif
        for( int blk = 1; blk <= (max_cols-1)/BLOCK_SIZE; blk++ )
        {
#ifdef OPENMP
            #pragma omp parallel for schedule(static) shared(input_itemsets, referrence) firstprivate(blk, max_rows, max_cols, penalty)
#endif
            for( int b_index_x = 0; b_index_x < blk; ++b_index_x)
            {
                int b_index_y = blk - 1 - b_index_x;
                int input_itemsets_l[(BLOCK_SIZE + 1) *(BLOCK_SIZE+1)] __attribute__ ((aligned (64)));
                int reference_l[BLOCK_SIZE * BLOCK_SIZE] __attribute__ ((aligned (64)));

                // Copy referrence to local memory
                for ( int i = 0; i < BLOCK_SIZE; ++i )
                {
                    #pragma omp simd
                    for ( int j = 0; j < BLOCK_SIZE; ++j)
                    {
                        reference_l[i*BLOCK_SIZE + j] = referrence[max_cols*(b_index_y*BLOCK_SIZE + i + 1) + b_index_x*BLOCK_SIZE +  j + 1];
                    }
                }

                // Copy input_itemsets to local memory
                for ( int i = 0; i < BLOCK_SIZE + 1; ++i )
                {
                    #pragma omp simd
                    for ( int j = 0; j < BLOCK_SIZE + 1; ++j)
                    {
                        input_itemsets_l[i*(BLOCK_SIZE + 1) + j] = input_itemsets[max_cols*(b_index_y*BLOCK_SIZE + i) + b_index_x*BLOCK_SIZE +  j];
                    }
                }

                // Compute
                for ( int i = 1; i < BLOCK_SIZE + 1; ++i )
                {
                    for ( int j = 1; j < BLOCK_SIZE + 1; ++j)
                    {
                        input_itemsets_l[i*(BLOCK_SIZE + 1) + j] = maximum( input_itemsets_l[(i - 1)*(BLOCK_SIZE + 1) + j - 1] + reference_l[(i - 1)*BLOCK_SIZE + j - 1],
                                input_itemsets_l[i*(BLOCK_SIZE + 1) + j - 1] - penalty,
                                input_itemsets_l[(i - 1)*(BLOCK_SIZE + 1) + j] - penalty);
                    }
                }

                // Copy results to global memory
                for ( int i = 0; i < BLOCK_SIZE; ++i )
                {
                    #pragma omp simd
                    for ( int j = 0; j < BLOCK_SIZE; ++j)
                    {
                        input_itemsets[max_cols*(b_index_y*BLOCK_SIZE + i + 1) + b_index_x*BLOCK_SIZE +  j + 1] = input_itemsets_l[(i + 1)*(BLOCK_SIZE+1) + j + 1];
                    }
                }

            }
        }

        //printf("Processing bottom-right matrix\n");

#ifdef OMP_OFFLOAD
        #pragma omp target
#endif
        for ( int blk = 2; blk <= (max_cols-1)/BLOCK_SIZE; blk++ )
        {
#ifdef OPENMP
            #pragma omp parallel for schedule(static) shared(input_itemsets, referrence) firstprivate(blk, max_rows, max_cols, penalty)
#endif
            for( int b_index_x = blk - 1; b_index_x < (max_cols-1)/BLOCK_SIZE; ++b_index_x)
            {
                int b_index_y = (max_cols-1)/BLOCK_SIZE + blk - 2 - b_index_x;

                int input_itemsets_l[(BLOCK_SIZE + 1) *(BLOCK_SIZE+1)] __attribute__ ((aligned (64)));
                int reference_l[BLOCK_SIZE * BLOCK_SIZE] __attribute__ ((aligned (64)));

                // Copy referrence to local memory
                for ( int i = 0; i < BLOCK_SIZE; ++i )
                {
                    #pragma omp simd
                    for ( int j = 0; j < BLOCK_SIZE; ++j)
                    {
                        reference_l[i*BLOCK_SIZE + j] = referrence[max_cols*(b_index_y*BLOCK_SIZE + i + 1) + b_index_x*BLOCK_SIZE +  j + 1];
                    }
                }

                // Copy input_itemsets to local memory
                for ( int i = 0; i < BLOCK_SIZE + 1; ++i )
                {
                    #pragma omp simd
                    for ( int j = 0; j < BLOCK_SIZE + 1; ++j)
                    {
                        input_itemsets_l[i*(BLOCK_SIZE + 1) + j] = input_itemsets[max_cols*(b_index_y*BLOCK_SIZE + i) + b_index_x*BLOCK_SIZE +  j];
                    }
                }

                // Compute
                for ( int i = 1; i < BLOCK_SIZE + 1; ++i )
                {
                    for ( int j = 1; j < BLOCK_SIZE + 1; ++j)
                    {
                        input_itemsets_l[i*(BLOCK_SIZE + 1) + j] = maximum( input_itemsets_l[(i - 1)*(BLOCK_SIZE + 1) + j - 1] + reference_l[(i - 1)*BLOCK_SIZE + j - 1],
                                input_itemsets_l[i*(BLOCK_SIZE + 1) + j - 1] - penalty,
                                input_itemsets_l[(i - 1)*(BLOCK_SIZE + 1) + j] - penalty);
                    }
                }

                // Copy results to global memory
                for ( int i = 0; i < BLOCK_SIZE; ++i )
                {
                    #pragma omp simd
                    for ( int j = 0; j < BLOCK_SIZE; ++j)
                    {
                        input_itemsets[max_cols*(b_index_y*BLOCK_SIZE + i + 1) + b_index_x*BLOCK_SIZE +  j + 1] = input_itemsets_l[(i + 1)*(BLOCK_SIZE+1) + j +1];
                    }
                }
            }
        }

#ifdef OMP_OFFLOAD
    }
#endif

}

int iteractions = 100000;
void
runTest( int argc, char** argv)
{

#ifdef TIMING
    setup_start = timing_get_time();
#endif
    int max_rows, max_cols, penalty;
    int *input_itemsets,  *referrence, *gold_itemsets;
    char * array_path, * gold_path;
    //int *matrix_cuda, *matrix_cuda_out, *referrence_cuda;
    //int size;
    int omp_num_threads;


    // the lengths of the two sequences should be able to divided by 16.
    // And at current stage  max_rows needs to equal max_cols
    if (argc == 7)
    {
        max_rows = atoi(argv[1]);
        max_cols = atoi(argv[1]);
        penalty = atoi(argv[2]);
        omp_num_threads = atoi(argv[3]);
        array_path = argv[4];
        gold_path =  argv[5];
        iteractions = atoi(argv[6]);
    }
    else {
        usage(argc, argv);
    }

    omp_set_num_threads(omp_num_threads);

#ifdef LOGS
    set_iter_interval_print(10);
    char test_info[200];
    snprintf(test_info, 200, "max_rows:%d max_cols:%d penalty:%d omp_num_threads:%d", max_rows, max_cols, penalty, omp_num_threads);
    start_log_file("openmpNW", test_info);
#endif
    max_rows = max_rows + 1;
    max_cols = max_cols + 1;
    referrence = (int *)malloc( max_rows * max_cols * sizeof(int) );
    input_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );

    gold_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );


    if (!input_itemsets)
        fprintf(stderr, "error: can not allocate memory");

    srand ( 7 );


    printf("Start Needleman-Wunsch\n");

    ReadArrayFromFile(input_itemsets, max_rows, array_path);
    ReadGoldFromFile(gold_itemsets, max_rows, gold_path);



    for (int i = 1 ; i < max_cols; i++) {
        for (int j = 1 ; j < max_rows; j++) {
            referrence[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
        }
    }

    for( int i = 1; i< max_rows ; i++)
        input_itemsets[i*max_cols] = -i * penalty;
    for( int j = 1; j< max_cols ; j++)
        input_itemsets[j] = -j * penalty;



    //Compute top-left matrix
    printf("Num of threads: %d\n", omp_num_threads);
    //printf("Processing top-left matrix\n");

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
            input_itemsets[0] = 102012;
            input_itemsets[10] = 102012;
            input_itemsets[55] = 102012;
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
        nw_optimized( input_itemsets, referrence,
                      max_rows, max_cols, penalty );
#ifdef LOGS
        end_iteration();
#endif
#ifdef TIMING
        kernel_end = timing_get_time();
#endif


#ifdef TIMING
        check_start = timing_get_time();
#endif
        int host_errors = 0;
        #pragma omp parallel for reduction(+:host_errors)
        for (int i = 0; i < max_rows; i++) {
            for (int j = 0; j < max_rows; j++) {
                if (input_itemsets[i + max_rows * j] != gold_itemsets[i + max_rows * j]) {
                    char error_detail[200];
                    sprintf(error_detail," p: [%d, %d], r: %i, e: %i", i, j, input_itemsets[i + max_rows * j], gold_itemsets[i + max_rows * j]);
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
            ReadArrayFromFile(input_itemsets, max_rows, array_path);
            #pragma omp parallel for
            for( int i = 1; i< max_rows ; i++)
                input_itemsets[i*max_cols] = -i * penalty;
            #pragma omp parallel for
            for( int j = 1; j< max_cols ; j++)
                input_itemsets[j] = -j * penalty;
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
    free(referrence);
    free(input_itemsets);

}



