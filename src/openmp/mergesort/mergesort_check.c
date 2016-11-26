#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include "omp.h"

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


void merge(unsigned * X, int l, int r, unsigned * tmp) {
    int i = l;
    int mid = (l+r)/2;
    int j = mid + 1;
    int ti = l;

    while (i<=mid && j<=r) {
        if (X[i] < X[j]) {
            tmp[ti] = X[i];
            ti++;
            i++;
        } else {
            tmp[ti] = X[j];
            ti++;
            j++;
        }
    }
    while (i<=mid) { /* finish up lower half */
        tmp[ti] = X[i];
        ti++;
        i++;
    }
    while (j<=r) { /* finish up upper half */
        tmp[ti] = X[j];
        ti++;
        j++;
    }

    for(i=l; i<=r; i++) {
        X[i]=tmp[i];
    }

}

void mergesort(unsigned * X, int l, int r, unsigned * tmp)
{
    if (l >= r) return;
    int mid = (l+r)/2;
    #pragma omp task firstprivate (X, l, mid, tmp)
    mergesort(X, l, mid, tmp);
    #pragma omp task firstprivate (X, mid, r, tmp)
    mergesort(X, mid+1, r, tmp);

    #pragma omp taskwait
    merge(X, l, r, tmp);
}

void readFileUnsigned(unsigned *input, char *filename, int size) {
    FILE *finput;
    if (finput = fopen(filename, "rb")) {
        fread(input, size * sizeof(unsigned), 1 , finput);
    } else {
        printf("Error reading input file");
        exit(1);
    }
}


int main(int argc, char** argv)
{
#ifdef TIMING
    setup_start = timing_get_time();
#endif
    int size, omp_num_threads, iterations;
    char * inputFile, *goldFile;
    unsigned *data, *tmp, *gold;

    if (argc == 6) {
        size = atoi(argv[1]);
        omp_num_threads = atoi(argv[2]);
        inputFile = argv[3];
        goldFile = argv[4];
        iterations = atoi(argv[5]);
    } else {
        fprintf(stderr, "Usage: %s <input size> <num_threads> <input file> <gold file> <#iterations>\n", argv[0]);
        exit(1);
    }

    omp_set_num_threads(omp_num_threads);

#ifdef LOGS
    set_iter_interval_print(10);
    char test_info[200];
    snprintf(test_info, 200, "size:%d omp_num_threads:%d", size, omp_num_threads);
    start_log_file("openmpMergesort", test_info);
#endif
    data = (unsigned *)malloc(size*sizeof(unsigned));
    tmp = (unsigned *)malloc(size*sizeof(unsigned));
    gold = (unsigned *)malloc(size*sizeof(unsigned));

    readFileUnsigned(data, inputFile, size);
    readFileUnsigned(gold, goldFile, size);

#ifdef TIMING
    setup_end = timing_get_time();
#endif
    int loop;
    for(loop=0; loop<iterations; loop++) {
#ifdef TIMING
        loop_start = timing_get_time();
#endif
#ifdef ERR_INJ
        if(loop == 2) {
            printf("injecting error, changing input!\n");
            data[0] = 102012;
            data[10] = 1012;
            data[11] = 1012;
            data[12] = 1012;
            data[55] = 102000012;
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
        #pragma omp parallel
        #pragma omp single
        mergesort(data, 0,size-1, tmp);

#ifdef LOGS
        end_iteration();
#endif
#ifdef TIMING
        kernel_end = timing_get_time();
#endif

#ifdef TIMING
        check_start = timing_get_time();
#endif
        int errors=0;
        int i;
        #pragma omp parallel for reduction(+:errors) private(i)
        for(i=0; i< size; i++) {
            if(data[i] != gold[i]) {
                errors++;
                char error_detail[200];
                sprintf(error_detail," p: [%d], r: %u, e: %u", i, data[i], gold[i]);
#ifdef LOGS
                log_error_detail(error_detail);
#endif
            }
        }
#ifdef LOGS
        log_error_count(errors);
#endif
        if(errors > 0) {
            printf("Errors: %d\n",errors);
            readFileUnsigned(gold, goldFile, size);
        } else {
            printf(".");
        }
        readFileUnsigned(data, inputFile, size);
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

    return 0;
}

