
#include "support/common.h"
#include "support/timer.h"
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

// Params ---------------------------------------------------------------------
struct Params {

    int   n_reps;
    int   in_size;
    int   compaction_factor;
    int   remove_value;
    int   n_work_items;

    Params(int argc, char **argv) {
        n_work_items      = 256;
        n_reps            = 50;
        in_size           = 1048576;
        compaction_factor = 50;
        remove_value      = 0;
        int opt;
        while((opt = getopt(argc, argv, "hp:d:i:g:t:w:r:a:n:c:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'i': n_work_items      = atoi(optarg); break;
            case 'r': n_reps            = atoi(optarg); break;
            case 'n': in_size           = atoi(optarg); break;
            case 'c': compaction_factor = atoi(optarg); break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./sc [options]"

                "\n    -r <R>    # of timed repetition iterations (default=50)"
                "\n    -i <I>    # of device work-items (default=256)"
                "\nBenchmark-specific options:"
                "\n    -n <N>    input size (default=1048576)"
                "\n    -c <C>    compaction factor (default=50)"
                "\n");
    }
};

// Input Data -----------------------------------------------------------------
void new_read_input(T *input, const Params &p) {

    FILE *f = NULL;
    char filename[100];
    snprintf(filename, 100, "input_%d_%d_%d",p.in_size,p.n_work_items,p.compaction_factor); // Gold com a resolução 
    const int n_tasks     = divceil(p.in_size, p.n_work_items * REGS);
    int in_size   = n_tasks * p.n_work_items * REGS * sizeof(T);
    FILE *finput;
    if (finput = fopen(filename, "rb")) {
        fread(input, in_size, 1 , finput);
    } else {
        printf("Error reading input file\n");
        exit(1);
    }
	fclose(finput);	
}
void read_input(T *input, const Params &p) {

    // Initialize the host input vectors
    srand(15);			// Colocamos  a mesma semente para gerar a mesma Sequencia de Números 
    for(int i = 0; i < p.in_size; i++) {
        input[i] = (T)p.remove_value;
    }
    int M = (p.in_size * p.compaction_factor) / 100;
    int m = M;
    while(m > 0) {
        int x = (int)(p.in_size * (((float)rand() / (float)RAND_MAX)));
        if(x < p.in_size)
            if(input[x] == p.remove_value) {
                input[x] = (T)(x + 2);
                m--;
            }
    }
}
// Sequential implementation for comparison purposes
inline double cpu_streamcompaction(T *input, int size, int value) {
    struct timeval t1, t2;
    int            pos = 0;
    // start timer
    gettimeofday(&t1, NULL);
    for(int my = 0; my < size; my++) {
        if(input[my] != value) {
            input[pos] = input[my];
            pos++;
        }
    }
    // end timer
    gettimeofday(&t2, NULL);
    double timer = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (t2.tv_usec - t1.tv_usec);
    //printf("Execute time: %f us\n", timer);
    return timer;
}

int main(int argc, char **argv) {

    const Params p(argc, argv);
    const int n_tasks     = divceil(p.in_size, p.n_work_items * REGS);
	int in_size = n_tasks * p.n_work_items * REGS * sizeof(T);

    T *    h_in_out = (T *)malloc(n_tasks * p.n_work_items * REGS * sizeof(T));
    read_input(h_in_out, p);

    cpu_streamcompaction(h_in_out, p.in_size, p.remove_value);


	FILE *finput;
    char filename[100];
    snprintf(filename, 100, "gold_%d_%d_%d",p.in_size,p.n_work_items,p.compaction_factor); // Gold com a resolução 
    if (finput = fopen(filename, "wb")) {
        fwrite(h_in_out, in_size, 1 , finput);
    } else {
        printf("Error writing input file");
        exit(1);
    }
    fclose(finput);
    free(h_in_out);

}
