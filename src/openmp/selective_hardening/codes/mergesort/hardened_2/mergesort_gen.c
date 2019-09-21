#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>


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

void readInput(unsigned *input, char *filename, int size) {
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
    int size, omp_num_threads;
    char * inputFile;
    unsigned *data, *tmp, *test;

    if (argc == 4) {
        size = atoi(argv[1]);
        omp_num_threads = atoi(argv[2]);
        inputFile = argv[3];
    } else {
        fprintf(stderr, "Usage: %s <input size> <num_threads> <input file> \n", argv[0]);
        exit(1);
    }

    omp_set_num_threads(omp_num_threads);

    data = (unsigned *)malloc(size*sizeof(unsigned));
    tmp = (unsigned *)malloc(size*sizeof(unsigned));
    test = (unsigned *)malloc(size*sizeof(unsigned));

    readInput(data, inputFile, size);

    printf("Executing Sorting...\n");
    //for(int i=0; i<size; i++) printf("%u ",data[i]);

    #pragma omp parallel
    #pragma omp single
    mergesort(data, 0,size-1, tmp);

    printf("\nDone\n");
    //for(int i=0; i<size; i++) printf("%u ",data[i]);
    printf("\n");
    int err=0;
    for(int i=1; i<size; i++) if(data[i-1] > data[i]) err++;
    printf("err: %d\n",err);

    FILE *finput;
    char filename[100];
    snprintf(filename, 100, "gold_%d",size);
    if (finput = fopen(filename, "wb")) {
        fwrite(data, size * sizeof(unsigned), 1 , finput);
    } else {
        printf("Error writing gold file");
        exit(1);
    }
    fclose(finput);


    //readInput(test, filename, size);
    //int errors =0;
    //for(int i=0; i< size; i++) {
    //    if(data[i] != test[i]) {
    //        errors++;
    //    }
    //}
    //printf("Errors: %d",errors);

    return 0;
}

