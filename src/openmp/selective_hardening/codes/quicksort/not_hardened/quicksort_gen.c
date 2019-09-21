#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

void qsort_parallel(unsigned *a,int l,int r){
	if(r>l){

		int pivot=a[r],tmp;
		int less=l-1,more;
		for(more=l;more<=r;more++){
			if(a[more]<=pivot){
				less++;
				tmp=a[less];
				a[less]=a[more];
				a[more]=tmp;
			}
		}
			#pragma omp task
			qsort_parallel(a, l,less-1);
			#pragma omp task
			qsort_parallel(a, less+1,r);
			#pragma omp taskwait
	}
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
    unsigned *data;

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

    readInput(data, inputFile, size);

    printf("Executing Sorting...\n");

	#pragma omp parallel	
	#pragma omp single
	qsort_parallel(data, 0,size-1);
    printf("\nDone\n");
    //for(int i=0;i<size;i++) printf("%u ",data[i]);
    //printf("\n");
    int err=0;
    for(int i=1;i<size;i++) if(data[i-1] > data[i]) err++;
    printf("sorted err: %d\n",err);

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


    return 0;
}
