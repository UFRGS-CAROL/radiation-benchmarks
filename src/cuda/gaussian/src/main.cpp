
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "utils.h"

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
void create_matrix(float *m, int size) {
	int i, j;
	float lamda = -0.01;
	float coe[2 * size - 1];
	float coe_i = 0.0;

	for (i = 0; i < size; i++) {
		coe_i = 10 * exp(lamda * i);
		j = size - 1 + i;
		coe[j] = coe_i;
		j = size - 1 - i;
		coe[j] = coe_i;
	}

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			m[i * size + j] = coe[size - 1 - i + j];
		}
	}

}

int main(int argc, char *argv[]) {
	printf("WG size of kernel 1 = %d, WG size of kernel 2= %d X %d\n",
			MAXBLOCKSIZE, BLOCK_SIZE_XY, BLOCK_SIZE_XY);
	int verbose = 1;
	int i, j;
	char flag;
	if (argc < 2) {
		printf("Usage: gaussian -f filename / -s size [-q]\n\n");
		printf(
				"-q (quiet) suppresses printing the matrix and result values.\n");
		printf("-f (filename) path of input file\n");
		printf(
				"-s (size) size of matrix. Create matrix and rhs in this program \n");
		printf(
				"The first line of the file contains the dimension of the matrix, n.");
		printf("The second line of the file is a newline.\n");
		printf(
				"The next n lines contain n tab separated values for the matrix.");
		printf("The next line of the file is a newline.\n");
		printf(
				"The next line of the file is a 1xn vector with tab separated values.\n");
		printf("The next line of the file is a newline. (optional)\n");
		printf(
				"The final line of the file is the pre-computed solution. (optional)\n");
		printf("Example: matrix4.txt:\n");
		printf("4\n");
		printf("\n");
		printf("-0.6	-0.5	0.7	0.3\n");
		printf("-0.3	-0.9	0.3	0.7\n");
		printf("-0.4	-0.5	-0.3	-0.8\n");
		printf("0.0	-0.1	0.2	0.9\n");
		printf("\n");
		printf("-0.85	-0.68	0.24	-0.53\n");
		printf("\n");
		printf("0.7	0.0	-0.4	-0.5\n");
		exit(0);
	}

	//PrintDeviceProperties();
	//char filename[100];
	//sprintf(filename,"matrices/matrix%d.txt",size);

	for (i = 1; i < argc; i++) {
		if (argv[i][0] == '-') {    // flag
			flag = argv[i][1];
			switch (flag) {
			case 's': // platform
				i++;
				Size = atoi(argv[i]);
				printf("Create matrix internally in parse, size = %d \n", Size);

				a = (float *) malloc(Size * Size * sizeof(float));
				create_matrix(a, Size);

				b = (float *) malloc(Size * sizeof(float));
				for (j = 0; j < Size; j++)
					b[j] = 1.0;

				m = (float *) malloc(Size * Size * sizeof(float));
				break;
			case 'f': // platform
				i++;
				printf("Read file from %s \n", argv[i]);
				InitProblemOnce(argv[i]);
				break;
			case 'q': // quiet
				verbose = 0;
				break;
			}
		}
	}

	//InitProblemOnce(filename);
	InitPerRun();
	//begin timing
	struct timeval time_start;
	gettimeofday(&time_start, NULL);

	// run kernels
	ForwardSub();

	//end timing
	struct timeval time_end;
	gettimeofday(&time_end, NULL);
	unsigned int time_total = (time_end.tv_sec * 1000000 + time_end.tv_usec)
			- (time_start.tv_sec * 1000000 + time_start.tv_usec);

	if (verbose) {
		printf("Matrix m is: \n");
		PrintMat(m, Size, Size);

		printf("Matrix a is: \n");
		PrintMat(a, Size, Size);

		printf("Array b is: \n");
		PrintAry(b, Size);
	}
	BackSub();
	if (verbose) {
		printf("The final solution is: \n");
		PrintAry(finalVec, Size);
	}
	printf("\nTime total (including memory transfers)\t%f sec\n",
			time_total * 1e-6);
	printf("Time for CUDA kernels:\t%f sec\n", totalKernelTime * 1e-6);

	/*printf("%d,%d\n",size,time_total);
	 fprintf(stderr,"%d,%d\n",size,time_total);*/

	free(m);
	free(a);
	free(b);
}
