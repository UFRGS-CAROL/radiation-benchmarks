#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string>
#include <sys/time.h>

#define MATRIX_PATH "./Double_"
#define DEFAULT_INPUT_SIZE 8192

double *A, *B;

char *a_matrix_path, *b_matrix_path;


void generateInputMatrices()
{
	double temp;
	int i, j;
	FILE *f_A, *f_B;

	f_A = fopen(a_matrix_path, "wb");
	f_B = fopen(b_matrix_path, "wb");


	srand ( time(NULL) );

	for(i=0; i<DEFAULT_INPUT_SIZE; i++)
	{
		for(j=0; j<DEFAULT_INPUT_SIZE; j++){
			temp = (rand()/((double)(RAND_MAX)+1)*(-4.06e16-4.0004e16))+4.1e16;
			fwrite( &temp, sizeof(double), 1, f_A );
		

			temp = (rand()/((double)(RAND_MAX)+1)*(-4.06e16-4.4e16))+4.1e16;
			fwrite( &temp, sizeof(double), 1, f_B );
			
			
		}
	}

	fclose(f_A);
	fclose(f_B);

	return;
}


int main (int argc, char** argv)
{
        a_matrix_path = new char[100];
        snprintf(a_matrix_path, 100, "dgemm_a_%i", (signed int)DEFAULT_INPUT_SIZE);
        printf("Using default input_a path: %s\n", a_matrix_path);

        b_matrix_path = new char[100];
        snprintf(b_matrix_path, 100, "dgemm_b_%i", (signed int)DEFAULT_INPUT_SIZE);
        printf("Using default input_a path: %s\n", b_matrix_path);

	
	printf("Generating input matrices...\n");
	generateInputMatrices();


	return 0;
}
