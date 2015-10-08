#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
//#include <cassert>
//#include <iostream>
//#include <sstream>
//#include <fstream>
//#include <stdlib.h>
#include <stdio.h>

void GenerateInputMatrices(int input_size)
{
    int i, j;
    FILE *f_A, *f_B;

    char a_matrix[150];
    char b_matrix[150];
    snprintf(a_matrix, 150, "Double_A_%d.matrix",input_size);
    snprintf(b_matrix, 150, "Double_B_%d.matrix",input_size);
    printf("Generating matrices:\n\t%s\n\t%s\n",a_matrix,b_matrix);

    f_A = fopen(a_matrix, "wb");
    f_B = fopen(b_matrix, "wb");

    if(!f_A || !f_B) {
        printf("ERROR: Could not open files\n");
        exit(1);
    }

    srand ( time(NULL) );

    double value;
    for(i=0; i<input_size; i++)
    {
        for(j=0; j<input_size; j++) {
            value= (rand()/((double)(RAND_MAX)+1)*(-4.06e16-4.0004e16))+4.1e16;
            fwrite( &value, sizeof(double), 1, f_A );

            value= (rand()/((double)(RAND_MAX)+1)*(-4.06e16-4.4e16))+4.1e16;
            fwrite( &value, sizeof(double), 1, f_B );
        }
    }

    fclose(f_A);
    fclose(f_B);
    printf("Matrices generated\n");
}


void usage() {
    printf("Usage: generateMatrices <input_size> \n");
}

int main(int argc, char ** argv)
{
    int input_size;
    if(argc == 2) {
        input_size = atoi(argv[1]);
    } else {
        usage();
        exit(1);
    }

    GenerateInputMatrices(input_size);

}


