#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>


void generateInputMatrices(const char *a_matrix_path, const char *b_matrix_path) {
	double temp;
	int i, j;
	FILE *f_A, *f_B;

	f_A = fopen(a_matrix_path, "wb");
	f_B = fopen(b_matrix_path, "wb");

	srand (time(NULL));

for(	i = 0; i < DEFAULT_INPUT_SIZE; i++) {
		for (j = 0; j < DEFAULT_INPUT_SIZE; j++) {
			temp = (rand() / ((double) (RAND_MAX) + 1) * (-4.06e16 - 4.0004e16))
			+ 4.1e16;
			fwrite(&temp, sizeof(double), 1, f_A);

			temp = (rand() / ((double) (RAND_MAX) + 1) * (-4.06e16 - 4.4e16))
			+ 4.1e16;
			fwrite(&temp, sizeof(double), 1, f_B);

		}
	}

	fclose(f_A);
	fclose(f_B);

	return;
}

int main(int argc, char** argv) {
	char *a_matrix_path, *b_matrix_path;

	if (argc == 3) {
		if (strcmp(argv[1], "") == 0 || strcmp(argv[2], "") == 0){
			printf("./%s <input path a> <input path b>\n", argv[0]);
			return -1;
		}
		a_matrix_path = argv[1];
		b_matrix_path = argv[2];
	} else {

		printf("Using default input paths\n");

		a_matrix_path = new char[100];
		snprintf(a_matrix_path, 100, "dgemm_a_%i",
				(signed int) DEFAULT_INPUT_SIZE);

		b_matrix_path = new char[100];
		snprintf(b_matrix_path, 100, "dgemm_b_%i",
				(signed int) DEFAULT_INPUT_SIZE);
	}

	printf("Matrix a: %s\nMatrix b: %s\n", a_matrix_path, b_matrix_path);

	printf("Generating input matrices...\n");
	generateInputMatrices(a_matrix_path, b_matrix_path);

	return 0;
}
