#include <stdio.h>
#include <stdlib.h>
#include "../include/selective_hardening.h"

int throw_error_int(int var1, int var2, const char* variable_name)
{
	if(var1 != var2)
	{
		printf("FATAL ERROR\n");
	
		FILE* error_file = fopen("/tmp/quicksort/ERROR_FILE.txt", "a");
		
		fprintf(error_file, "Fatal error detected:\n");
		fprintf(error_file, "\tVariable: %s\n", variable_name);
		fprintf(error_file, "\t1st value: %d\n", var1);
		fprintf(error_file, "\t2nd value: %d\n", var2);
	
		fclose(error_file);

		exit(-1);
	}

	return var1;
}
