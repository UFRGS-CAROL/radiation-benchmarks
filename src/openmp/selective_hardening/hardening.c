#include "hardening.h"
#include <stdio.h>
#include <stdlib.h> 
#include <string.h>

static int error_occured = 0;
static int current_iteration = -1;

inline int hardened_compare_and_return_int(int var_a, int var_b, char* file, long line, char* var_name)
{
	int result;
	
	int a = var_a;
	int b = var_b;
	result = int_xor(a, b);
		
	if(result != 0)
	{
		dump_error_info(&a, &b, sizeof(int), file, line, var_name);
	}

	return a;

}

inline float hardened_compare_and_return_float(float var_a, float var_b, char* file, long line, char* var_name)
{
	int result;
	
	float a = var_a;
	float b = var_b;
	result = float_xor(a, b);
		
	if(result != 0)
	{
		dump_error_info(&a, &b, sizeof(float), file, line, var_name);
	}

	return a;

}

inline double hardened_compare_and_return_double(double var_a, double var_b, char* file, long line, char* var_name)
{
	long result;
	
	double a = var_a;
	double b = var_b;
	result = double_xor(a, b);
		
	if(result != 0)
	{
		dump_error_info(&a, &b, sizeof(double), file, line, var_name);
	}

	return a;
}

inline void* hardened_compare_and_return_ptr(void* var_a, void* var_b, char* file, long line, char* var_name)
{
	void* result;
	
	void* a = var_a;
	void* b = var_b;
	result = ptr_xor(a, b);
		
	if(result != 0)
	{
		dump_error_info(&a, &b, sizeof(void*), file, line, var_name);
	}

	return a;
}


/*
inline void* hardened_compare_and_return_double_array(void* var_a, void* var_b, int size, char* file, long line, char* var_name)
{
	long result;

	double* a = (double*)var_a;
	double* b = (double*)var_b;
	
	int num_elem = size / sizeof(double);
	int i;

	for(i = 0; i < num_elem; i++)
	{
		result = double_xor(a[i], b[i]);
		if(result != 0)
		{
			dump_error_info(&a[i], &b[i], sizeof(double), file, line, var_name);
		}
	} 

	return var_a;
}

void* hardened_compare_and_return_double_struct(void* var_a, void* var_b, int size, char* file, long line, char* var_name)
{
	long result;

	double* a = (double*)var_a;
	double* b = (double*)var_b;
	
	int num_elem = size / sizeof(double);
	int i;

	for(i = 0; i < num_elem; i++)
	{
		result = double_xor(a[i], b[i]);
		if(result != 0)
		{
			dump_error_info(&a[i], &b[i], sizeof(double), file, line, var_name);
		}
	} 

	return var_a;

}
*/

void dump_error_info(void* var_a, void* var_b, long long size, char* file, long line, char* var_name)
{
	//#pragma omp critical(c1)
		{
			if(current_iteration != get_iteration_number())
			{
				current_iteration = get_iteration_number();
				error_occured = 0;	
			}

			if(error_occured == 0)                	
			{
				error_occured = 1;

				printf("\nHardening error:\n");
				printf("\tfile: \"%s\"\n", file);
				printf("\tvariable: \"%s\"\n", var_name);
				printf("\tline %d\n", line);
				printf("\tsize %d bytes\n", size);

				char var_1_bits[1024];
				char var_2_bits[1024];
		
				get_bits_str(var_1_bits, var_a, size);
				get_bits_str(var_2_bits, var_b, size);
	
				printf("\tvar_1 bits: %s\n", var_1_bits);
				printf("\tvar_2 bits: %s\n", var_2_bits);

#ifdef LOGS
				//end_iteration();

				char error_details[500];
				sprintf(error_details, " file: [%s], var_name: [%s], line: [%d], size_bytes: [%d], var_1_bits: [%s], var_2_bits: [%s]", file, var_name, line, size, var_1_bits, var_2_bits);

				log_error_detail(error_details);

				//log_error_count(1);
		
				//end_log_file();
#endif
		
	                	//exit(1);
			}
		}

}

/*
inline void* hardened_compare_and_return(void* var_a, void* var_b, long long size, char* file, long line, char* var_name)
{
        if(memcmp(var_a, var_b, size) != 0)
        {
#pragma omp critical(c1)
		{
			if(current_iteration != get_iteration_number())
			{
				current_iteration = get_iteration_number();
				error_occured = 0;	
			}

			if(error_occured == 0)                	
			{
				error_occured = 1;

				printf("\nHardening error:\n");
				printf("\tfile: \"%s\"\n", file);
				printf("\tvariable: \"%s\"\n", var_name);
				printf("\tline %d\n", line);
				printf("\tsize %d bytes\n", size);

				char var_1_bits[1024];
				char var_2_bits[1024];
		
				get_bits_str(var_1_bits, var_a, size);
				get_bits_str(var_2_bits, var_b, size);
	
				printf("\tvar_1 bits: %s\n", var_1_bits);
				printf("\tvar_2 bits: %s\n", var_2_bits);
				printf("\tvar_1 address: %p\n", var_a);
				printf("\tvar_2 address: %p\n", var_b);

#ifdef LOGS
				//end_iteration();

				char error_details[500];
				sprintf(error_details, " file: [%s], var_name: [%s], line: [%d], size_bytes: [%d], var_1_bits: [%s], var_2_bits: [%s], var_1_addr: [%p], var_2_addr: [%p]", file, var_name, line, size, var_1_bits, var_2_bits, var_a, var_b);

				log_error_detail(error_details);

				//log_error_count(1);
		
				//end_log_file();
#endif
		
	                	//exit(1);
			}
		}
        }
        
	return var_a;
}

inline void* hardened_compare_and_return_array(void* array_ptr_a, void* array_ptr_b, long long size)
{
	char* bytes_array_a = (char*)((char**)array_ptr_a);
	char* bytes_array_b = (char*)((char**)array_ptr_b);

#ifdef HARDENING_DEBUG
	printf("hardening_array: array_ptr_1 = %p, array_ptr_2 = %p, array_size = %d\n", bytes_array_a, bytes_array_b, size);
#endif

        if(memcmp(bytes_array_a, bytes_array_b, size) != 0)
        {
                printf("\nHardening error: at file \"%s\"\n\n", __FILE__);
                exit(1);
        }

        return array_ptr_a;
}
*/

void get_bits_str(char* dest_buffer, void* value, long long size)
{
	char* char_array = (char*)value;
		
	int temp_buffer[1024];
	
	int i, j;
	for(i = 0; i < size; i++)
	{
		int power = 1;
		for(j = 0; j < 8; j++)
		{	
			temp_buffer[(size*8 - 1) - (i*8 + j)] = ((int)char_array[i] & power) >> j;
			power *= 2;
		}
	}

	for(i = 0; i < size * 8; i++)
	{
		sprintf(dest_buffer + i, "%d", temp_buffer[i]);
	}
}
