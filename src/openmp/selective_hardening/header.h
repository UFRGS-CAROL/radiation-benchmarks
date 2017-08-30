#include "../../include/log_helper.h"

//#define HARDENING_DEBUG
#define READ_HARDENED_VAR(VAR_NAME_1, VAR_NAME_2, VAR_TYPE, VAR_SIZE, VAR_NAME) (*((VAR_TYPE*)hardened_compare_and_return((void*)(&VAR_NAME_1), (void*)(&VAR_NAME_2), VAR_SIZE, __FILE__, __LINE__, VAR_NAME)))
#define READ_HARDENED_ARRAY(POSITION, ARRAY_NAME_1, ARRAY_NAME_2, ARRAY_TYPE, ARRAY_SIZE) ((ARRAY_TYPE)((void*)hardened_compare_and_return_array(POSITION, (void*)(&ARRAY_NAME_1), (void*)(&ARRAY_NAME_2), ARRAY_SIZE)))

static int error_occured = 0;

void get_bits_str(char* dest_buffer, void* value, long long size);

inline void* hardened_compare_and_return(void* var_a, void* var_b, long long size, char* file, long line, char* var_name)
{
        if(memcmp(var_a, var_b, size) != 0)
        {
#pragma omp critical(c1)
		{
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

void get_bits_str(char* dest_buffer, void* value, long long size)
{
	char* char_array = (char*)value;
		
	int temp_buffer[1024];

	int i, j;
	for(i = 0; i < size; i++)
	{
		for(j = 0; j < 8; j++)
		{	
			temp_buffer[(size*8 - 1) - (i*8 + j)] = ((int)char_array[i] & (int)pow(2, j)) >> j;
		}
	}

	for(i = 0; i < size * 8; i++)
	{
		sprintf(dest_buffer + i, "%d", temp_buffer[i]);
	}
}
