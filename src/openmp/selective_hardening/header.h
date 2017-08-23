#include "../../include/log_helper.h"

//#define HARDENING_DEBUG
#define READ_HARDENED_VAR(VAR_NAME_1, VAR_NAME_2, VAR_TYPE, VAR_SIZE, VAR_NAME) (*((VAR_TYPE*)hardened_compare_and_return((void*)(&VAR_NAME_1), (void*)(&VAR_NAME_2), VAR_SIZE, __FILE__, __LINE__, VAR_NAME)))
#define READ_HARDENED_ARRAY(POSITION, ARRAY_NAME_1, ARRAY_NAME_2, ARRAY_TYPE, ARRAY_SIZE) ((ARRAY_TYPE)((void*)hardened_compare_and_return_array(POSITION, (void*)(&ARRAY_NAME_1), (void*)(&ARRAY_NAME_2), ARRAY_SIZE)))

inline void* hardened_compare_and_return(void* var_a, void* var_b, long long size, char* file, long line, char* var_name)
{
        if(memcmp(var_a, var_b, size) != 0)
        {
	
#pragma omp critical(c1)
		{
                	printf("\nHardening error:\n");
			printf("\tfile: \"%s\"\n", file);
			printf("\tvariable: \"%s\"\n", var_name);
			printf("\tline %d\n\n", line);
			printf("\tvar_1 double value: %lf\n", *((double*)var_a));
			printf("\tvar_2 double value: %lf\n", *((double*)var_b));
			printf("\tvar_1 address: %p\n", var_a);
			printf("\tvar_2 address: %p\n", var_b);

#ifdef LOGS

			end_iteration();

			char error_details[500];
			sprintf(error_details, " file: [%s], var_name: [%s], line: [%d], var_1_d_val: [%lf], var_2_d_val: [%lf], var_1_addr: [%p], var_2_addr: [%p]", file, var_name, line, *((double*)var_a), *((double*)var_b), var_a, var_b);

			log_error_detail(error_details);

			log_error_count(1);
	
			end_log_file();
#endif
		
                	exit(1);
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

