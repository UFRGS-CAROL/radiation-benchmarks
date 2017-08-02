//#define HARDENING_DEBUG
#define READ_HARDENED_VAR(VAR_NAME_1, VAR_NAME_2, VAR_TYPE, VAR_SIZE, VAR_NAME) (*((VAR_TYPE*)hardened_compare_and_return((void*)(&VAR_NAME_1), (void*)(&VAR_NAME_2), VAR_SIZE, __FILE__, __LINE__, VAR_NAME)))
#define READ_HARDENED_ARRAY(POSITION, ARRAY_NAME_1, ARRAY_NAME_2, ARRAY_TYPE, ARRAY_SIZE) ((ARRAY_TYPE)((void*)hardened_compare_and_return_array(POSITION, (void*)(&ARRAY_NAME_1), (void*)(&ARRAY_NAME_2), ARRAY_SIZE)))

inline void* hardened_compare_and_return(void* var_a, void* var_b, long long size, char* file, long line, char* var_name)
{
        if(memcmp(var_a, var_b, size) != 0)
        {
                printf("\nHardening error:\n");
		printf("\tfile: \"%s\"\n", file);
		printf("\tvariable: \"%s\"\n", var_name);
		printf("\tline %d\n\n", line);
                exit(1);
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

