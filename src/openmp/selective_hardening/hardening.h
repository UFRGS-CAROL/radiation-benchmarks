#ifndef HARDENING_H
#define HARDENING_H

#include "../../include/log_helper.h"
#include "xor_asm.h"

//#define HARDENING_DEBUG
//#define READ_HARDENED_VAR(VAR_NAME_1, VAR_NAME_2, VAR_TYPE, VAR_SIZE, VAR_NAME) (*((VAR_TYPE*)hardened_compare_and_return((void*)(&VAR_NAME_1), (void*)(&VAR_NAME_2), VAR_SIZE, __FILE__, __LINE__, VAR_NAME)))
//#define READ_HARDENED_ARRAY(POSITION, ARRAY_NAME_1, ARRAY_NAME_2, ARRAY_TYPE, ARRAY_SIZE) ((ARRAY_TYPE)((void*)hardened_compare_and_return_array(POSITION, (void*)(&ARRAY_NAME_1), (void*)(&ARRAY_NAME_2), ARRAY_SIZE)))
#define READ_HARDENED_VAR_INT(VAR_NAME_1, VAR_NAME_2, VAR_NAME) hardened_compare_and_return_int(VAR_NAME_1, VAR_NAME_2, __FILE__, __LINE__, VAR_NAME)
#define READ_HARDENED_VAR_LONG(VAR_NAME_1, VAR_NAME_2, VAR_NAME) hardened_compare_and_return_long(VAR_NAME_1, VAR_NAME_2, __FILE__, __LINE__, VAR_NAME)
#define READ_HARDENED_VAR_FLOAT(VAR_NAME_1, VAR_NAME_2, VAR_NAME) hardened_compare_and_return_float(VAR_NAME_1, VAR_NAME_2, __FILE__, __LINE__, VAR_NAME)
#define READ_HARDENED_VAR_DOUBLE(VAR_NAME_1, VAR_NAME_2, VAR_NAME) hardened_compare_and_return_double(VAR_NAME_1, VAR_NAME_2, __FILE__, __LINE__, VAR_NAME)
#define READ_HARDENED_VAR_PTR(VAR_NAME_1, VAR_NAME_2, VAR_NAME) (void*)hardened_compare_and_return_ptr((void*)VAR_NAME_1, (void*)VAR_NAME_2, __FILE__, __LINE__, VAR_NAME)

//#define READ_HARDENED_STRUCT_DOUBLE(VAR_NAME_1, VAR_NAME_2, VAR_TYPE, VAR_NAME) (*((VAR_TYPE*)hardened_compare_and_return_double_array((void*)&VAR_NAME_1, (void*)&VAR_NAME_2, sizeof(VAR_TYPE), __FILE__, __LINE__, VAR_NAME)))
//#define READ_HARDENED_ARRAY_DOUBLE(VAR_NAME_1, VAR_NAME_2, VAR_TYPE, VAR_NAME) (VAR_TYPE*)hardened_compare_and_return_double_array((void*)VAR_NAME_1, (void*)VAR_NAME_2, sizeof(VAR_TYPE), __FILE__, __LINE__, VAR_NAME)

void get_bits_str(char* dest_buffer, void* value, long long size);
void dump_error_info(void* var_a, void* var_b, long long size, char* file, long line, char* var_name);

int hardened_compare_and_return_int(int var_a, int var_b, char* file, long line, char* var_name);
long hardened_compare_and_return_long(long var_a, long var_b, char* file, long line, char* var_name);
float hardened_compare_and_return_float(float var_a, float var_b, char* file, long line, char* var_name);
double hardened_compare_and_return_double(double var_a, double var_b, char* file, long line, char* var_name);
void* hardened_compare_and_return_ptr(void* var_a, void* var_b, char* file, long line, char* var_name);
//void* hardened_compare_and_return_double_struct(void* var_a, void* var_b, int size, char* file, long line, char* var_name);
//void* hardened_compare_and_return_double_array(void* var_a, void* var_b, int size, char* file, long line, char* var_name);

//void* hardened_compare_and_return(void* var_a, void* var_b, long long size, char* file, long line, char* var_name);

#endif
