#ifndef OPENCL_HELPER_LIBRARY_H
#define OPENCL_HELPER_LIBRARY_H

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>

#include <stdio.h>
#include <sys/time.h>


// Function prototypes
char *load_kernel_source(const char *filename);
long long get_time_helper();
void fatal(const char *s);
void fatal_CL(cl_int error, int line_no);


#endif
