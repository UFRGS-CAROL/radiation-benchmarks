#ifdef __cplusplus
extern "C" {
#endif


#include <stdio.h>					// (in library path known to compiler)		needed by printf

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>					// (in library path provided to compiler)	needed by OpenCL types


char *
load_kernel_source(const char *filename);


void
fatal(const char *s);


void
fatal_CL(cl_int error, int line_no);

#ifdef __cplusplus
}
#endif
