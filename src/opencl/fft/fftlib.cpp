#include <stdio.h>
#include <assert.h>
#include <string>
#include <iostream>
#include <cfloat>
#include <sstream>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

/*
#include "OpenCLDeviceInfo.h"
#include "Event.h"
#include "ResultDatabase.h"
#include "support.h"
*/
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_ext.h>
#include "fftlib.h"

#include <map>

using namespace std;

#define CL_CHECK_ERROR(err) \
	{ \
		if (err != CL_SUCCESS) \
		std::cerr << "Error: " \
		<< CLErrorString(err) \
		<< " in " << __FILE__ \
		<< " line " << __LINE__ \
		<< std::endl; \
	}

inline const char *CLErrorString(cl_int err) {
    switch (err) {
        // break;
    case CL_SUCCESS:
        return "CL_SUCCESS";
        // break;
    case CL_DEVICE_NOT_FOUND:
        return "CL_DEVICE_NOT_FOUND";
        // break;
    case CL_DEVICE_NOT_AVAILABLE:
        return "CL_DEVICE_NOT_AVAILABLE";
        // break;
    case CL_COMPILER_NOT_AVAILABLE:
        return "CL_COMPILER_NOT_AVAILABLE";
        // break;
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        // break;
    case CL_OUT_OF_RESOURCES:
        return "CL_OUT_OF_RESOURCES";
        // break;
    case CL_OUT_OF_HOST_MEMORY:
        return "CL_OUT_OF_HOST_MEMORY";
        // break;
    case CL_PROFILING_INFO_NOT_AVAILABLE:
        return "CL_PROFILING_INFO_NOT_AVAILABLE";
        // break;
    case CL_MEM_COPY_OVERLAP:
        return "CL_MEM_COPY_OVERLAP";
        // break;
    case CL_IMAGE_FORMAT_MISMATCH:
        return "CL_IMAGE_FORMAT_MISMATCH";
        // break;
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        // break;
    case CL_BUILD_PROGRAM_FAILURE:
        return "CL_BUILD_PROGRAM_FAILURE";
        // break;
    case CL_MAP_FAILURE:
        return "CL_MAP_FAILURE";
        // break;
    case CL_INVALID_VALUE:
        return "CL_INVALID_VALUE";
        // break;
    case CL_INVALID_DEVICE_TYPE:
        return "CL_INVALID_DEVICE_TYPE";
        // break;
    case CL_INVALID_PLATFORM:
        return "CL_INVALID_PLATFORM";
        // break;
    case CL_INVALID_DEVICE:
        return "CL_INVALID_DEVICE";
        // break;
    case CL_INVALID_CONTEXT:
        return "CL_INVALID_CONTEXT";
        // break;
    case CL_INVALID_QUEUE_PROPERTIES:
        return "CL_INVALID_QUEUE_PROPERTIES";
        // break;
    case CL_INVALID_COMMAND_QUEUE:
        return "CL_INVALID_COMMAND_QUEUE";
        // break;
    case CL_INVALID_HOST_PTR:
        return "CL_INVALID_HOST_PTR";
        // break;
    case CL_INVALID_MEM_OBJECT:
        return "CL_INVALID_MEM_OBJECT";
        // break;
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        // break;
    case CL_INVALID_IMAGE_SIZE:
        return "CL_INVALID_IMAGE_SIZE";
        // break;
    case CL_INVALID_SAMPLER:
        return "CL_INVALID_SAMPLER";
        // break;
    case CL_INVALID_BINARY:
        return "CL_INVALID_BINARY";
        // break;
    case CL_INVALID_BUILD_OPTIONS:
        return "CL_INVALID_BUILD_OPTIONS";
        // break;
    case CL_INVALID_PROGRAM:
        return "CL_INVALID_PROGRAM";
        // break;
    case CL_INVALID_PROGRAM_EXECUTABLE:
        return "CL_INVALID_PROGRAM_EXECUTABLE";
        // break;
    case CL_INVALID_KERNEL_NAME:
        return "CL_INVALID_KERNEL_NAME";
        // break;
    case CL_INVALID_KERNEL_DEFINITION:
        return "CL_INVALID_KERNEL_DEFINITION";
        // break;
    case CL_INVALID_KERNEL:
        return "CL_INVALID_KERNEL";
        // break;
    case CL_INVALID_ARG_INDEX:
        return "CL_INVALID_ARG_INDEX";
        // break;
    case CL_INVALID_ARG_VALUE:
        return "CL_INVALID_ARG_VALUE";
        // break;
    case CL_INVALID_ARG_SIZE:
        return "CL_INVALID_ARG_SIZE";
        // break;
    case CL_INVALID_KERNEL_ARGS:
        return "CL_INVALID_KERNEL_ARGS";
        // break;
    case CL_INVALID_WORK_DIMENSION:
        return "CL_INVALID_WORK_DIMENSION";
        // break;
    case CL_INVALID_WORK_GROUP_SIZE:
        return "CL_INVALID_WORK_GROUP_SIZE";
        // break;
    case CL_INVALID_WORK_ITEM_SIZE:
        return "CL_INVALID_WORK_ITEM_SIZE";
        // break;
    case CL_INVALID_GLOBAL_OFFSET:
        return "CL_INVALID_GLOBAL_OFFSET";
        // break;
    case CL_INVALID_EVENT_WAIT_LIST:
        return "CL_INVALID_EVENT_WAIT_LIST";
        // break;
    case CL_INVALID_EVENT:
        return "CL_INVALID_EVENT";
        // break;
    case CL_INVALID_OPERATION:
        return "CL_INVALID_OPERATION";
        // break;
    case CL_INVALID_GL_OBJECT:
        return "CL_INVALID_GL_OBJECT";
        // break;
    case CL_INVALID_BUFFER_SIZE:
        return "CL_INVALID_BUFFER_SIZE";
        // break;
    case CL_INVALID_MIP_LEVEL:
        return "CL_INVALID_MIP_LEVEL";
        // break;
    case CL_INVALID_GLOBAL_WORK_SIZE:
        return "CL_INVALID_GLOBAL_WORK_SIZE";
        // break;
    case CL_INVALID_PROPERTY:
        return "CL_INVALID_PROPERTY";
        // break;
    default:
        return "UNKNOWN";
    }
}


static map<void*, cl_mem> memobjmap;

void
init(bool do_dp,
     char * source_str,
     cl_device_id fftDev,
     cl_context fftCtx,
     cl_command_queue fftQueue,
     cl_program& fftProg,
     cl_kernel& fftKrnl,
     cl_kernel& ifftKrnl//,
     //cl_kernel& chkKrnl,
     //cl_kernel& goldChkKrnl
     ) {
    cl_int err;

    //FILE*  theFile = fopen(kernel_file, "r");
    //if (!theFile) {
    //    fprintf(stderr, "Failed to load kernel file.\n");
    //    exit(1);
    //}
    //char* source_str;
    //// Obtain length of source file.
    //fseek(theFile, 0, SEEK_END);
    //size_t source_size = ftell(theFile);
    //rewind(theFile);
    //// Read in the file.
    //source_str = (char*) malloc(sizeof(char) * source_size);
    //fread(source_str, 1, source_size, theFile);
    //fclose(theFile);
    //source_str[source_size] = '\0';

    // create the program...
    fftProg = clCreateProgramWithSource(fftCtx, 1, (const char **) &source_str, NULL, &err);

    free(source_str);

//    err = clBuildProgram(fftProg, 0, NULL, "-cl-nv-arch sm_35", NULL, NULL);
    err = clBuildProgram(fftProg, 0, NULL, NULL, NULL, NULL);
    {
        char* log = NULL;
        size_t bytesRequired = 0;
        err = clGetProgramBuildInfo(fftProg,
                                    fftDev,
                                    CL_PROGRAM_BUILD_LOG,
                                    0,
                                    NULL,
                                    &bytesRequired );
        log = (char*)malloc( bytesRequired + 1 );

        err = clGetProgramBuildInfo(fftProg,
                                    fftDev,
                                    CL_PROGRAM_BUILD_LOG,
                                    bytesRequired,
                                    log,
                                    NULL );
        cout << log << endl;
        free( log );
    }
    if (err != CL_SUCCESS) {
        char log[50000];
        size_t retsize = 0;
        err = clGetProgramBuildInfo(fftProg, fftDev, CL_PROGRAM_BUILD_LOG,
                                    50000*sizeof(char),  log, &retsize);
        CL_CHECK_ERROR(err);
	cout << "BUILD LOG:\n";
        cout << "Retsize: " << retsize << endl;
        cout << "Log: " << log << endl;
        exit(-1);
    }

    // Create kernel for forward FFT
    fftKrnl = clCreateKernel(fftProg, "fft1D_512", &err);
    CL_CHECK_ERROR(err);
    // Create kernel for inverse FFT
    ifftKrnl = clCreateKernel(fftProg, "ifft1D_512", &err);
    CL_CHECK_ERROR(err);
    //// Create kernel for check
    //chkKrnl = clCreateKernel(fftProg, "chk1D_512", &err);
    //CL_CHECK_ERROR(err);
    //// Create kernel for efective gold check
    //goldChkKrnl = clCreateKernel(fftProg, "GoldChk", &err);
    //CL_CHECK_ERROR(err);
}

/*
int ocl_exec_gchk(cplxdbl *gold, cl_command_queue& fftQueue, cl_context& context, void* d_odata, cl_kernel& gchk_kernel, int n, int mem_size, size_t thread_per_block, double avoidzero, double acceptdiff)
{
	cl_int err;
	// Prepare GoldChk on GPU...
	int *kerrors=(int*)malloc(sizeof(int));
	cl_mem d_gold;
	cl_mem d_kerrors;
	d_gold = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &err);
	CL_CHECK_ERROR(err);
	d_kerrors = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
	CL_CHECK_ERROR(err);

	*kerrors=0;
	err = clEnqueueWriteBuffer(fftQueue, d_kerrors, CL_TRUE, 0, sizeof(int), kerrors, 0, NULL, NULL);
	CL_CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(fftQueue, d_gold, CL_TRUE, 0, mem_size, gold, 0, NULL, NULL);
	CL_CHECK_ERROR(err);
	
	int *size=(int*)malloc(sizeof(int));
	*size=n;
	err = clSetKernelArg(gchk_kernel, 0, sizeof(cl_mem), (void*)&d_gold);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(gchk_kernel, 1, sizeof(cl_mem), (void*)d_odata);
        CL_CHECK_ERROR(err);
        err = clSetKernelArg(gchk_kernel, 2, sizeof(int), (void*)&size);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(gchk_kernel, 3, sizeof(cl_mem), (void*)&d_kerrors);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(gchk_kernel, 4, sizeof(double), (void*)&acceptdiff);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(gchk_kernel, 5, sizeof(double), (void*)&avoidzero);
	CL_CHECK_ERROR(err);



	// Run GoldChk on GPU...
	size_t gchk_wgsize = n;
	size_t  gchk_lgsize = thread_per_block;
	err = clEnqueueNDRangeKernel(fftQueue, gchk_kernel, 1, NULL, (size_t*)&gchk_wgsize, (size_t*)&gchk_lgsize, 0, NULL, NULL);
	CL_CHECK_ERROR(err);
	err = clFinish(fftQueue);
	CL_CHECK_ERROR(err);	

	// Retrieve kerrors...
	err = clEnqueueReadBuffer(fftQueue, d_kerrors, CL_TRUE, 0, sizeof(int), kerrors, 0, NULL, NULL);
	CL_CHECK_ERROR(err);
	// Release GoldChk GPU resources...
	clReleaseMemObject(d_gold);
	clReleaseMemObject(d_kerrors);

	//printf("kerrors = %i\n", *kerrors);

	return *kerrors;
}
*/
/*
void deinit(cl_command_queue fftQueue,
            cl_program& fftProg,
            cl_kernel& fftKrnl,
            cl_kernel& ifftKrnl,
            cl_kernel& chkKrnl) {
    for (map<void*, cl_mem>::iterator it = memobjmap.begin(); it != memobjmap.end(); ++it) {
        clEnqueueUnmapMemObject(fftQueue, it->second, it->first, 0, NULL, NULL);
        clReleaseMemObject(it->second);
    }

    clReleaseKernel(fftKrnl);
    clReleaseKernel(ifftKrnl);
    clReleaseKernel(chkKrnl);
    clReleaseProgram(fftProg);
}
*/

void
transform(void* workp,
          const int n_ffts,
          cl_kernel& fftKrnl,
          cl_command_queue& fftQueue,
          int distr,
          int fromGPU, int block_size) {
    cl_int err;
    size_t localsz = block_size;
    size_t globalsz = localsz * n_ffts;

    clSetKernelArg(fftKrnl, 0, sizeof(cl_mem), workp);
    clSetKernelArg(fftKrnl, 1, sizeof(distr), (void*) &distr);
    clSetKernelArg(fftKrnl, 2, sizeof(fromGPU), (void*) &fromGPU);
    err = clEnqueueNDRangeKernel(fftQueue, fftKrnl, 1, NULL,
                                 &globalsz, &localsz, 0,
                                 NULL, NULL);

}

/*
int check(const void* workp,
          const void* checkp,
          const int half_n_ffts,
          const int half_n_cmplx,
          cl_kernel& chkKrnl,
          cl_command_queue& fftQueue) {
    cl_int err;
    size_t localsz = 64;
    size_t globalsz = localsz * half_n_ffts;
    int result;

    clSetKernelArg(chkKrnl, 0, sizeof(cl_mem), workp);
    clSetKernelArg(chkKrnl, 1, sizeof(int), (void*)&half_n_cmplx);
    clSetKernelArg(chkKrnl, 2, sizeof(cl_mem), checkp);

    err = clEnqueueNDRangeKernel(fftQueue, chkKrnl, 1, NULL,
                                 &globalsz, &localsz, 0,
                                 NULL, NULL);
    CL_CHECK_ERROR(err);

    err = clEnqueueReadBuffer(fftQueue, *(cl_mem*)checkp, CL_TRUE, 0, sizeof(result),
                              &result, 1, NULL, NULL);
    CL_CHECK_ERROR(err);
    return result;
}
*/

void allocHostBuffer(void** bufp,
                     const unsigned long bytes,
                     cl_context fftCtx,
                     cl_command_queue fftQueue) {
    cl_int err;
    cl_mem memobj = clCreateBuffer(fftCtx,
                                   CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bytes, NULL, &err);
    CL_CHECK_ERROR(err);

    *bufp = clEnqueueMapBuffer(fftQueue, memobj, true,
                               CL_MAP_READ | CL_MAP_WRITE,
                               0,bytes,0,NULL,NULL,&err);
    memobjmap[*bufp] = memobj;
    CL_CHECK_ERROR(err);
}


void freeHostBuffer(void* buf,
                    cl_context fftCtx,
                    cl_command_queue fftQueue) {
    cl_int err;
    cl_mem memobj = memobjmap[buf];
    err = clEnqueueUnmapMemObject(fftQueue, memobj, buf, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(memobj);
    CL_CHECK_ERROR(err);
    memobjmap.erase(buf);
}


void allocDeviceBuffer(void** bufferp,
                       const unsigned long bytes,
                       cl_context fftCtx,
                       cl_command_queue fftQueue) {
    cl_int err;
    *(cl_mem**)bufferp = new cl_mem;
    **(cl_mem**)bufferp = clCreateBuffer(fftCtx, CL_MEM_READ_WRITE, bytes,
                                         NULL, &err);
    CL_CHECK_ERROR(err);
}


void freeDeviceBuffer(void* buffer,
                      cl_context fftCtx,
                      cl_command_queue fftQueue) {
    clReleaseMemObject(*(cl_mem*)buffer);
}


void copyToDevice(void* to_device, void* from_host,
                  const unsigned long bytes, cl_command_queue fftQueue) {
    cl_int err = clEnqueueWriteBuffer(fftQueue, *(cl_mem*)to_device, CL_TRUE,
                                      0, bytes, from_host, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
}


void copyFromDevice(void* to_host, void* from_device,
                    const unsigned long bytes, cl_command_queue fftQueue) {
    cl_int err = clEnqueueReadBuffer(fftQueue, *(cl_mem*)from_device, CL_TRUE,
                                     0, bytes, to_host, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
}


/*****************************************************************************/
