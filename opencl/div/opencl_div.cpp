
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_ext.h>


#define N 15*10*1024
#define NUM_REP 100

#define ITERACTIONS 1

#define KERNEL_FILE "/home/daniel/Dropbox/ufrgs/radiation-benchmarks/opencl/div/div_kernel.cl"

#define CL_CHECK_ERROR(err) \
	{ \
		if (err != CL_SUCCESS) \
		std::cerr << "Error: " \
		<< CLErrorString(err) \
		<< " in " << __FILE__ \
		<< " line " << __LINE__ \
		<< std::endl; \
	}

inline const char *CLErrorString(cl_int err)
{
    switch (err)
    {
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

double *A;
double *B;
double *C;
double *GOLD;

double fRand(double fmin, double fmax) {

    double f = (double) rand() / RAND_MAX;

    return (double )fmin + f * (fmax - fmin);
}


void generateInput() {

    int i;
    for(i=0; i<N; i++)
    {
        do {
            A[i] = fRand(-1590.35, 1987.59);
            B[i] = fRand(-15.65, 15.68);
            GOLD[i] = A[i] / pow(B[i], NUM_REP);
        } while(A[i] == 0 || B[i] == 0 || GOLD[i] == 0 );

    }

}


double mysecond() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

cl_platform_id          platform_id[100];
cl_device_id            device_id[100];
cl_context              context;
cl_command_queue        command_queue;
cl_int ret;
void getDevices(cl_device_type deviceType)
{
    cl_uint         platforms_n = 0;
    cl_uint         devices_n   = 0;

    clGetPlatformIDs(100, platform_id, &platforms_n);

    clGetDeviceIDs(platform_id[0], deviceType, 100, device_id, &devices_n);

    // Create an OpenCL context.
    context = clCreateContext(NULL, devices_n, device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS)
    {
        printf("\nError at clCreateContext! Error code %i\n\n", ret);
        exit(1);
    }

    // Create a command queue.
    command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);
    if (ret != CL_SUCCESS)
    {
        printf("\nError at clCreateCommandQueue! Error code %i\n\n", ret);
        exit(1);
    }
}


int
main(int argc, char** argv)
{

    int size = N;

    A = ( double* ) malloc( size * sizeof( double ) );
    B = ( double* ) malloc( size * sizeof( double ) );

    GOLD = ( double* ) malloc( size * sizeof( double ) );
    C = ( double* ) malloc( size * sizeof( double ) );

    generateInput();

    cl_program clProgram;
    cl_kernel clKernel;
    cl_kernel clKernelCheck;

    size_t dataBytes;
    size_t kernelLength;
    cl_int errcode;

    cl_mem d_A;
    cl_mem d_B;
    cl_mem d_GOLD;
    cl_mem d_num_errors;

    // Initialize OpenCL
    getDevices(CL_DEVICE_TYPE_GPU);

    // Setup device memory
    unsigned int mem_size = sizeof(double) * size;
    d_num_errors = clCreateBuffer(context,
                                  CL_MEM_READ_WRITE,
                                  sizeof(int), NULL, &errcode);
    d_A = clCreateBuffer(context,
                         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                         mem_size, A, &errcode);
    d_B = clCreateBuffer(context,
                         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                         mem_size, B, &errcode);
    d_GOLD = clCreateBuffer(context,
                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            mem_size, GOLD, &errcode);

    // Load and build OpenCL kernel
    FILE*  theFile = fopen(KERNEL_FILE, "r");
    if (!theFile)
    {
        fprintf(stderr, "Failed to load kernel file.\n");
        exit(1);
    }
    char* source_str;

    // Obtain length of source file.
    fseek(theFile, 0, SEEK_END);
    size_t source_size = ftell(theFile);
    rewind(theFile);

    // Read in the file.
    source_str = (char*) malloc(sizeof(char) * source_size);
    fread(source_str, 1, source_size, theFile);
    fclose(theFile);
    source_str[source_size] = '\0';
    CL_CHECK_ERROR(errcode);
    clProgram = clCreateProgramWithSource(context,
                                          1, (const char **)&source_str,
                                          NULL, &errcode);
    CL_CHECK_ERROR(errcode);

    free(source_str);

    errcode = clBuildProgram(clProgram, 1,
                             &device_id[0], NULL, NULL, NULL);
    CL_CHECK_ERROR(errcode);

    cl_build_status status;
    size_t logSize;
    char *programLog;
    // check build error and build status first
    clGetProgramBuildInfo(clProgram, device_id[0], CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, NULL);


    // check build log
    clGetProgramBuildInfo(clProgram, device_id[0],
                          CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    programLog = (char*) calloc (logSize+1, sizeof(char));
    clGetProgramBuildInfo(clProgram, device_id[0],
                          CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);
    //printf("Build; error=%d, status=%d, programLog:\n\n%s",errcode, status, programLog);
    free(programLog);


    clKernel = clCreateKernel(clProgram,
                              "simpleDiv", &errcode);
    CL_CHECK_ERROR(errcode);


    // Prepare to launch OpenCL kernel
    size_t localWorkSize[2], globalWorkSize[2];


    int n = NUM_REP;
    errcode |= clSetKernelArg(clKernel, 0,
                              sizeof(cl_mem), (void *)&d_A);
    errcode |= clSetKernelArg(clKernel, 1,
                              sizeof(cl_mem), (void *)&d_B);
    errcode |= clSetKernelArg(clKernel, 2,
                              sizeof(int), (void *)&n);
    CL_CHECK_ERROR(errcode);

    localWorkSize[0] = 256;
    globalWorkSize[0] = N;

    clFinish(command_queue);

    // Run kernel
    double timeG = mysecond();
    errcode = clEnqueueNDRangeKernel(command_queue,
                                     clKernel, 1, NULL, globalWorkSize,
                                     localWorkSize, 0, NULL, NULL);
    CL_CHECK_ERROR(errcode);
    clFinish(command_queue);
    timeG = mysecond() - timeG;



    // Retrieve result from device
    errcode = clEnqueueReadBuffer(command_queue,
                                  d_A, CL_TRUE, 0, mem_size,
                                  A, 0, NULL, NULL);
    CL_CHECK_ERROR(errcode);

    // CHECK GOLD
    clKernelCheck = clCreateKernel(clProgram,
                                   "checkGold", &errcode);
    CL_CHECK_ERROR(errcode);

    clEnqueueWriteBuffer(command_queue, d_GOLD, CL_FALSE, 0,
                         mem_size, GOLD, 0, NULL, NULL);

    int kernel_errors = 0;
    clEnqueueWriteBuffer(command_queue, d_num_errors, CL_FALSE, 0, sizeof(int), &kernel_errors, 0, NULL, NULL);

    clFinish(command_queue);
    errcode = clSetKernelArg(clKernelCheck, 0,
                             sizeof(cl_mem), (void *)&d_A);
    errcode |= clSetKernelArg(clKernelCheck, 1,
                              sizeof(cl_mem), (void *)&d_GOLD);
    errcode |= clSetKernelArg(clKernelCheck, 2,
                              sizeof(cl_mem), (void *)&d_num_errors);
    CL_CHECK_ERROR(errcode);

    errcode = clEnqueueNDRangeKernel(command_queue,
                                     clKernelCheck, 1, NULL, globalWorkSize,
                                     localWorkSize, 0, NULL, NULL);
    CL_CHECK_ERROR(errcode);
    clFinish(command_queue);

    errcode = clEnqueueReadBuffer(command_queue,
                                  d_num_errors, CL_TRUE, 0, sizeof(int),
                                  &kernel_errors, 0, NULL, NULL);
    CL_CHECK_ERROR(errcode);

    printf("\ncheck kernel ea = %d\n",kernel_errors);
    printf("kernel time: %f\n", timeG);


    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_GOLD);
    clReleaseMemObject(d_num_errors);

    clReleaseContext(context);
    clReleaseKernel(clKernel);
    clReleaseKernel(clKernelCheck);
    clReleaseProgram(clProgram);
    clReleaseCommandQueue(command_queue);


}
