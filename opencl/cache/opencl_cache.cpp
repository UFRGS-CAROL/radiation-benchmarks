
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

#define KERNEL_FILE "/home/daniel/Dropbox/ufrgs/radiation-benchmarks/opencl/cache/cache_kernel.cl"


//define some integer as refword
#define REFWORD 0x00000000
// number of tests
#define NREPS 1

#define TEST_ARRAY_SIZE 15*10*1024 // keep the same number in the define of file => KERNEL_FILE "cache_kernel.cl"

#define OUTPUT_ARRAY_SIZE 3*10 // stores 10 errors details (3 infos for each error)

#define CTAS 10 // define how many blocks we will have
// these two defines below cannot be change because of HW limitation
#define WARPS_PER_CTA 1 // max number of workgroups per compute unit is 16
#define THREADS_PER_WARP 64 //warp == wavefront for opencl

#define STRIDE WARPS_PER_CTA * THREADS_PER_WARP * CTAS;


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
		case CL_SUCCESS:                         return "CL_SUCCESS";
								 // break;
		case CL_DEVICE_NOT_FOUND:                return "CL_DEVICE_NOT_FOUND";
								 // break;
		case CL_DEVICE_NOT_AVAILABLE:            return "CL_DEVICE_NOT_AVAILABLE";
								 // break;
		case CL_COMPILER_NOT_AVAILABLE:          return "CL_COMPILER_NOT_AVAILABLE";
								 // break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:   return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
								 // break;
		case CL_OUT_OF_RESOURCES:                return "CL_OUT_OF_RESOURCES";
								 // break;
		case CL_OUT_OF_HOST_MEMORY:              return "CL_OUT_OF_HOST_MEMORY";
								 // break;
		case CL_PROFILING_INFO_NOT_AVAILABLE:    return "CL_PROFILING_INFO_NOT_AVAILABLE";
								 // break;
		case CL_MEM_COPY_OVERLAP:                return "CL_MEM_COPY_OVERLAP";
								 // break;
		case CL_IMAGE_FORMAT_MISMATCH:           return "CL_IMAGE_FORMAT_MISMATCH";
								 // break;
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
								 // break;
		case CL_BUILD_PROGRAM_FAILURE:           return "CL_BUILD_PROGRAM_FAILURE";
								 // break;
		case CL_MAP_FAILURE:                     return "CL_MAP_FAILURE";
								 // break;
		case CL_INVALID_VALUE:                   return "CL_INVALID_VALUE";
								 // break;
		case CL_INVALID_DEVICE_TYPE:             return "CL_INVALID_DEVICE_TYPE";
								 // break;
		case CL_INVALID_PLATFORM:                return "CL_INVALID_PLATFORM";
								 // break;
		case CL_INVALID_DEVICE:                  return "CL_INVALID_DEVICE";
								 // break;
		case CL_INVALID_CONTEXT:                 return "CL_INVALID_CONTEXT";
								 // break;
		case CL_INVALID_QUEUE_PROPERTIES:        return "CL_INVALID_QUEUE_PROPERTIES";
								 // break;
		case CL_INVALID_COMMAND_QUEUE:           return "CL_INVALID_COMMAND_QUEUE";
								 // break;
		case CL_INVALID_HOST_PTR:                return "CL_INVALID_HOST_PTR";
								 // break;
		case CL_INVALID_MEM_OBJECT:              return "CL_INVALID_MEM_OBJECT";
								 // break;
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
								 // break;
		case CL_INVALID_IMAGE_SIZE:              return "CL_INVALID_IMAGE_SIZE";
								 // break;
		case CL_INVALID_SAMPLER:                 return "CL_INVALID_SAMPLER";
								 // break;
		case CL_INVALID_BINARY:                  return "CL_INVALID_BINARY";
								 // break;
		case CL_INVALID_BUILD_OPTIONS:           return "CL_INVALID_BUILD_OPTIONS";
								 // break;
		case CL_INVALID_PROGRAM:                 return "CL_INVALID_PROGRAM";
								 // break;
		case CL_INVALID_PROGRAM_EXECUTABLE:      return "CL_INVALID_PROGRAM_EXECUTABLE";
								 // break;
		case CL_INVALID_KERNEL_NAME:             return "CL_INVALID_KERNEL_NAME";
								 // break;
		case CL_INVALID_KERNEL_DEFINITION:       return "CL_INVALID_KERNEL_DEFINITION";
								 // break;
		case CL_INVALID_KERNEL:                  return "CL_INVALID_KERNEL";
								 // break;
		case CL_INVALID_ARG_INDEX:               return "CL_INVALID_ARG_INDEX";
								 // break;
		case CL_INVALID_ARG_VALUE:               return "CL_INVALID_ARG_VALUE";
								 // break;
		case CL_INVALID_ARG_SIZE:                return "CL_INVALID_ARG_SIZE";
								 // break;
		case CL_INVALID_KERNEL_ARGS:             return "CL_INVALID_KERNEL_ARGS";
								 // break;
		case CL_INVALID_WORK_DIMENSION:          return "CL_INVALID_WORK_DIMENSION";
								 // break;
		case CL_INVALID_WORK_GROUP_SIZE:         return "CL_INVALID_WORK_GROUP_SIZE";
								 // break;
		case CL_INVALID_WORK_ITEM_SIZE:          return "CL_INVALID_WORK_ITEM_SIZE";
								 // break;
		case CL_INVALID_GLOBAL_OFFSET:           return "CL_INVALID_GLOBAL_OFFSET";
								 // break;
		case CL_INVALID_EVENT_WAIT_LIST:         return "CL_INVALID_EVENT_WAIT_LIST";
								 // break;
		case CL_INVALID_EVENT:                   return "CL_INVALID_EVENT";
								 // break;
		case CL_INVALID_OPERATION:               return "CL_INVALID_OPERATION";
								 // break;
		case CL_INVALID_GL_OBJECT:               return "CL_INVALID_GL_OBJECT";
								 // break;
		case CL_INVALID_BUFFER_SIZE:             return "CL_INVALID_BUFFER_SIZE";
								 // break;
		case CL_INVALID_MIP_LEVEL:               return "CL_INVALID_MIP_LEVEL";
								 // break;
		case CL_INVALID_GLOBAL_WORK_SIZE:        return "CL_INVALID_GLOBAL_WORK_SIZE";
								 // break;
		case CL_INVALID_PROPERTY:                return "CL_INVALID_PROPERTY";
								 // break;
		default:                                 return "UNKNOWN";
	}
}

double mysecond(){
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
	if (0)
	{
		printf("Using the default platform (platform 0)...\n\n");
		printf("=== %d OpenCL device(s) found on platform:\n", devices_n);
		for (int i = 0; i < devices_n; i++)
		{
			char buffer[10240];
			cl_uint buf_uint;
			cl_ulong buf_ulong;
			printf("  -- %d --\n", i);
			clGetDeviceInfo(device_id[i], CL_DEVICE_NAME, sizeof(buffer), buffer,
				NULL);
			printf("  DEVICE_NAME = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer,
				NULL);
			printf("  DEVICE_VENDOR = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DEVICE_VERSION, sizeof(buffer), buffer,
				NULL);
			printf("  DEVICE_VERSION = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DRIVER_VERSION, sizeof(buffer), buffer,
				NULL);
			printf("  DRIVER_VERSION = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DEVICE_MAX_COMPUTE_UNITS,
				sizeof(buf_uint), &buf_uint, NULL);
			printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int) buf_uint);
			clGetDeviceInfo(device_id[i], CL_DEVICE_MAX_CLOCK_FREQUENCY,
				sizeof(buf_uint), &buf_uint, NULL);
			printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int) buf_uint);
			clGetDeviceInfo(device_id[i], CL_DEVICE_GLOBAL_MEM_SIZE,
				sizeof(buf_ulong), &buf_ulong, NULL);
			printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n",
				(unsigned long long) buf_ulong);
			clGetDeviceInfo(device_id[i], CL_DEVICE_LOCAL_MEM_SIZE,
				sizeof(buf_ulong), &buf_ulong, NULL);
			printf("  CL_DEVICE_LOCAL_MEM_SIZE = %llu\n",
				(unsigned long long) buf_ulong);
		}
		printf("\n");
	}

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


/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////

int
main(int argc, char** argv)
{



	// 5. Initialize OpenCL
	// OpenCL specific variables
	cl_program clProgram;
	cl_kernel clKernel;

	cl_int errcode;

	int * h_output_array = (int *)malloc(sizeof(int)*OUTPUT_ARRAY_SIZE);
//printf("cl_mem...\n");
	// OpenCL device memory for matrices
	cl_mem d_test_array;
	cl_mem d_output_array;
	cl_mem d_num_errors;

	/*****************************************/
	/* Initialize OpenCL */
	/*****************************************/
	getDevices(CL_DEVICE_TYPE_ALL);

//printf("getDevices...\n");
	// Setup device memory
	d_num_errors = clCreateBuffer(context,
		CL_MEM_READ_WRITE,
		sizeof(int), NULL, &errcode);
	d_test_array = clCreateBuffer(context,
		CL_MEM_READ_WRITE,
		TEST_ARRAY_SIZE*sizeof(int), NULL, &errcode);
	d_output_array = clCreateBuffer(context,
		CL_MEM_READ_WRITE,
		OUTPUT_ARRAY_SIZE*sizeof(int), NULL, &errcode);

//printf("createBuffer...\n");

	// 6. Load and build OpenCL kernel
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

//printf("read kernel.cl...\n");
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

//printf("BuildProgram...\n");
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
        //printf("Build; error=%d, status=%d, programLog:\n\n%s", errcode, status, programLog);
        free(programLog);


//printf("createKernel...\n");
	clKernel = clCreateKernel(clProgram,
		"testCache", &errcode);
	CL_CHECK_ERROR(errcode);


	// 7. Launch OpenCL kernel
	size_t localWorkSize[2], globalWorkSize[2];

//printf("parameters...\n");
	int kernel_errors = 0;
	clEnqueueWriteBuffer(command_queue, d_num_errors, CL_FALSE, 0, 
                sizeof(int), &kernel_errors, 0, NULL, NULL); 
	clFinish(command_queue);

	int refword = REFWORD;
	int nreps = NREPS;
	int stride = STRIDE;
	errcode |= clSetKernelArg(clKernel, 0,
		sizeof(cl_mem), (void *)&d_test_array);
	errcode |= clSetKernelArg(clKernel, 0,
		sizeof(cl_mem), (void *)&d_output_array);
	errcode |= clSetKernelArg(clKernel, 1,
		sizeof(int), (void *)&stride);
	errcode |= clSetKernelArg(clKernel, 2,
		sizeof(int), (void *)&refword);
	errcode |= clSetKernelArg(clKernel, 3,
		sizeof(int), (void *)&nreps);
	errcode |= clSetKernelArg(clKernel, 4,
		sizeof(cl_mem), (void *)&d_num_errors);

	CL_CHECK_ERROR(errcode);

// CUDA:
// WARPS_PER_CTA * THREADS_PER_WARP => blocks
// CTAS => grid
	localWorkSize[0] = WARPS_PER_CTA * THREADS_PER_WARP;
	globalWorkSize[0] = WARPS_PER_CTA * THREADS_PER_WARP * CTAS;

	clFinish(command_queue);

//printf("RUN...\n");
	double timeG = mysecond();
	errcode = clEnqueueNDRangeKernel(command_queue,
		clKernel, 1, NULL, globalWorkSize,
		localWorkSize, 0, NULL, NULL);
	CL_CHECK_ERROR(errcode);
	clFinish(command_queue);
	timeG = mysecond() - timeG;


	errcode = clEnqueueReadBuffer(command_queue,
		d_num_errors, CL_TRUE, 0, sizeof(int),
		&kernel_errors, 0, NULL, NULL);
	CL_CHECK_ERROR(errcode);

	printf("check kernel errors = %d\n",kernel_errors);
	printf("kernel time: %f\n", timeG);

	// download error details from output_array
	if(kernel_errors > 0){
		errcode = clEnqueueReadBuffer(command_queue,
			d_output_array, CL_TRUE, 0, OUTPUT_ARRAY_SIZE*sizeof(int),
			&h_output_array, 0, NULL, NULL);
		CL_CHECK_ERROR(errcode);
		int i;
		for(i = 0; i < kernel_errors*3; i=i+3){
			printf("%d it, %d pos, 0x%08x syndrome\n",h_output_array[i],h_output_array[i+1],h_output_array[i+2]);
		}
	}

	// 10. clean up memory

	clReleaseMemObject(d_num_errors);

	clReleaseContext(context);
	clReleaseKernel(clKernel);
	clReleaseProgram(clProgram);
	clReleaseCommandQueue(command_queue);


}
