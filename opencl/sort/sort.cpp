#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <iostream>
#include <fstream>
#include <vector>

#include <sys/time.h>
#include <time.h>

#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_ext.h>

#define ITERACTIONS 1		 //loop

using namespace std;

cl_platform_id          platform_id[100];
cl_device_id            device_id[100];
cl_context              context;
cl_command_queue        command_queue[2];
cl_kernel reduce;
cl_kernel top_scan;
cl_kernel bottom_scan;
cl_kernel cpu_reduce;
cl_kernel cpu_top_scan;
cl_kernel cpu_bottom_scan;

int gpu_work = 8; // default value is 8 (100% GPU, 0% CPU), max value is also 8

// Returns the current system time in microseconds
long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}



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



template <class T>
void runTest(int test_number) {

	int err;
	// Problem Sizes
	int probSizes[7] = { 1, 8, 32, 64 , 128, 256, 512};
	int size = probSizes[5];

	// Convert to MiB
	size = (size * 1024 * 1024) / sizeof(T);

	// Create input data on CPU
	unsigned int bytes = size * sizeof(T);

	// Allocate pinned host memory for input data (h_idata)
	cl_mem h_i = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		bytes, NULL, &err);
	CL_CHECK_ERROR(err);
	T* h_idata = (T*)clEnqueueMapBuffer(command_queue[0], h_i, true,
		CL_MAP_READ|CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
	CL_CHECK_ERROR(err);

	// Allocate pinned host memory for output data (h_odata)
	cl_mem h_o = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		bytes, NULL, &err);
	CL_CHECK_ERROR(err);
	T* h_odata = (T*)clEnqueueMapBuffer(command_queue[0], h_o, true,
		CL_MAP_READ|CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
	CL_CHECK_ERROR(err);

	FILE *fp;
	if( (fp = fopen("input_sort", "rb" )) == 0 )
		printf( "The file input_sort was not opened\n");
	// Initialize host memory
	for (int i = 0; i < size; i++) {
		fread(&(h_idata[i]), 1, sizeof(T), fp);

		h_odata[i] = -1;
	}
	fclose(fp);

	// The radix width in bits
	const int radix_width = 4;	 // Changing this requires major kernel updates
								 // n possible digits
	const int num_digits = (int)pow((double)2, radix_width);

	// Allocate device memory for input array
	cl_mem d_idata = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
	CL_CHECK_ERROR(err);

	// Allocate device memory for output array
	cl_mem d_odata = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
	CL_CHECK_ERROR(err);

	// Allocate device memory for input array
	cl_mem cpu_d_idata = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
	CL_CHECK_ERROR(err);

	// Allocate device memory for output array
	cl_mem cpu_d_odata = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
	CL_CHECK_ERROR(err);

	// Number of local work items per group
	const size_t local_wsize  = 256;

	// Number of global work items
								 // i.e. 64 work groups
	const size_t global_wsize = 16384;
	const size_t num_work_groups = global_wsize / local_wsize;

	// Allocate device memory for local work group intermediate sums
	cl_mem d_isums = clCreateBuffer(context, CL_MEM_READ_WRITE,
		num_work_groups * num_digits * sizeof(T), NULL, &err);
	CL_CHECK_ERROR(err);
	// Allocate device memory for local work group intermediate sums
	cl_mem cpu_d_isums = clCreateBuffer(context, CL_MEM_READ_WRITE,
		num_work_groups * num_digits * sizeof(T), NULL, &err);
	CL_CHECK_ERROR(err);

	// Set the kernel arguments for the reduction kernel
	err = clSetKernelArg(reduce, 0, sizeof(cl_mem), (void*)&d_idata);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(reduce, 1, sizeof(cl_mem), (void*)&d_isums);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(reduce, 2, sizeof(cl_int), (void*)&size);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(reduce, 3, local_wsize * sizeof(T), NULL);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(cpu_reduce, 0, sizeof(cl_mem), (void*)&cpu_d_idata);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(cpu_reduce, 1, sizeof(cl_mem), (void*)&cpu_d_isums);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(cpu_reduce, 2, sizeof(cl_int), (void*)&size);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(cpu_reduce, 3, local_wsize * sizeof(T), NULL);
	CL_CHECK_ERROR(err);

	// Set the kernel arguments for the top-level scan
	err = clSetKernelArg(top_scan, 0, sizeof(cl_mem), (void*)&d_isums);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(top_scan, 1, sizeof(cl_int), (void*)&num_work_groups);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(top_scan, 2, local_wsize * 2 * sizeof(T), NULL);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(cpu_top_scan, 0, sizeof(cl_mem), (void*)&cpu_d_isums);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(cpu_top_scan, 1, sizeof(cl_int), (void*)&num_work_groups);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(cpu_top_scan, 2, local_wsize * 2 * sizeof(T), NULL);
	CL_CHECK_ERROR(err);

	// Set the kernel arguments for the bottom-level scan
	err = clSetKernelArg(bottom_scan, 0, sizeof(cl_mem), (void*)&d_idata);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(bottom_scan, 1, sizeof(cl_mem), (void*)&d_isums);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(bottom_scan, 2, sizeof(cl_mem), (void*)&d_odata);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(bottom_scan, 3, sizeof(cl_int), (void*)&size);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(bottom_scan, 4, local_wsize * 2 * sizeof(T), NULL);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(cpu_bottom_scan, 0, sizeof(cl_mem), (void*)&cpu_d_idata);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(cpu_bottom_scan, 1, sizeof(cl_mem), (void*)&cpu_d_isums);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(cpu_bottom_scan, 2, sizeof(cl_mem), (void*)&cpu_d_odata);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(cpu_bottom_scan, 3, sizeof(cl_int), (void*)&size);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(cpu_bottom_scan, 4, local_wsize * 2 * sizeof(T), NULL);
	CL_CHECK_ERROR(err);

	long long time0, time1;
	double gpu_kernel_time = 0, cpu_kernel_time = 0;
	int shift = 0;
	bool even;


if(gpu_work > 0){
	// Copy data to GPU
	//cout << "Copying input data to device." << endl;
	err = clEnqueueWriteBuffer(command_queue[0], d_idata, true, 0, bytes, h_idata, 0,
		NULL, NULL);
	CL_CHECK_ERROR(err);
	err = clFinish(command_queue[0]);
	CL_CHECK_ERROR(err);

/////////////
// GPU run
/////////////
	time0 = get_time();
	for (shift = 0; shift < sizeof(T)*gpu_work; shift += radix_width) {
		// Like scan, we use a reduce-then-scan approach

		// But before proceeding, update the shift appropriately
		// for each kernel. This is how many bits to shift to the
		// right used in binning.
		err = clSetKernelArg(reduce, 4, sizeof(cl_int), (void*)&shift);
		CL_CHECK_ERROR(err);

		err = clSetKernelArg(bottom_scan, 5, sizeof(cl_int), (void*)&shift);
		CL_CHECK_ERROR(err);

		// Also, the sort is not in place, so swap the input and output
		// buffers on each pass.
		even = ((shift / radix_width) % 2 == 0) ? true : false;

		if (even) {
printf("GPU even, shift = %d\n",shift);
			// Set the kernel arguments for the reduction kernel
			err = clSetKernelArg(reduce, 0, sizeof(cl_mem),
				(void*)&d_idata);
			CL_CHECK_ERROR(err);
			// Set the kernel arguments for the bottom-level scan
			err = clSetKernelArg(bottom_scan, 0, sizeof(cl_mem),
				(void*)&d_idata);
			CL_CHECK_ERROR(err);
			err = clSetKernelArg(bottom_scan, 2, sizeof(cl_mem),
				(void*)&d_odata);
			CL_CHECK_ERROR(err);
		}
		else {				 // i.e. odd pass
printf("GPU odd, shift = %d\n",shift);
			// Set the kernel arguments for the reduction kernel
			err = clSetKernelArg(reduce, 0, sizeof(cl_mem),
				(void*)&d_odata);
			CL_CHECK_ERROR(err);
			// Set the kernel arguments for the bottom-level scan
			err = clSetKernelArg(bottom_scan, 0, sizeof(cl_mem),
				(void*)&d_odata);
			CL_CHECK_ERROR(err);
			err = clSetKernelArg(bottom_scan, 2, sizeof(cl_mem),
				(void*)&d_idata);
			CL_CHECK_ERROR(err);
		}

		// Each thread block gets an equal portion of the
		// input array, and computes occurrences of each digit.
		err = clEnqueueNDRangeKernel(command_queue[0], reduce, 1, NULL,
			&global_wsize, &local_wsize, 0, NULL, NULL);

		// Next, a top-level exclusive scan is performed on the
		// per block histograms.  This is done by a single
		// work group (note global size here is the same as local).
		err = clEnqueueNDRangeKernel(command_queue[0], top_scan, 1, NULL,
			&local_wsize, &local_wsize, 0, NULL, NULL);

		// Finally, a bottom-level scan is performed by each block
		// that is seeded with the scanned histograms which rebins,
		// locally scans, then scatters keys to global memory
		err = clEnqueueNDRangeKernel(command_queue[0], bottom_scan, 1, NULL,
			&global_wsize, &local_wsize, 0, NULL, NULL);
	}
	err = clFinish(command_queue[0]);
	CL_CHECK_ERROR(err);
	time1 = get_time();
	gpu_kernel_time = (double) (time1-time0) / 1000000;
	if (even){
		err = clEnqueueReadBuffer(command_queue[0], d_odata, true, 0, bytes, h_odata,
			0, NULL, NULL);
	}else{
		err = clEnqueueReadBuffer(command_queue[0], d_idata, true, 0, bytes, h_odata,
			0, NULL, NULL);
	}

	CL_CHECK_ERROR(err);
	err = clFinish(command_queue[0]);
	CL_CHECK_ERROR(err);
}

if(gpu_work < 8){
//////////////
// CPU part
//////////////

	// Copy data to CPU device
	if (even){
		if(gpu_work > 0) {
			err = clEnqueueWriteBuffer(command_queue[1], cpu_d_idata, true, 0, bytes, h_odata, 0, NULL, NULL);
		} else {
			err = clEnqueueWriteBuffer(command_queue[1], cpu_d_idata, true, 0, bytes, h_idata, 0, NULL, NULL);
		}
	}else{
		if(gpu_work > 0) {
			err = clEnqueueWriteBuffer(command_queue[1], cpu_d_odata, true, 0, bytes, h_odata, 0, NULL, NULL);
		} else {
			err = clEnqueueWriteBuffer(command_queue[1], cpu_d_odata, true, 0, bytes, h_idata, 0, NULL, NULL);
		}
	}
	CL_CHECK_ERROR(err);
	err = clFinish(command_queue[1]);
	CL_CHECK_ERROR(err);


/////////////
// CPU run
/////////////
	time0 = get_time();
	for (; shift < sizeof(T)*8; shift += radix_width) {
		// Like scan, we use a reduce-then-scan approach

		// But before proceeding, update the shift appropriately
		// for each kernel. This is how many bits to shift to the
		// right used in binning.
		err = clSetKernelArg(cpu_reduce, 4, sizeof(cl_int), (void*)&shift);
		CL_CHECK_ERROR(err);

		err = clSetKernelArg(cpu_bottom_scan, 5, sizeof(cl_int), (void*)&shift);
		CL_CHECK_ERROR(err);

		// Also, the sort is not in place, so swap the input and output
		// buffers on each pass.
		even = ((shift / radix_width) % 2 == 0) ? true : false;

		if (even) {
printf("CPU even, shift = %d\n",shift);
			// Set the kernel arguments for the reduction kernel
			err = clSetKernelArg(cpu_reduce, 0, sizeof(cl_mem),
				(void*)&cpu_d_idata);
			CL_CHECK_ERROR(err);
			// Set the kernel arguments for the bottom-level scan
			err = clSetKernelArg(cpu_bottom_scan, 0, sizeof(cl_mem),
				(void*)&cpu_d_idata);
			CL_CHECK_ERROR(err);
			err = clSetKernelArg(cpu_bottom_scan, 2, sizeof(cl_mem),
				(void*)&cpu_d_odata);
			CL_CHECK_ERROR(err);
		}
		else {				 // i.e. odd pass
printf("CPU odd, shift = %d\n",shift);
			// Set the kernel arguments for the reduction kernel
			err = clSetKernelArg(cpu_reduce, 0, sizeof(cl_mem),
				(void*)&cpu_d_odata);
			CL_CHECK_ERROR(err);
			// Set the kernel arguments for the bottom-level scan
			err = clSetKernelArg(cpu_bottom_scan, 0, sizeof(cl_mem),
				(void*)&cpu_d_odata);
			CL_CHECK_ERROR(err);
			err = clSetKernelArg(cpu_bottom_scan, 2, sizeof(cl_mem),
				(void*)&cpu_d_idata);
			CL_CHECK_ERROR(err);
		}

		// Each thread block gets an equal portion of the
		// input array, and computes occurrences of each digit.
		err = clEnqueueNDRangeKernel(command_queue[1], cpu_reduce, 1, NULL,
			&global_wsize, &local_wsize, 0, NULL, NULL);

		// Next, a top-level exclusive scan is performed on the
		// per block histograms.  This is done by a single
		// work group (note global size here is the same as local).
		err = clEnqueueNDRangeKernel(command_queue[1], cpu_top_scan, 1, NULL,
			&local_wsize, &local_wsize, 0, NULL, NULL);

		// Finally, a bottom-level scan is performed by each block
		// that is seeded with the scanned histograms which rebins,
		// locally scans, then scatters keys to global memory
		err = clEnqueueNDRangeKernel(command_queue[1], cpu_bottom_scan, 1, NULL,
			&global_wsize, &local_wsize, 0, NULL, NULL);
	}
	err = clFinish(command_queue[1]);
	CL_CHECK_ERROR(err);
	time1 = get_time();
	cpu_kernel_time = (double) (time1-time0) / 1000000;
	if (even){
		err = clEnqueueReadBuffer(command_queue[1], cpu_d_odata, true, 0, bytes, h_odata,
			0, NULL, NULL);
	}else{
		err = clEnqueueReadBuffer(command_queue[1], cpu_d_idata, true, 0, bytes, h_odata,
			0, NULL, NULL);
	}

	CL_CHECK_ERROR(err);
	err = clFinish(command_queue[1]);
	CL_CHECK_ERROR(err);
}


	// check output with GOLD
	if( (fp = fopen("output_sort", "rb" )) == 0 ) {
		printf( "error file output_sort was not opened\n");
		return;
	}
	int num_errors = 0;
	int gold;
	int i;
	int order_errors = 0;
	for (i = 0; i < size; i++) {
		fread(&gold, 1, sizeof(T), fp);
		if(h_odata[i] != gold){
			num_errors++;
if (num_errors < 5)
	printf("error:\ne: %d\nr: %d\n",gold, h_odata[i]);
		}

		if (i < size -1 && h_odata[i] > h_odata[i + 1]) {
			order_errors++;
		}
	}

	fclose(fp);


	printf("\ntest number: %d", test_number);
	printf("\nGPU kernel time: %.12f", gpu_kernel_time);
	printf("\nCPU kernel time: %.12f", cpu_kernel_time);
	printf("\namount of errors: %d", num_errors);
	printf("\nelements with order errors: %d\n", order_errors);


	// Clean up device memory
	err = clReleaseMemObject(d_idata);
	CL_CHECK_ERROR(err);
	err = clReleaseMemObject(d_odata);
	CL_CHECK_ERROR(err);
	err = clReleaseMemObject(d_isums);
	CL_CHECK_ERROR(err);

	// Clean up pinned host memory
	err = clEnqueueUnmapMemObject(command_queue[0], h_i, h_idata, 0, NULL, NULL);
	CL_CHECK_ERROR(err);
	err = clEnqueueUnmapMemObject(command_queue[0], h_o, h_odata, 0, NULL, NULL);
	CL_CHECK_ERROR(err);
	err = clReleaseMemObject(h_i);
	CL_CHECK_ERROR(err);
	err = clReleaseMemObject(h_o);
	CL_CHECK_ERROR(err);

}

void initOpenCL() {
    cl_uint         platforms_n = 0;
    cl_uint         devices_n   = 0;
    cl_int                  ret;

    clGetPlatformIDs(100, platform_id, &platforms_n);
	/*
	CL_DEVICE_TYPE_DEFAULT
	CL_DEVICE_TYPE_CPU
	CL_DEVICE_TYPE_GPU
	CL_DEVICE_TYPE_ACCELERATOR
	CL_DEVICE_TYPE_ALL
	*/
    clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_ALL, 100, device_id, &devices_n);

    // Create an OpenCL context.
    context = clCreateContext(NULL, devices_n, device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS) {
        printf("\nError at clCreateContext! Error code %i\n\n", ret);
        exit(1);
    }

    // Create a command queue.
    command_queue[0] = clCreateCommandQueue(context, device_id[0], 0, &ret);
    if (ret != CL_SUCCESS) {
        printf("\nError at clCreateCommandQueue! Error code %i\n\n", ret);
        exit(1);
    }

    // Create a command queue.
    command_queue[1] = clCreateCommandQueue(context, device_id[1], 0, &ret);
    if (ret != CL_SUCCESS) {
        printf("\nError at clCreateCommandQueue! Error code %i\n\n", ret);
        exit(1);
    }


	int err = 0;

	FILE*  theFile = fopen("sort.cl", "r");
	if (!theFile) {
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
	// Program Setup
	cl_program gpuprog = clCreateProgramWithSource(context,
		1,
		(const char **) &source_str,
		NULL,
		&err);
	CL_CHECK_ERROR(err);
	cl_program cpuprog = clCreateProgramWithSource(context,
		1,
		(const char **) &source_str,
		NULL,
		&err);
	CL_CHECK_ERROR(err);

	// Before proceeding, make sure the kernel code compiles and
	// all kernels are valid.
	//cout << "Compiling sort kernels." << endl;
	err = clBuildProgram(gpuprog, 1, &device_id[0], NULL, NULL, NULL);
	CL_CHECK_ERROR(err);
	if (err != CL_SUCCESS) {
		printf("Error compiling sort kernels\n");
		exit(0);
	}
	err = clBuildProgram(cpuprog, 1, &device_id[1], NULL, NULL, NULL);
	CL_CHECK_ERROR(err);
	if (err != CL_SUCCESS) {
		printf("Error compiling sort kernels\n");
		exit(0);
	}

	// Extract out the 3 kernels
	// Note that these kernels are analogs of those in use for
	// scan, but have had "visiting" logic added to them
	// as described by Merrill et al. See
	// http://www.cs.virginia.edu/~dgm4d/
	reduce = clCreateKernel(gpuprog, "reduce", &err);
	CL_CHECK_ERROR(err);
	top_scan = clCreateKernel(gpuprog, "top_scan", &err);
	CL_CHECK_ERROR(err);
	bottom_scan = clCreateKernel(gpuprog, "bottom_scan", &err);
	CL_CHECK_ERROR(err);
	cpu_reduce = clCreateKernel(cpuprog, "reduce", &err);
	CL_CHECK_ERROR(err);
	cpu_top_scan = clCreateKernel(cpuprog, "top_scan", &err);
	CL_CHECK_ERROR(err);
	cpu_bottom_scan = clCreateKernel(cpuprog, "bottom_scan", &err);
	CL_CHECK_ERROR(err);
}


int main(int argc, char** argv) {

	if(argc > 1){
		gpu_work = atoi(argv[1]);
	}

	printf("gpu_work = %d\n",gpu_work);

	//LOOP START
	int loop;

	initOpenCL();

	for(loop=0; loop<ITERACTIONS; loop++) {


		runTest<unsigned int>(loop);
	}
}
