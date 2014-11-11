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


#define ITERACTIONS 10000		 //loop

using namespace std;

cl_platform_id          platform_id[100];
cl_device_id            device_id[100];
cl_context              context;
cl_command_queue        command_queue;
cl_program              program;

char file_name[60];
char file_name_log[60];

// Returns the current system time in microseconds
long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}


bool verifySort(unsigned int *keys, const size_t size) {
	bool passed = true;

	int i;
	for (i = 0; i < size - 1; i++) {
		if (keys[i] > keys[i + 1]) {
			passed = false;
		}
	}
	cout << "Test ";
	if (passed) {
		cout << "Passed" << endl;

		cout << "Saving output GOLD" << endl;
		FILE *fp;
		if( (fp = fopen("output_sort", "wb" )) == 0 )
			printf( "The file output_sort was not opened\n");
		for (i = 0; i < size; i++) {
			fwrite(&(keys[i]), 1, sizeof(unsigned int), fp);
		}
		fclose(fp);
	}
	else
		cout << "---FAILED---" << endl;

	return passed;
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


inline size_t
getMaxWorkGroupSize (cl_context &ctx, cl_kernel &ker, cl_device_id devid) {
	int err;
	// Find the maximum work group size
	size_t retSize = 0;
	size_t maxGroupSize = 0;
	// we must find the device asociated with this context first
	//cl_device_id devid;   // we create contexts with a single device only
	//err = clGetContextInfo (ctx, CL_CONTEXT_DEVICES, sizeof(devid), &devid, &retSize);
	//CL_CHECK_ERROR(err);
	//if (retSize < sizeof(devid))  // we did not get any device, pass 0 to the function
	//   devid = 0;
	err = clGetKernelWorkGroupInfo (ker, devid, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
		&maxGroupSize, &retSize);
	CL_CHECK_ERROR(err);
	return (maxGroupSize);
}


template <class T>
void runTest(cl_device_id dev, cl_context ctx, cl_command_queue queue) {
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
	cl_program prog = clCreateProgramWithSource(ctx,
		1,
		(const char **) &source_str,
		NULL,
		&err);
	CL_CHECK_ERROR(err);

	// Before proceeding, make sure the kernel code compiles and
	// all kernels are valid.
	//cout << "Compiling sort kernels." << endl;
	err = clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
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
	cl_kernel reduce = clCreateKernel(prog, "reduce", &err);
	CL_CHECK_ERROR(err);

	cl_kernel top_scan = clCreateKernel(prog, "top_scan", &err);
	CL_CHECK_ERROR(err);

	cl_kernel bottom_scan = clCreateKernel(prog, "bottom_scan", &err);
	CL_CHECK_ERROR(err);

	// If the device doesn't support at least 256 work items in a
	// group, use a different kernel (TODO)
	if ( getMaxWorkGroupSize(ctx, reduce, dev)      < 256 ||
		getMaxWorkGroupSize(ctx, top_scan, dev)    < 256 ||
	getMaxWorkGroupSize(ctx, bottom_scan, dev) < 256) {
		cout << "Sort requires a device that supports a work group size " <<
			"of at least 256" << endl;
		exit(0);
	}

	// Problem Sizes
	int probSizes[7] = { 1, 8, 32, 64 , 128, 256, 512};
	int size = probSizes[5];

	// Convert to MiB
	size = (size * 1024 * 1024) / sizeof(T);

	// Create input data on CPU
	unsigned int bytes = size * sizeof(T);

	// Allocate pinned host memory for input data (h_idata)
	cl_mem h_i = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		bytes, NULL, &err);
	CL_CHECK_ERROR(err);
	T* h_idata = (T*)clEnqueueMapBuffer(queue, h_i, true,
		CL_MAP_READ|CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
	CL_CHECK_ERROR(err);

	// Allocate pinned host memory for output data (h_odata)
	cl_mem h_o = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		bytes, NULL, &err);
	CL_CHECK_ERROR(err);
	T* h_odata = (T*)clEnqueueMapBuffer(queue, h_o, true,
		CL_MAP_READ|CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
	CL_CHECK_ERROR(err);

	FILE *fp;
	if( (fp = fopen("input_sort", "wb" )) == 0 )
		printf( "The file input_sort was not opened\n");
	// Initialize host memory
	cout << "Initializing host memory." << endl;
	for (int i = 0; i < size; i++) {
		h_idata[i] = rand();	 // Fill with some pattern
		fwrite(&(h_idata[i]), 1, sizeof(T), fp);

		h_odata[i] = -1;
	}
	fclose(fp);

	// The radix width in bits
	const int radix_width = 4;	 // Changing this requires major kernel updates
								 // n possible digits
	const int num_digits = (int)pow((double)2, radix_width);

	// Allocate device memory for input array
	cl_mem d_idata = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, NULL, &err);
	CL_CHECK_ERROR(err);

	// Allocate device memory for output array
	cl_mem d_odata = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, NULL, &err);
	CL_CHECK_ERROR(err);

	// Number of local work items per group
	const size_t local_wsize  = 256;

	// Number of global work items
								 // i.e. 64 work groups
	const size_t global_wsize = 16384;
	const size_t num_work_groups = global_wsize / local_wsize;

	// Allocate device memory for local work group intermediate sums
	cl_mem d_isums = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
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

	// Set the kernel arguments for the top-level scan
	err = clSetKernelArg(top_scan, 0, sizeof(cl_mem), (void*)&d_isums);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(top_scan, 1, sizeof(cl_int), (void*)&num_work_groups);
	CL_CHECK_ERROR(err);
	err = clSetKernelArg(top_scan, 2, local_wsize * 2 * sizeof(T), NULL);
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

	// Copy data to GPU
	cout << "Copying input data to device." << endl;
	err = clEnqueueWriteBuffer(queue, d_idata, true, 0, bytes, h_idata, 0,
		NULL, NULL);
	CL_CHECK_ERROR(err);
	err = clFinish(queue);
	CL_CHECK_ERROR(err);


	cout << "Running benchmark with size " << size << endl;

	long long time0, time1;
	time0 = get_time();

	for (int shift = 0; shift < sizeof(T)*8; shift += radix_width) {
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
		bool even = ((shift / radix_width) % 2 == 0) ? true : false;

		if (even) {
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
		err = clEnqueueNDRangeKernel(queue, reduce, 1, NULL,
			&global_wsize, &local_wsize, 0, NULL, NULL);

		// Next, a top-level exclusive scan is performed on the
		// per block histograms.  This is done by a single
		// work group (note global size here is the same as local).
		err = clEnqueueNDRangeKernel(queue, top_scan, 1, NULL,
			&local_wsize, &local_wsize, 0, NULL, NULL);

		// Finally, a bottom-level scan is performed by each block
		// that is seeded with the scanned histograms which rebins,
		// locally scans, then scatters keys to global memory
		err = clEnqueueNDRangeKernel(queue, bottom_scan, 1, NULL,
			&global_wsize, &local_wsize, 0, NULL, NULL);
	}
	err = clFinish(queue);
	CL_CHECK_ERROR(err);

	time1 = get_time();
	double kernel_time = (double) (time1-time0) / 1000000;
	printf("\nkernel time: %.12f\n", kernel_time);

	err = clEnqueueReadBuffer(queue, d_idata, true, 0, bytes, h_odata,
		0, NULL, NULL);
	CL_CHECK_ERROR(err);
	err = clFinish(queue);
	CL_CHECK_ERROR(err);

	// If answer is incorrect, stop test and do not report performance
	if (! verifySort(h_odata, size)) {
		return;
	}


	// Clean up device memory
	err = clReleaseMemObject(d_idata);
	CL_CHECK_ERROR(err);
	err = clReleaseMemObject(d_odata);
	CL_CHECK_ERROR(err);
	err = clReleaseMemObject(d_isums);
	CL_CHECK_ERROR(err);

	// Clean up pinned host memory
	err = clEnqueueUnmapMemObject(queue, h_i, h_idata, 0, NULL, NULL);
	CL_CHECK_ERROR(err);
	err = clEnqueueUnmapMemObject(queue, h_o, h_odata, 0, NULL, NULL);
	CL_CHECK_ERROR(err);
	err = clReleaseMemObject(h_i);
	CL_CHECK_ERROR(err);
	err = clReleaseMemObject(h_o);
	CL_CHECK_ERROR(err);

	// Clean up program and kernel objects
	err = clReleaseProgram(prog);
	CL_CHECK_ERROR(err);
	err = clReleaseKernel(reduce);
	CL_CHECK_ERROR(err);
	err = clReleaseKernel(top_scan);
	CL_CHECK_ERROR(err);
	err = clReleaseKernel(bottom_scan);
	CL_CHECK_ERROR(err);
}


void getDevices(cl_device_type deviceType) {
	cl_uint         platforms_n = 0;
	cl_uint         devices_n   = 0;
	cl_int                  ret;

	clGetPlatformIDs(100, platform_id, &platforms_n);
	clGetDeviceIDs(platform_id[0], deviceType, 100, device_id, &devices_n);

	// Create an OpenCL context.
	context = clCreateContext(NULL, devices_n, device_id, NULL, NULL, &ret);
	if (ret != CL_SUCCESS) {
		printf("\nError at clCreateContext! Error code %i\n\n", ret);
		exit(1);
	}

	// Create a command queue.
	command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);
	if (ret != CL_SUCCESS) {
		printf("\nError at clCreateCommandQueue! Error code %i\n\n", ret);
		exit(1);
	}
}


int main(int argc, char** argv) {


	/*
	CL_DEVICE_TYPE_DEFAULT
	CL_DEVICE_TYPE_CPU
	CL_DEVICE_TYPE_GPU
	CL_DEVICE_TYPE_ACCELERATOR
	CL_DEVICE_TYPE_ALL
	*/
	getDevices(CL_DEVICE_TYPE_GPU);

	runTest<unsigned int>(device_id[0], context, command_queue);

}
