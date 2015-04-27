#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include <iostream>
#include <fstream>
#include <vector>

#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_ext.h>

using namespace std;


bool scanCPU(double *data, double* reference, double* dev_result, const size_t size)
{

    bool passed = true;

    double last = 0.0f;
    for (unsigned int i = 0; i < size; ++i)
    {
        reference[i] = data[i] + last;
        last = reference[i];
    }
    for (unsigned int i = 0; i < size; ++i)
    {
        if (reference[i] != dev_result[i])
        {
            cout << "Mismatch at i: " << i << " ref: " << reference[i]
                 << " dev: " << dev_result[i] << endl;
            passed = false;
        }
    }
    return passed;
}



extern const char *cl_source_scan;



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

inline
bool
checkExtension( cl_device_id devID, const std::string& ext )
{
    cl_int err;

    size_t nBytesNeeded = 0;
    err = clGetDeviceInfo( devID,
                        CL_DEVICE_EXTENSIONS,
                        0,
                        NULL,
                        &nBytesNeeded );
    CL_CHECK_ERROR(err);
    char* extensions = new char[nBytesNeeded+1];
    err = clGetDeviceInfo( devID,
                        CL_DEVICE_EXTENSIONS,
                        nBytesNeeded + 1,
                        extensions,
                        NULL );

    std::string extString = extensions;
    delete[] extensions;

    return (extString.find(ext) != std::string::npos);
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

int main()
{
    cl_int err = 0;

    // Initialize OpenCL
    getDevices(CL_DEVICE_TYPE_ALL);

    // Program Setup
    cl_program prog = clCreateProgramWithSource(context,
                                                1,
                                                &cl_source_scan,
                                                NULL,
                                                &err);
    CL_CHECK_ERROR(err);

    string compileFlags;
    if (checkExtension(device_id[0], "cl_khr_fp64"))
    {
        compileFlags = "-DK_DOUBLE_PRECISION ";
    }
    else if (checkExtension(device_id[0], "cl_amd_fp64"))
    {
        compileFlags = "-DAMD_DOUBLE_PRECISION ";
    }
    // Before proceeding, make sure the kernel code compiles and
    // all kernels are valid.
    cout << "Compiling scan kernels." << endl;
    err = clBuildProgram(prog, 1, &device_id[0], compileFlags.c_str(), NULL, NULL);
    CL_CHECK_ERROR(err);

    if (err != CL_SUCCESS)
    {
        char log[5000];
        size_t retsize = 0;
        err = clGetProgramBuildInfo(prog, device_id[0], CL_PROGRAM_BUILD_LOG, 5000
                * sizeof(char), log, &retsize);

        CL_CHECK_ERROR(err);
        cout << "Build error." << endl;
        cout << "Retsize: " << retsize << endl;
        cout << "Log: " << log << endl;
        return 1;
    }

    // Extract out the 3 kernels
    cl_kernel reduce = clCreateKernel(prog, "reduce", &err);
    CL_CHECK_ERROR(err);

    cl_kernel top_scan = clCreateKernel(prog, "top_scan", &err);
    CL_CHECK_ERROR(err);

    cl_kernel bottom_scan = clCreateKernel(prog, "bottom_scan", &err);
    CL_CHECK_ERROR(err);


    // Problem Sizes
    int probSizes[7] = { 1, 8, 32, 64, 128, 256, 512 };
    int size = probSizes[6];

    // Convert to MB
    size = (size * 1024 * 1024) / sizeof(double);

    // Create input data on CPU
    unsigned int bytes = size * sizeof(double);
    double* reference = new double[size];

    // Allocate pinned host memory for input data (h_idata)
    cl_mem h_i = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            bytes, NULL, &err);
    CL_CHECK_ERROR(err);
    double* h_idata = (double*)clEnqueueMapBuffer(command_queue, h_i, true,
            CL_MAP_READ|CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
    CL_CHECK_ERROR(err);

    // Allocate pinned host memory for output data (h_odata)
    cl_mem h_o = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            bytes, NULL, &err);
    CL_CHECK_ERROR(err);
    double* h_odata = (double*)clEnqueueMapBuffer(command_queue, h_o, true,
            CL_MAP_READ|CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
    CL_CHECK_ERROR(err);

    // Initialize host memory
    cout << "Initializing host memory." << endl;
    for (int i = 0; i < size; i++)
    {
        h_idata[i] = i % 3; //Fill with some pattern
        h_odata[i] = -1;
    }

    // Allocate device memory for input array
    cl_mem d_idata = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    CL_CHECK_ERROR(err);

    // Allocate device memory for output array
    cl_mem d_odata = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    CL_CHECK_ERROR(err);

    // Number of local work items per group
    const size_t local_wsize  = 256;

    // Number of global work items
    const size_t global_wsize = 16384; // i.e. 64 work groups
    const size_t num_work_groups = global_wsize / local_wsize;

    // Allocate device memory for local work group intermediate sums
    cl_mem d_isums = clCreateBuffer(context, CL_MEM_READ_WRITE,
            num_work_groups * sizeof(double), NULL, &err);
    CL_CHECK_ERROR(err);

    // Set the kernel arguments for the reduction kernel
    err = clSetKernelArg(reduce, 0, sizeof(cl_mem), (void*)&d_idata);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(reduce, 1, sizeof(cl_mem), (void*)&d_isums);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(reduce, 2, sizeof(cl_int), (void*)&size);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(reduce, 3, local_wsize * sizeof(double), NULL);
    CL_CHECK_ERROR(err);

    // Set the kernel arguments for the top-level scan
    err = clSetKernelArg(top_scan, 0, sizeof(cl_mem), (void*)&d_isums);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(top_scan, 1, sizeof(cl_int), (void*)&num_work_groups);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(top_scan, 2, local_wsize * 2 * sizeof(double), NULL);
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
    err = clSetKernelArg(bottom_scan, 4, local_wsize * 2 * sizeof(double), NULL);
    CL_CHECK_ERROR(err);

    // Copy data to GPU
    cout << "Copying input data to device." << endl;
    err = clEnqueueWriteBuffer(command_queue, d_idata, true, 0, bytes, h_idata, 0,
            NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clFinish(command_queue);
    CL_CHECK_ERROR(err);

    // Repeat the test multiplie times to get a good measurement
    int passes = 1;
    int iters  = 1;

    double tk1, tk2, tk3;
    cout << "Running benchmark with size " << size << endl;
    for (int k = 0; k < passes; k++)
    {
        double th = mysecond();
        for (int j = 0; j < iters; j++)
        {
            // For scan, we use a reduce-then-scan approach

            // Each thread block gets an equal portion of the
            // input array, and computes the sum.
            tk1 = mysecond();
            err = clEnqueueNDRangeKernel(command_queue, reduce, 1, NULL,
                        &global_wsize, &local_wsize, 0, NULL, NULL);
            CL_CHECK_ERROR(err);
            clFinish(command_queue);
            tk1 = mysecond() - tk1;

            // Next, a top-level exclusive scan is performed on the array
            // of block sums
            tk2 = mysecond();
            err = clEnqueueNDRangeKernel(command_queue, top_scan, 1, NULL,
                        &local_wsize, &local_wsize, 0, NULL, NULL);
            CL_CHECK_ERROR(err);
            clFinish(command_queue);
            tk2 = mysecond() - tk2;

            // Finally, a bottom-level scan is performed by each block
            // that is seeded with the scanned value in block sums
            tk3 = mysecond();
            err = clEnqueueNDRangeKernel(command_queue, bottom_scan, 1, NULL,
                        &global_wsize, &local_wsize, 0, NULL, NULL);
            CL_CHECK_ERROR(err);
            clFinish(command_queue);
            tk3 = mysecond() - tk3;
        }
        err = clFinish(command_queue);
        CL_CHECK_ERROR(err);
        double totalScanTime = mysecond() - th;

        err = clEnqueueReadBuffer(command_queue, d_odata, true, 0, bytes, h_odata,
                0, NULL, NULL);
        CL_CHECK_ERROR(err);
        err = clFinish(command_queue);
        CL_CHECK_ERROR(err);

        // If answer is incorrect, stop test and do not report performance
        if (! scanCPU(h_idata, reference, h_odata, size))
        {
            printf("Error computing scan, incorrect answer\n");
            return 1;
        }

        printf("reduce time: %f\n", tk1);
        printf("top-scan time: %f\n", tk2);
        printf("bottom-scan time: %f\n", tk3);

        char atts[1024];
        double avgTime = totalScanTime / (double) iters;
        double gbs = (double) (size * sizeof(double)) / (1000. * 1000. * 1000.);
        cout << "kernel time: " << avgTime << "\ngbs: " << gbs << " GB/s\n";
    }

    // Clean up device memory
    err = clReleaseMemObject(d_idata);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_odata);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(d_isums);
    CL_CHECK_ERROR(err);

    // Clean up pinned host memory
    err = clEnqueueUnmapMemObject(command_queue, h_i, h_idata, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clEnqueueUnmapMemObject(command_queue, h_o, h_odata, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(h_i);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(h_o);
    CL_CHECK_ERROR(err);

    // Clean up other host memory
    delete[] reference;

    err = clReleaseProgram(prog);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(reduce);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(top_scan);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(bottom_scan);
    CL_CHECK_ERROR(err);
}

