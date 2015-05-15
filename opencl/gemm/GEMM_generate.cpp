#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cassert>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

#include <CL/cl.h>

#include "support.h"

#define DEVICE_ID 0
#define ITERATIONS 1000000000000

char *kernel_gemmN_path;
char *gold_matrix, *a_matrix, *b_matrix;
int input_size;

using namespace std;

inline long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

//void ReadMatrixFromFile(double *h_A, double *h_B, double *h_GOLD)
void ReadMatrixFromFile(double *h_A, double *h_B)
{
    FILE *f_A, *f_B, *f_GOLD;

    f_A = fopen(a_matrix, "rb");
    f_B = fopen(b_matrix, "rb");
//   f_GOLD = fopen(gold_matrix, "rb");

    if (!(f_A && f_B )) { //&& f_GOLD)) {
        printf("Error opening matrix.\n");
        exit(-1);
    }

    printf("read...");
    fread(h_A, sizeof(double)*input_size*input_size, 1, f_A);
    fread(h_B, sizeof(double)*input_size*input_size, 1, f_B);
    // fread(h_GOLD, sizeof(double)*input_size*input_size, 1, f_GOLD);

    fclose(f_A);
    fclose(f_B);
    // fclose(f_GOLD);


}

#define CL_BAIL_ON_ERROR(err) \
{                             \
    CL_CHECK_ERROR(err);      \
    if (err != CL_SUCCESS)    \
        return;               \
}


template <class T>
void runTest(const string& testName, cl_device_id dev, cl_context ctx,
             cl_command_queue queue, const string& compileFlags);

void
RunBenchmark(cl_device_id dev, cl_context ctx, cl_command_queue queue);

void usage() {
    printf("Usage: gemm <input_size> <cl_device_type> <kernel_file> <A_MATRIX> <B_MATRIX> <GOLD_MATRIX_TO_GENERATE> \n");
    printf("  cl_device_types\n");
    printf("    Default: %d\n",CL_DEVICE_TYPE_DEFAULT);
    printf("    CPU: %d\n",CL_DEVICE_TYPE_CPU);
    printf("    GPU: %d\n",CL_DEVICE_TYPE_GPU);
    printf("    ACCELERATOR: %d\n",CL_DEVICE_TYPE_ACCELERATOR);
    printf("    ALL: %d\n",CL_DEVICE_TYPE_ALL);
}

int main(int argc, char ** argv)
{
    int devType;
    if(argc == 7) {
        input_size = atoi(argv[1]);
        devType = atoi(argv[2]);
        kernel_gemmN_path = argv[3];
        a_matrix = argv[4];
        b_matrix = argv[5];
        gold_matrix = argv[6];
    } else {
        usage();
        exit(1);
    }

    // =========> OpenCl vars
    cl_context clGPUContext;
    cl_command_queue clCommandQue;
    cl_int numplat;
    cl_platform_id platforms[5];
    cl_int errcode;

    size_t dataBytes;

    clGetPlatformIDs(2, platforms, (cl_uint*)&numplat);
    std::cout << "OpenCL Platforms available  : " << numplat << "\n";

    // Setup OpenCL context and device for NVIDIA GTX850m
    cl_context_properties props[3];
    props[0] = (cl_context_properties)CL_CONTEXT_PLATFORM;  // indicates that next element is platform
    props[1] = (cl_context_properties)platforms[numplat - 1];  // platform is of type cl_platform_id
    props[2] = (cl_context_properties)0;   // last element must be 0

    clGPUContext = clCreateContextFromType(props,
                                           devType,				// It could be CL_DEVICE_TYPE_ALL as on the selected platform this is the only computing device, btw...
                                           NULL, NULL, &errcode);
    if (errcode != CL_SUCCESS) std::cout << "error clCreateContextFromType : " << errcode << "\n";


    cl_int num_devices;
    errcode = clGetContextInfo(clGPUContext, // This just to show how many devices are avail on the platform
                               CL_CONTEXT_NUM_DEVICES, sizeof(cl_int),
                               &num_devices, NULL);
    std::cout << "Number of devices: " << num_devices << "\n";
    errcode = clGetContextInfo(clGPUContext,
                               CL_CONTEXT_DEVICES, 0, NULL,
                               &dataBytes);
    cl_device_id *clDevices = (cl_device_id *)
                              malloc(dataBytes);
    errcode |= clGetContextInfo(clGPUContext,
                                CL_CONTEXT_DEVICES, dataBytes,
                                clDevices, NULL);

    if (errcode != CL_SUCCESS) std::cout << "error clGetContextInfo : " << errcode << "\n";

    char clName[50];

    errcode = clGetDeviceInfo(clDevices[DEVICE_ID],
                              CL_DEVICE_NAME,
                              sizeof(char) * 50,
                              clName,
                              NULL);
    if (errcode != CL_SUCCESS) std::cout << "error clGetDeviceInfo : " << errcode << "\n";
    std::cout << "Device name : " << clName << "\n";

    clCommandQue = clCreateCommandQueue(clGPUContext,
                                        clDevices[DEVICE_ID], 0, &errcode);
    if (errcode != CL_SUCCESS) std::cout << "error clCommandQue : " << errcode << "\n";

    std::cout << "Now calling RunBenchmark\n";

    RunBenchmark(clDevices[DEVICE_ID], clGPUContext, clCommandQue);

}


void
RunBenchmark(cl_device_id dev,
             cl_context ctx,
             cl_command_queue queue)
{
    runTest<double>("DGEMM", dev, ctx, queue,
                    "-DK_DOUBLE_PRECISION ");
}

template <class T>
void runTest(const string& testName, cl_device_id dev, cl_context ctx,
             cl_command_queue queue, const string& compileFlags)
{


    cl_int err;
    int waitForEvents = 1;
    size_t m = input_size, n = input_size, k = input_size;
    size_t lda, ldb, ldc;
    const T alpha = 1;
    const T beta = -1;
    int i, j;

    lda = ldb = ldc = input_size;

    // check
    cl_uint numDimensions = 0;
    clGetDeviceInfo (dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                     sizeof(cl_uint), &numDimensions, NULL);
    size_t *maxWorkSizes = new size_t[numDimensions];
    clGetDeviceInfo (dev, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                     sizeof(size_t)*numDimensions, maxWorkSizes, NULL);

    if (numDimensions<2 || maxWorkSizes[0]<16 || maxWorkSizes[1] < 4)
    {
        cout << "SGEMM needs a 2-dimensional work group size of at least {16,4}." << endl;
        return;
    }

    size_t localWorkSize[2] = {16,4};

    cout << "Ready, build program...";

    std::ifstream kernelfile(kernel_gemmN_path); // This will read the file to the memory as OpenCL needs to compile it from there
    std::string kernelstr((std::istreambuf_iterator<char>(kernelfile)),
                          std::istreambuf_iterator<char>());
    const char* cl_source_gemmN = kernelstr.c_str();

    const size_t kernelLength=kernelstr.length();
    cout << "L:" << kernelLength << endl; //kernelstr.size() << " ";
    // Create program object
    cl_program prog = clCreateProgramWithSource(ctx, 1,
                      &cl_source_gemmN, &kernelLength, &err);
    CL_CHECK_ERROR(err);

    //string flags = compileFlags + " -cl-mad-enable";
    err = clBuildProgram(prog, 0, NULL, "-DK_DOUBLE_PRECISION", NULL,
                         NULL);
    //CL_CHECK_ERROR(err);

    // If compilation fails, print error messages and return
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        //if (err != CL_SUCCESS) {
        char log[5000];
        size_t retsize = 0;
        err =  clGetProgramBuildInfo (prog, dev, CL_PROGRAM_BUILD_LOG,
                                      5000*sizeof(char),  log, &retsize);

        CL_CHECK_ERROR(err);
        cout << "Retsize: " << retsize << endl;
        cout << "Log: " << log << endl;
        exit(-1);
    }
    cout << "Ok\n";

    // Generate the kernel objects
    cl_kernel sgemmNN = clCreateKernel(prog, "sgemmNN", &err);
    CL_CHECK_ERROR(err);

    cl_kernel sgemmNT = clCreateKernel(prog, "sgemmNT", &err);
    CL_CHECK_ERROR(err);

    // Allocate memory for the matrices
    T *A, *B, *C, *GOLD;
    int *kerrors;
    cl_mem Aobj, Bobj, Cobj, GOLDobj, kerrorsobj;
    if (true) // pinned
    {
        Aobj = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                              sizeof(T)*input_size*input_size, NULL, &err);
        CL_CHECK_ERROR(err);
        A =(T*)clEnqueueMapBuffer(queue,Aobj,true,CL_MAP_READ|CL_MAP_WRITE,
                                  0,sizeof(T)*input_size*input_size,0, NULL,NULL,&err);
        CL_CHECK_ERROR(err);

        Bobj = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                              sizeof(T)*input_size*input_size, NULL, &err);
        CL_CHECK_ERROR(err);
        B =(T*)clEnqueueMapBuffer(queue,Bobj,true,CL_MAP_READ|CL_MAP_WRITE,
                                  0,sizeof(T)*input_size*input_size,0, NULL,NULL,&err);
        CL_CHECK_ERROR(err);

        Cobj = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                              sizeof(T)*input_size*input_size, NULL, &err);
        CL_CHECK_ERROR(err);
        C =(T*)clEnqueueMapBuffer(queue,Cobj,true,CL_MAP_READ|CL_MAP_WRITE,
                                  0,sizeof(T)*input_size*input_size,0, NULL,NULL,&err);
        CL_CHECK_ERROR(err);

        GOLDobj = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                 sizeof(T)*input_size*input_size, NULL, &err);
        CL_CHECK_ERROR(err);
        GOLD = (T*)clEnqueueMapBuffer(queue, GOLDobj, true, CL_MAP_READ | CL_MAP_WRITE,
                                      0, sizeof(T)*input_size*input_size, 0, NULL, NULL, &err);
        CL_CHECK_ERROR(err);

        kerrorsobj = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                    sizeof(int), NULL, &err);
        CL_CHECK_ERROR(err);
        kerrors = (int*)clEnqueueMapBuffer(queue, kerrorsobj, true, CL_MAP_READ | CL_MAP_WRITE,
                                           0, sizeof(int), 0, NULL, NULL, &err);
        CL_CHECK_ERROR(err)
    }
    else
    {
        A = (T*)malloc( input_size*input_size*sizeof( T ) );
        B = (T*)malloc( input_size*input_size*sizeof( T ) );
        C = (T*)malloc( input_size*input_size*sizeof( T ) );
    }

    cout << "Allocating GPU and Host memory...";
    clFinish(queue);
    cout << "Done\n";

    ReadMatrixFromFile(A, B);//, GOLD);

    // clean C
    for (i = 0; i<m; ++i) {
        for (j = 0; j<n; ++j) {
            C[i*n + j] = 0.0;
        }
    }

    // Pass A and B to the GPU and create a GPU buffer for C
    cl_mem Agpu = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 m*k * sizeof(T), NULL, &err);
    CL_BAIL_ON_ERROR(err);
    cl_mem Bgpu = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 k*n * sizeof(T), NULL, &err);
    CL_BAIL_ON_ERROR(err);
    cl_mem Cgpu = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 m*n * sizeof(T), NULL, &err);
    CL_BAIL_ON_ERROR(err);
    cl_mem kerrorsgpu = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                       sizeof(int), NULL, &err);
    CL_BAIL_ON_ERROR(err);


    // Set arguments to the sgemmNN kernel
    err = clSetKernelArg(sgemmNN, 0, sizeof(cl_mem), (void*)&Agpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 1, sizeof(int), (void*)&lda);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 2, sizeof(cl_mem), (void*)&Bgpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 3, sizeof(int), (void*)&ldb);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 4, sizeof(cl_mem), (void*)&Cgpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 5, sizeof(int), (void*)&ldc);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 6, sizeof(int), (void*)&k);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 7, sizeof(T), (void*)&alpha);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNN, 8, sizeof(T), (void*)&beta);
    CL_BAIL_ON_ERROR(err);

    // Pass arguments to the sgemmNT kernel
    err = clSetKernelArg(sgemmNT, 0, sizeof(cl_mem), (void*)&Agpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 1, sizeof(int), (void*)&lda);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 2, sizeof(cl_mem), (void*)&Bgpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 3, sizeof(int), (void*)&ldb);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 4, sizeof(cl_mem), (void*)&Cgpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 5, sizeof(int), (void*)&ldc);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 6, sizeof(int), (void*)&k);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 7, sizeof(T), (void*)&alpha);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(sgemmNT, 8, sizeof(T), (void*)&beta);
    CL_BAIL_ON_ERROR(err);


    const size_t globalWorkSize[2] = {m/4,n/4};



    err = clEnqueueWriteBuffer(queue, Agpu, CL_TRUE, 0, m*n*sizeof(T),
                               A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, Bgpu, CL_TRUE, 0, m*n*sizeof(T),
                               B, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, Cgpu, CL_TRUE, 0, m*n*sizeof(T),
                               C, 0, NULL, NULL);


    clFinish(queue);
    CL_BAIL_ON_ERROR(err);


    long long k_start = get_time();

    //Launch Kernels
    err = clEnqueueNDRangeKernel(queue, sgemmNN, 2, NULL, globalWorkSize,
                                 localWorkSize, 0, NULL, NULL);

    clFinish(queue);
    CL_BAIL_ON_ERROR(err);


    double kernel_time = (double) (get_time() - k_start) / 1000000;
    double flops = 2.0*(double)input_size*input_size*input_size;
    double gflops = flops / kernel_time;
    printf("kernel time: %lf\n",kernel_time);
    printf("SIZE:%d FLOPS:%f\n",input_size, gflops);



    std::cout << "Saving GOLD output\n";

    err = clEnqueueReadBuffer(queue, Cgpu, CL_TRUE, 0, m*n*sizeof(T),
                              C, 0, NULL, NULL);
    CL_BAIL_ON_ERROR(err);
    clFinish(queue);
    CL_BAIL_ON_ERROR(err);

    FILE *gold;
    gold = fopen(gold_matrix, "wb");
    if (!(gold) ) { //&& f_GOLD)) {
        printf("Error opening output matrix.\n");
        exit(-1);
    }
    fwrite(C, sizeof(double)*input_size*input_size, 1, gold );
    //for (int i = 0; (i<input_size); i++)
    //{
    //    for (int j = 0; (j<input_size); j++)
    //    {
    //    	fwrite( &C[i + input_size*j], sizeof(double), 1, gold );
    //    }
    //}





    if (true) // pinned
    {
        err = clReleaseMemObject(Aobj);
        CL_CHECK_ERROR(err);
        err = clReleaseMemObject(Bobj);
        CL_CHECK_ERROR(err);
        err = clReleaseMemObject(Cobj);
        CL_CHECK_ERROR(err);
        err = clReleaseMemObject(kerrorsobj);
        CL_CHECK_ERROR(err);
    }
    else
    {
        free(A);
        free(B);
        free(C);
    }

    err = clReleaseProgram(prog);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(sgemmNN);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(sgemmNT);
    CL_CHECK_ERROR(err);

    err = clReleaseMemObject(Agpu);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(Bgpu);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(Cgpu);
    CL_CHECK_ERROR(err);
    err = clReleaseMemObject(kerrorsgpu);
    CL_CHECK_ERROR(err);

}
