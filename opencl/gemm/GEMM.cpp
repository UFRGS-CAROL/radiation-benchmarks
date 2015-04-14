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

#ifdef LOGS
#include "/home/carol/log_helper/log_helper.h"
#endif /* LOGS */

#define N 1024

#define GOLD_MATRIX_PATH "/home/carol/TestGPU/GenerateGoldMatrix/Double_GOLD_1024.matrix"
#define LOGFILE_MATRIXNAME "openclGEMM1024"
// #define PLATFORM_ID 0
#define DEVICE_ID 0
#define SWITCH_CHAR  '-'
#define N_ERRORS_LOG 500
#define ITERATIONS 1000000000000

char kernel_gemmN_path [] = "/home/carol/DSN15_codes/openclgemm/gemmN.cl";

using namespace std;

inline long long get_time() {
     struct timeval tv;
     gettimeofday(&tv, NULL);
     return (tv.tv_sec * 1000000) + tv.tv_usec;
}

void ReadMatrixFromFile(double *h_A, double *h_B, double *h_GOLD)
{
    FILE *f_A, *f_B, *f_GOLD;

    f_A = fopen("/home/carol/TestGPU/GenerateGoldMatrix/Double_B_8192.matrix", "rb");
    f_B = fopen("/home/carol/TestGPU/GenerateGoldMatrix/Double_A_8192.matrix", "rb");
    f_GOLD = fopen(GOLD_MATRIX_PATH, "rb");

    if (!(f_A && f_B && f_GOLD)) {
        printf("Error opening matrix.\n");
        exit(-1);
    }

    printf("read...");
    fread(h_A, sizeof(double)*N*N, 1, f_A);
    fread(h_B, sizeof(double)*N*N, 1, f_B);
    fread(h_GOLD, sizeof(double)*N*N, 1, f_GOLD);

    fclose(f_A);
    fclose(f_B);
    fclose(f_GOLD);


}

#define CL_BAIL_ON_ERROR(err) \
{                             \
    CL_CHECK_ERROR(err);      \
    if (err != CL_SUCCESS)    \
        return;               \
}

// Forward declaration
template <class T> inline std::string toString (const T& t) {
    std::stringstream ss;
    ss << t;
    return ss.str();
}

template <class T>
void runTest(const string& testName, cl_device_id dev, cl_context ctx,
             cl_command_queue queue, const string& compileFlags);

void
RunBenchmark(cl_device_id dev, cl_context ctx, cl_command_queue queue);

int main()
{
    // =========> OpenCl vars
    cl_context clGPUContext;
    cl_command_queue clCommandQue;
    cl_int numplat;
    cl_platform_id platforms[5];
    cl_int errcode;

    size_t dataBytes;

    clGetPlatformIDs(2, platforms, (cl_uint*)&numplat);
    std::cout << "OpenCL Platforms available  : " << numplat << "\n";
    // This returns a platform list available on the system, on my system this means:
    //	[0] = Intel Core i7 (4Cores, 8Logical threads) / OpenCL1.2
    //			Intel HD4600 Integrated Graphics / OpenCL1.2
    //	[1] = NVIDIA GTX850m Dedicated Graphics / CUDA 6.5 / OpenCL1.1


    // Setup OpenCL context and device for NVIDIA GTX850m
    cl_context_properties props[3];
    props[0] = (cl_context_properties)CL_CONTEXT_PLATFORM;  // indicates that next element is platform
    props[1] = (cl_context_properties)platforms[numplat - 1];  // platform is of type cl_platform_id
    props[2] = (cl_context_properties)0;   // last element must be 0

    clGPUContext = clCreateContextFromType(props,
                                           CL_DEVICE_TYPE_GPU,				// It could be CL_DEVICE_TYPE_ALL as on the selected platform this is the only computing device, btw...
                                           NULL, NULL, &errcode);
    if (errcode != CL_SUCCESS) std::cout << "error clCreateContextFromType : " << errcode << "\n";

    // get the list of GPU devices associated
    // with context

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

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Benchmarks the GEMM codes
//
// Arguments:
//   dev: the opencl device id to use for the benchmark
//   ctx: the opencl context to use for the benchmark
//   queue: the opencl command queue to issue commands to
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
//
// Returns:  nothing
//
// Programmer: Anthony Danalis
// Creation: August 26, 2009
//
// Modifications:
//   Jeremy Meredith, Thu Sep 24 17:30:18 EDT 2009
//   Use implicit include of source file instead of runtime loading.
//
//   Jeremy Meredith, Thu Aug 19 13:59:09 EDT 2010
//   Added transfer vs computation equivalence calculation.
//
//   Jeremy Meredith, Thu Aug 19 14:16:49 EDT 2010
//   Use pinned memory for better PCIe speeds.
//
// ****************************************************************************

void
RunBenchmark(cl_device_id dev,
             cl_context ctx,
             cl_command_queue queue)
{
    // OpenCL doesn't support templated kernels, so we have to use macros

    runTest<double>("DGEMM", dev, ctx, queue,
                    "-DK_DOUBLE_PRECISION ");
}

template <class T>
void runTest(const string& testName, cl_device_id dev, cl_context ctx,
             cl_command_queue queue, const string& compileFlags)
{

    double timeG;

    cl_int err;
    int waitForEvents = 1;
    size_t m = N, n = N, k = N;
    size_t lda, ldb, ldc;
    const T alpha = 1;
    const T beta = -1;
    int i, j;

    lda = ldb = ldc = N;

#ifdef LOGS
    // Log files
    FILE* file;
    FILE* log_file;
    FILE* timefile;
#endif /* LOGS */

    int loop2;

    int ea = 0; //wrong integers in the current loop
    int t_ea = 0; //total number of wrong integers
    int old_ea = 0;

    double total_time = 0.0;



    char test_info[100];
    snprintf(test_info, 100, "size:%dx%d",N,N);
    start_log_file(LOGFILE_MATRIXNAME, test_info);
    set_max_errors_iter(200);
    set_iter_interval_print(5);

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

    cl_kernel goldChk = clCreateKernel(prog, "GoldChk", &err);
    CL_CHECK_ERROR(err);

    // Allocate memory for the matrices
    T *A, *B, *C, *GOLD;
    int *kerrors;
    cl_mem Aobj, Bobj, Cobj, GOLDobj, kerrorsobj;
    if (true) // pinned
    {
        Aobj = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                              sizeof(T)*N*N, NULL, &err);
        CL_CHECK_ERROR(err);
        A =(T*)clEnqueueMapBuffer(queue,Aobj,true,CL_MAP_READ|CL_MAP_WRITE,
                                  0,sizeof(T)*N*N,0, NULL,NULL,&err);
        CL_CHECK_ERROR(err);

        Bobj = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                              sizeof(T)*N*N, NULL, &err);
        CL_CHECK_ERROR(err);
        B =(T*)clEnqueueMapBuffer(queue,Bobj,true,CL_MAP_READ|CL_MAP_WRITE,
                                  0,sizeof(T)*N*N,0, NULL,NULL,&err);
        CL_CHECK_ERROR(err);

        Cobj = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                              sizeof(T)*N*N, NULL, &err);
        CL_CHECK_ERROR(err);
        C =(T*)clEnqueueMapBuffer(queue,Cobj,true,CL_MAP_READ|CL_MAP_WRITE,
                                  0,sizeof(T)*N*N,0, NULL,NULL,&err);
        CL_CHECK_ERROR(err);

        GOLDobj = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                 sizeof(T)*N*N, NULL, &err);
        CL_CHECK_ERROR(err);
        GOLD = (T*)clEnqueueMapBuffer(queue, GOLDobj, true, CL_MAP_READ | CL_MAP_WRITE,
                                      0, sizeof(T)*N*N, 0, NULL, NULL, &err);
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
        A = (T*)malloc( N*N*sizeof( T ) );
        B = (T*)malloc( N*N*sizeof( T ) );
        C = (T*)malloc( N*N*sizeof( T ) );
    }

    cout << "Allocating GPU and Host memory...";
    clFinish(queue);
    cout << "Done\n";

    ReadMatrixFromFile(A, B, GOLD);

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

    // Pass arguments to the goldChk kernel
    err = clSetKernelArg(goldChk, 0, sizeof(cl_mem), (void*)&Agpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(goldChk, 1, sizeof(cl_mem), (void*)&Cgpu);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(goldChk, 2, sizeof(int), (void*)&m);
    CL_BAIL_ON_ERROR(err);
    err = clSetKernelArg(goldChk, 3, sizeof(cl_mem), (void*)&kerrorsgpu);
    CL_BAIL_ON_ERROR(err);

    const size_t globalWorkSize[2] = {m/4,n/4};


    // Run NN
    for (int it = 0; it < ITERATIONS; it++) {


        err = clEnqueueWriteBuffer(queue, Agpu, CL_TRUE, 0, m*n*sizeof(T),
                                   A, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(queue, Bgpu, CL_TRUE, 0, m*n*sizeof(T),
                                   B, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(queue, Cgpu, CL_TRUE, 0, m*n*sizeof(T),
                                   C, 0, NULL, NULL);


        clFinish(queue);
        CL_BAIL_ON_ERROR(err);


#ifdef LOGS
        start_iteration();
#endif /* LOGS */

	long long k_start = get_time();

        //Launch Kernels
        err = clEnqueueNDRangeKernel(queue, sgemmNN, 2, NULL, globalWorkSize,
                                     localWorkSize, 0, NULL, NULL);

        clFinish(queue);
        CL_BAIL_ON_ERROR(err);


	double kernel_time = (double) (get_time() - it_time_start) / 1000000;
	double flops = 2.0*(double)N*N*N;
	double gflops = flops / kernel_time;
	printf("GFLOPS: %lf\n",gflops);

#ifdef LOGS
        end_iteration();
#endif /* LOGS */

        // Check Gold
        *kerrors = 0;

        err = clEnqueueWriteBuffer(queue, Agpu, CL_TRUE, 0, m*n*sizeof(T),
                                   GOLD, 0, NULL, NULL);
        CL_BAIL_ON_ERROR(err);
        err = clEnqueueWriteBuffer(queue, kerrorsgpu, CL_TRUE, 0, sizeof(int),
                                   kerrors, 0, NULL, NULL);
        CL_BAIL_ON_ERROR(err);


        clFinish(queue);
        CL_BAIL_ON_ERROR(err);


        //Launch Kernels
        err = clEnqueueNDRangeKernel(queue, goldChk, 2, NULL, globalWorkSize,
                                     localWorkSize, 0, NULL, NULL);

        clFinish(queue);
        CL_BAIL_ON_ERROR(err);



        err = clEnqueueReadBuffer(queue, kerrorsgpu, CL_TRUE, 0, sizeof(int),
                                  kerrors, 0, NULL, NULL);
        CL_BAIL_ON_ERROR(err);
        clFinish(queue);
        CL_BAIL_ON_ERROR(err);

        if (it==1) cout << "Errors : " << *kerrors << "\n";

        ea = 0;
        t_ea += *kerrors;

        log_error_count(*kerrors);

        if (*kerrors>0)
        {
            std::cout << "Error detected! kerrors = " << *kerrors << "\n";

            err = clEnqueueReadBuffer(queue, Cgpu, CL_TRUE, 0, m*n*sizeof(T),
                                      C, 0, NULL, NULL);
            CL_BAIL_ON_ERROR(err);
            clFinish(queue);
            CL_BAIL_ON_ERROR(err);

            char error_detail[150];
            for (int i = 0; (i<N) && (ea < N_ERRORS_LOG); i++)
            {
                for (int j = 0; (j<N) && (ea < N_ERRORS_LOG); j++)
                {
                    if ((fabs((C[i + N*j] - GOLD[i + N*j]) / C[i + N*j]) > 0.0000000001) || (fabs((C[i + N*j] - GOLD[i + N*j]) / GOLD[i + N*j]) > 0.0000000001))
                    {
                        //fprintf(file, "\n p: [%d, %d], r: %1.16e, e: %1.16e, error: %d\n", i, j, C[i + N * j], GOLD[i + N * j], ea);
                        snprintf(error_detail, 150, "p: [%d, %d], r: %1.16e, e: %1.16e", i, j, A[i + N * j], GOLD[i + N * j]);
#ifdef LOGS
                        log_error_detail(error_detail);
#endif /* LOGS */
                    }
                }
            }


            ReadMatrixFromFile(A, B, GOLD);
        }
        if(*kerrors > 0 || (it % 10 == 0))
        {
            printf("\ntest number: %d", it);
            printf("\nerrors: %d", *kerrors);
        }
    }

#ifdef LOGS
    end_log_file();
#endif /* LOGS */


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
    err = clReleaseKernel(goldChk);
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
