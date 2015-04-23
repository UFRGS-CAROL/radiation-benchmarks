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



//#define DEVICE_ID 0

//char kernel_gemmN_path [] = "/home/carol/DSN15_codes/openclgemm/gemmN.cl";
char *kernel_gemmN_path;
int input_size;

using namespace std;


void ReadMatrixFromFile(double *h_A, double *h_B)
{
    FILE *f_A, *f_B;

    f_A = fopen("Double_A_16384.matrix", "rb");
    f_B = fopen("Double_B_16384.matrix", "rb");

    if (!(f_A && f_B)) {
        printf("Error opening matrix.\n");
        exit(-1);
    }

    fread(h_A, sizeof(double)*input_size*input_size, 1, f_A);
    fread(h_B, sizeof(double)*input_size*input_size, 1, f_B);

    fclose(f_A);
    fclose(f_B);


}
void GenerateInputMatrices()
{
    int i, j;
    FILE *f_A, *f_B;

    f_A = fopen("Double_A_16384.matrix", "wb");
    f_B = fopen("Double_B_16384.matrix", "wb");


    srand ( time(NULL) );

    double value;
    for(i=0; i<input_size; i++)
    {
        for(j=0; j<input_size; j++){
            value= (rand()/((double)(RAND_MAX)+1)*(-4.06e16-4.0004e16))+4.1e16;
            fwrite( &value, sizeof(double), 1, f_A );
			        
            value= (rand()/((double)(RAND_MAX)+1)*(-4.06e16-4.4e16))+4.1e16;
            fwrite( &value, sizeof(double), 1, f_B );
        }
    }
    fclose(f_A);
    fclose(f_B);

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

void usage(){
        printf("Usage: generateMatrices <cl_device_type> <kernel_file> \n");
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
    if(argc == 3) {
        devType = atoi(argv[1]);
        kernel_gemmN_path = argv[2];
    } else {
        usage();
        exit(1);
    }

    cl_platform_id          platform_id[100];
    cl_device_id            device_id[100];
    cl_context              context;
    cl_command_queue        command_queue;
    cl_uint         platforms_n = 0;
    cl_uint         devices_n   = 0;
    cl_int                  ret;

    clGetPlatformIDs(100, platform_id, &platforms_n);
    clGetDeviceIDs(platform_id[0], devType, 100, device_id, &devices_n);

    // Create an OpenCL context.
    context = clCreateContext(NULL, devices_n, device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS) {
        printf("\nError at clCreateContext! Error code %i\n\n", ret);
        exit(1);
    }

    printf("Using the default platform (platform 0)...\n\n");
    printf("=== %d OpenCL device(s) found on platform:\n", devices_n);
    for (int i = 0; i < devices_n; i++) {
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
    }
    // Create a command queue.
    command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);
    if (ret != CL_SUCCESS) {
        printf("\nError at clCreateCommandQueue! Error code %i\n\n", ret);
        exit(1);
    }
    RunBenchmark(device_id[0], context, command_queue);

}


void
RunBenchmark(cl_device_id dev,
             cl_context ctx,
             cl_command_queue queue)
{
    // OpenCL doesn't support templated kernels, so we have to use macros

    input_size = 16384;
    GenerateInputMatrices();
    runTest<double>("DGEMM", dev, ctx, queue,
                    "-DK_DOUBLE_PRECISION ");
    input_size = 8192;
    runTest<double>("DGEMM", dev, ctx, queue,
                    "-DK_DOUBLE_PRECISION ");
    input_size = 4096;
    runTest<double>("DGEMM", dev, ctx, queue,
                    "-DK_DOUBLE_PRECISION ");
    input_size = 2048;
    runTest<double>("DGEMM", dev, ctx, queue,
                    "-DK_DOUBLE_PRECISION ");
    input_size = 1024;
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

    cl_kernel goldChk = clCreateKernel(prog, "GoldChk", &err);
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

    ReadMatrixFromFile(A, B);

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




        err = clEnqueueWriteBuffer(queue, Agpu, CL_TRUE, 0, m*n*sizeof(T),
                                   A, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(queue, Bgpu, CL_TRUE, 0, m*n*sizeof(T),
                                   B, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(queue, Cgpu, CL_TRUE, 0, m*n*sizeof(T),
                                   C, 0, NULL, NULL);


        clFinish(queue);
        CL_BAIL_ON_ERROR(err);


        //Launch Kernels
        err = clEnqueueNDRangeKernel(queue, sgemmNN, 2, NULL, globalWorkSize,
                                     localWorkSize, 0, NULL, NULL);

        clFinish(queue);
        CL_BAIL_ON_ERROR(err);




            err = clEnqueueReadBuffer(queue, Cgpu, CL_TRUE, 0, m*n*sizeof(T),
                                      C, 0, NULL, NULL);
            CL_BAIL_ON_ERROR(err);
            clFinish(queue);
            CL_BAIL_ON_ERROR(err);

            for (int i = 0; (i<input_size); i++)
            {
                for (int j = 0; (j<input_size); j++)
                {
		    	// Save GOLD to file;
    			FILE *f_A;

            		char gold_file[150];
                        snprintf(gold_file, 150, "GOLD_%d.matrix",input_size);
    			f_A = fopen(gold_file, "wb");

    			for(i=0; i<input_size; i++)
    			{
    			    for(j=0; j<input_size; j++){
    			        fwrite( &C[i + input_size*j], sizeof(double), 1, f_A );
    			    }
    			}
    			fclose(f_A);
                }
            }

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
