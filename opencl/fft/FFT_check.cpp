//g++ simpleFFT.cpp fftlib.cpp -I/usr/local/cuda-5.5/include/CL/ -L/usr/lib/nvidia-current/ -lOpenCL -o fft

#include <assert.h>
#include <cfloat>
#include <iostream>
#include <sstream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_ext.h>

#include "fftlib.h"

//////////
// LOOPS
//////////
#define ITERACTIONS 10

#define AVOIDZERO 1e-200
#define ACCEPTDIFF 1e-5

cl_platform_id          platform_id[100];
cl_device_id            device_id[100];
cl_context              context;
cl_command_queue        command_queue[2];
cl_program              program;
cl_program		cpuProgram;

using namespace std;

template <class T2> inline bool dp(void);
template <> inline bool dp<cplxflt>(void) {
    return false;
}
template <> inline bool dp<cplxdbl>(void) {
    return true;
}

int t_ea = 0;
int last_num_errors = 0;
int last_num_errors_i = 0;
double total_kernel_time = 0;
int sizeIndex;

// Returns the current system time in microseconds
long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}


cplxdbl* source, * gold;

cl_program fftProg;
cl_program cpufftProg;
cl_kernel fftKrnl, ifftKrnl, chkKrnl;
cl_kernel cpuFftKrnl, cpuIfftKrnl, cpuChkKrnl;
int distribution;

template <class T2> void dump(cl_device_id id,
                              cl_context ctx, cl_command_queue queue[2], int test_number)
{

    void* work;

    double N;
    int i;
    long long time0, time1;
    int kernelIndex = 0;

    unsigned long bytes = 0;

    int probSizes[6] = { 1, 8, 96, 256, 512, 1024};
    //int sizeIndex = 5;

    bytes = probSizes[sizeIndex];

    // Convert to MB
    bytes *= 1024 * 1024;
    bool do_dp = dp<T2>();

    // now determine how much available memory will be used
    int half_n_ffts = bytes / (512*sizeof(T2)*2);
    int n_ffts = half_n_ffts * 2;
    int half_n_cmplx = half_n_ffts * 512;
    unsigned long used_bytes = half_n_cmplx * 2 * sizeof(T2);
    N = half_n_cmplx*2;


    // alloc device memory
    allocDeviceBuffer(&work, used_bytes, ctx, queue[0]);

    copyToDevice(work, source, used_bytes, queue[0]);

    time0 = get_time();

    transform(work, n_ffts, kernelIndex == 0 ? fftKrnl : ifftKrnl, queue[0], distribution, 1);

    clFinish(queue[0]);
    time1 = get_time();

    T2 *workT = (T2*) malloc(used_bytes);
    if(workT == NULL)
    {
        printf("error alloc\n");
        exit(1);
    }

    copyFromDevice(workT, work, used_bytes, queue[0]);
    clFinish(queue[0]);

    freeDeviceBuffer(work, ctx, queue[0]);
    void *workCPU;
    T2* resultCPU;
    // allocate host memory
    allocHostBuffer((void**)&resultCPU, used_bytes, ctx, queue[1]);

    // alloc device memory
    allocDeviceBuffer(&workCPU, used_bytes, ctx, queue[1]);

    clFinish(queue[1]);

    copyToDevice(workCPU, workT, used_bytes, queue[1]);
    clFinish(queue[1]);

    transform(workCPU, n_ffts, kernelIndex == 0 ? cpuFftKrnl : cpuIfftKrnl, queue[1], distribution, 0);

    copyFromDevice(resultCPU, workCPU, used_bytes, queue[1]);
    clFinish(queue[1]);

    freeDeviceBuffer(workCPU, ctx, queue[1]);
    free(workT);

    cout << "\n\n\n";

    //imprime os primeiros 20 valores do vetor resultado e do gold, para comparação. PROBLEMA: mesmo com valores iguais, indica erro! WHYYYYYYYYY?
    /*	int c;
    	for(c = 0; c < 20; c++)
    	{
    		printf("[%d]:\nr.x - %le g.x - %le\nr.y - %le g.y - %le\n",c, resultCPU[c].x, gold[c].x, resultCPU[c].y, gold[c].y);
    	}
    */
    /// PROBABLY NOT NEEDED, OR CAN BE REDUCED
    double kernel_time = (double) (time1-time0) / 1000000;
    total_kernel_time += kernel_time;

    int num_errors = 0;
    int num_errors_i = 0; //complex

    #pragma omp parallel for reduction(+:num_errors)
    for (i = 0; i < N/2; i++) {

        if ((fabs(gold[i].x)>AVOIDZERO)&&
                ((fabs((resultCPU[i].x-gold[i].x)/resultCPU[i].x)>ACCEPTDIFF)||
                 (fabs((resultCPU[i].x-gold[i].x)/gold[i].x)>ACCEPTDIFF))) {
            if(num_errors < 20)
            printf("Error [%d]\ne (%f, %f)\nr (%f, %f)\n", i, gold[i].x, gold[i].y, resultCPU[i].x, resultCPU[i].y);
            num_errors++;
        }
        if ((fabs(gold[i].y)>AVOIDZERO)&&
                ((fabs((resultCPU[i].y-gold[i].y)/resultCPU[i].y)>ACCEPTDIFF)||
                 (fabs((resultCPU[i].y-gold[i].y)/gold[i].y)>ACCEPTDIFF))) {
            if(num_errors < 20)
            printf("Error [%d]\ne (%f, %f)\nr (%f, %f)\n", i, gold[i].x, gold[i].y, resultCPU[i].x, resultCPU[i].y);
            num_errors++;
        }

        if ((fabs(gold[i + (int) N/2].x)>AVOIDZERO)&&
                ((fabs((resultCPU[i + (int)N/2].x-gold[i +(int) N/2].x)/resultCPU[i+(int) N/2].x)>ACCEPTDIFF)||
                 (fabs((resultCPU[i +(int) N/2].x-gold[i +(int) N/2].x)/gold[i +(int) N/2].x)>ACCEPTDIFF))) {
            if(num_errors < 20)
            printf("Error [%d]\ne (%f, %f)\nr (%f, %f)\n", i, gold[i].x, gold[i].y, resultCPU[i].x, resultCPU[i].y);
            num_errors_i++;
        }
        if ((fabs(gold[i + (int) N/2].y)>AVOIDZERO)&&
                ((fabs((resultCPU[i + (int)N/2].y-gold[i + (int)N/2].y)/resultCPU[i +(int) N/2].y)>ACCEPTDIFF)||
                 (fabs((resultCPU[i +(int) N/2].y-gold[i +(int) N/2].y)/gold[i + (int) N/2].y)>ACCEPTDIFF))) {
            if(num_errors < 20)
            printf("Error [%d]\ne (%f, %f)\nr (%f, %f)\n", i, gold[i].x, gold[i].y, resultCPU[i].x, resultCPU[i].y);
            num_errors_i++;
        }

    }

    if(num_errors > 0 || num_errors_i > 0) {
        t_ea++;
    }


    //if(num_errors > 0 || (test_number % 10 == 0)) {
        printf("\ntest number: %d", test_number);
        printf("\nkernel time: %.12f", kernel_time);
        printf("\naccumulated kernel time: %f", total_kernel_time);
        printf("\namount of errors: %d", num_errors);
        printf("\ntotal runs with errors: %d\n", t_ea);

    //}
    //else {
    //    printf(".");
    //}

    freeHostBuffer(resultCPU, ctx, queue[1]);
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
}


int main(int argc, char** argv) {

    if(argc > 2){
        sizeIndex = atoi(argv[1]);
        distribution = atoi(argv[2]);
    } else {
        printf("ERROR! enter input size (0 to 5) and work distribution \
\ndistr = 0,   0%%  cpu | gpu 100%%\
\ndistr = 1,  33%%  cpu | gpu  66%%\
\ndistr = 2,  66%%  cpu | gpu  33%%\
\ndistr = 3, 100%%  cpu | gpu   0%%\n\n");
        exit(1);
    }



    FILE *fp, *fp_gold;
    // init host memory...
    if( (fp = fopen("/home/carol/daniel/opencl_fft/input_fft", "rb" )) == 0 ) {
        printf( "error file input_fft was not opened\n");
        return 0;
    }
    if( (fp_gold = fopen("/home/carol/daniel/opencl_fft/output_fft", "rb" )) == 0 ) {
        printf( "error file output_fft was not opened\n");
        return 0;
    }
    int return_value, return_value2, return_value3, return_value4;
    int i;

    unsigned long bytes = 0;
    int probSizes[6] = { 1, 8, 96, 256, 512, 1024};
    //int sizeIndex = 5;
    bytes = probSizes[sizeIndex];
    // Convert to MB
    bytes *= 1024 * 1024;
    int half_n_ffts = bytes / (512*sizeof(cplxdbl)*2);
    int half_n_cmplx = half_n_ffts * 512;
    double N = half_n_cmplx*2;

    source = (cplxdbl*)malloc(N*sizeof(cplxdbl));
    gold = (cplxdbl*)malloc(N*sizeof(cplxdbl));

    for (i = 0; i < N; i++) {
        return_value = fread(&(source[i].x), 1, sizeof(double), fp);
        return_value2 = fread(&(source[i].y), 1, sizeof(double), fp);
        return_value3 = fread(&(gold[i].x), 1, sizeof(double), fp_gold);
        return_value4 = fread(&(gold[i].y), 1, sizeof(double), fp_gold);
        if(return_value == 0 || return_value2 == 0 || return_value3 == 0 || return_value4 == 0) {
            printf("error reading input_fft or output_fft\n");
            return 0;
        }
    }
    fclose(fp);
    fclose(fp_gold);


    /*
    CL_DEVICE_TYPE_DEFAULT
    CL_DEVICE_TYPE_CPU
    CL_DEVICE_TYPE_GPU
    CL_DEVICE_TYPE_ACCELERATOR
    CL_DEVICE_TYPE_ALL
    */
    getDevices(CL_DEVICE_TYPE_ALL);

    init(true, device_id[0], context, command_queue[0], fftProg, fftKrnl,
         ifftKrnl, chkKrnl);

    init(true, device_id[1], context, command_queue[1], cpufftProg, cpuFftKrnl,
         cpuIfftKrnl, cpuChkKrnl);


    //LOOP START
    int loop;
    printf("%d ITERACTIONS\n", ITERACTIONS);
    for(loop=0; loop<ITERACTIONS; loop++) {
        dump<cplxdbl>(device_id[0], context, command_queue, loop);
    }
    free(source);
    free(gold);
}
