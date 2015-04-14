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

#ifdef LOGS
#include "/home/carol/log_helper/log_helper.h"
#endif /* LOGS */

//////////
// LOOPS
//////////
#define ITERACTIONS 10000000

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

int ocl_exec_gchk(cplxdbl *gold, int n, int mem_size, size_t thread_per_block, double avoidzero, double acceptdiff);

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
cl_kernel fftKrnl, ifftKrnl, chkKrnl, goldChkKrnl;
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

#ifdef LOGS
    start_iteration();
#endif /* LOGS */
    transform(work, n_ffts, kernelIndex == 0 ? fftKrnl : ifftKrnl, queue[0], distribution, 1);
#ifdef LOGS
    end_iteration();
#endif /* LOGS */

    clFinish(queue[0]);
    time1 = get_time();
    double kernel_time = (double) (time1-time0) / 1000000;
    double fftsz = 512;
    double Gflops = n_ffts*(5*fftsz*log2(fftsz))/kernel_time;

    if(distribution == 3)
        kernel_time = 0;

    T2 *workT = (T2*) malloc(used_bytes);
    if(workT == NULL)
    {
        printf("error alloc\n");
        exit(1);
    }

    copyFromDevice(workT, work, used_bytes, queue[0]);
    clFinish(queue[0]);

    T2* resultCPU = workT;

    int kerrors=0;
    //gold[323].x=2;
    kerrors = ocl_exec_gchk(gold, queue[0], ctx, work, goldChkKrnl, N, sizeof(cplxdbl)*N, 64, AVOIDZERO, ACCEPTDIFF);
    int num_errors = 0;
    int num_errors_i = 0; //complex
    log_error_count(kerrors);
    if (kerrors!=0)
    {

        #pragma omp parallel for reduction(+:num_errors)
        for (i = 0; i < N/2; i++) {

            char error_detail[150];

            if ((fabs(gold[i].x)>AVOIDZERO)&&
                    ((fabs((resultCPU[i].x-gold[i].x)/resultCPU[i].x)>ACCEPTDIFF)||
                     (fabs((resultCPU[i].x-gold[i].x)/gold[i].x)>ACCEPTDIFF))) {
                num_errors++;
#ifdef LOGS
                snprintf(error_detail, 150, "pos:%d real r:%1.16e e:%1.16e",i, resultCPU[i].x, gold[i].x);
                log_error_detail(error_detail);
#endif /* LOGS */
                //log_error_detail("pos:%d real r:%1.16e e:%1.16e",i, resultCPU[i].x, gold[i].x);
            }
            if ((fabs(gold[i].y)>AVOIDZERO)&&
                    ((fabs((resultCPU[i].y-gold[i].y)/resultCPU[i].y)>ACCEPTDIFF)||
                     (fabs((resultCPU[i].y-gold[i].y)/gold[i].y)>ACCEPTDIFF))) {
                num_errors++;
#ifdef LOGS
                snprintf(error_detail, 150, "pos:%d real r:%1.16e e:%1.16e",i, resultCPU[i].y, gold[i].y);
                log_error_detail(error_detail);
#endif /* LOGS */
                //log_error_detail("pos:%d real r:%1.16e e:%1.16e",i, resultCPU[i].y, gold[i].y);
            }

            if ((fabs(gold[i + (int) N/2].x)>AVOIDZERO)&&
                    ((fabs((resultCPU[i + (int)N/2].x-gold[i +(int) N/2].x)/resultCPU[i+(int) N/2].x)>ACCEPTDIFF)||
                     (fabs((resultCPU[i +(int) N/2].x-gold[i +(int) N/2].x)/gold[i +(int) N/2].x)>ACCEPTDIFF))) {
                num_errors_i++;
#ifdef LOGS
                snprintf(error_detail, 150, "pos:%d imag r:%1.16e e:%1.16e",i, resultCPU[i+ (int)N/2].x, gold[i + (int)N/2].x);
                log_error_detail(error_detail);
#endif /* LOGS */
                //log_error_detail("pos:%d imag r:%1.16e e:%1.16e",i, resultCPU[i+ (int)N/2].x, gold[i + (int)N/2].x);
            }
            if ((fabs(gold[i + (int) N/2].y)>AVOIDZERO)&&
                    ((fabs((resultCPU[i + (int)N/2].y-gold[i + (int)N/2].y)/resultCPU[i +(int) N/2].y)>ACCEPTDIFF)||
                     (fabs((resultCPU[i +(int) N/2].y-gold[i +(int) N/2].y)/gold[i + (int) N/2].y)>ACCEPTDIFF))) {
                num_errors_i++;
#ifdef LOGS
                snprintf(error_detail, 150, "pos:%d imag r:%1.16e e:%1.16e",i, resultCPU[i+ (int)N/2].y, gold[i + (int)N/2].y);
                log_error_detail(error_detail);
#endif /* LOGS */
                //log_error_detail("pos:%d imag r:%1.16e e:%1.16e",i, resultCPU[i+ (int)N/2].y, gold[i + (int)N/2].y);
            }


        }


    }

    if (test_number % 15 == 0)
        printf ("it:%d. cpu errors check: r=%d i=%d\n", test_number, num_errors, num_errors_i);

    freeDeviceBuffer(work, ctx, queue[0]);


    free(resultCPU);

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
    command_queue[0] = clCreateCommandQueue(context, device_id[0], 0, &ret);
    if (ret != CL_SUCCESS) {
        printf("\nError at clCreateCommandQueue! Error code %i\n\n", ret);
        exit(1);
    }

}


int main(int argc, char** argv) {

    if(argc > 1) {
        sizeIndex = atoi(argv[1]);
        distribution = 0;//atoi(argv[2]);
    } else {
        printf("ERROR! enter input size (0 to 3)\n");
        exit(1);
    }



    FILE *fp, *fp_gold;
    // init host memory...
    if( (fp = fopen("/home/carol/DSN15_codes/openclfft/input_fft", "rb" )) == 0 ) {
        printf( "error file input_fft was not opened\n");
        return 0;
    }
    if( (fp_gold = fopen("/home/carol/DSN15_codes/openclfft/output_fft", "rb" )) == 0 ) {
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

#ifdef LOGS
    char test_info[100];
    snprintf(test_info, 100, "size:%d",(int)N);
    start_log_file("openclfft", test_info);
    set_max_errors_iter(100);
    set_iter_interval_print(15);
#endif /* LOGS */
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
    getDevices(CL_DEVICE_TYPE_GPU);

    init(true, device_id[0], context, command_queue[0], fftProg, fftKrnl,
         ifftKrnl, chkKrnl, goldChkKrnl);

    //init(true, device_id[1], context, command_queue[1], cpufftProg, cpuFftKrnl,
    //     cpuIfftKrnl, cpuChkKrnl);


    //LOOP START
    int loop;
    printf("%d ITERACTIONS\n", ITERACTIONS);
    for(loop=0; loop<ITERACTIONS; loop++) {
        dump<cplxdbl>(device_id[0], context, command_queue, loop);
    }
    free(source);
    free(gold);
#ifdef LOGS
    end_log_file();
#endif /* LOGS */

}

