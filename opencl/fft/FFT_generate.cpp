//g++ simpleFFT.cpp fftlib.cpp -I/usr/local/cuda-5.5/include/CL/ -L/usr/lib/nvidia-current/ -lOpenCL -o fft

#include <cfloat>
#include <iostream>
#include <sstream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_ext.h>

#include "fftlib.h"

#define AVOIDZERO 1e-6
#define AVOIDZERON -1e-6
#define ACCEPTDIFF 1e-6

cl_platform_id          platform_id[100];
cl_device_id            device_id[100];
cl_context              context;
cl_command_queue        command_queue;
cl_program              program;

int sizeIndex;
char *kernel_file;

using namespace std;

template <class T2> inline bool dp(void);
template <> inline bool dp<cplxflt>(void) {
    return false;
}
template <> inline bool dp<cplxdbl>(void) {
    return true;
}

// Returns the current system time in microseconds
long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}


template <class T2> void dump(cl_device_id id,
                              cl_context ctx, cl_command_queue queue) {

    char input_fft[150], output_fft[150];
    snprintf(input_fft, 150, "input_fft_size_%d", sizeIndex);
    snprintf(output_fft, 150, "output_fft_size_%d", sizeIndex);

    int i;
    void* work;
    T2* source, * result;
    unsigned long bytes = 0;

    int probSizes[6] = { 1, 8, 96, 256, 512, 1024};

    bytes = probSizes[sizeIndex];

    // Convert to MB
    bytes *= 1024 * 1024;

    bool do_dp = dp<T2>();
    cl_program fftProg;
    cl_kernel fftKrnl, ifftKrnl, chkKrnl, goldChkKrnl;


    init(do_dp, kernel_file, id, ctx, queue, fftProg, fftKrnl,
         ifftKrnl, chkKrnl, goldChkKrnl);

    // now determine how much available memory will be used
    int half_n_ffts = bytes / (512*sizeof(T2)*2);
    int n_ffts = half_n_ffts * 2;
    int half_n_cmplx = half_n_ffts * 512;
    unsigned long used_bytes = half_n_cmplx * 2 * sizeof(T2);
    double N = half_n_cmplx*2;

    printf("used_bytes=%lu, N=%.12f\n", used_bytes, N);
    // allocate host and device memory
    allocHostBuffer((void**)&source, used_bytes, ctx, queue);
    allocHostBuffer((void**)&result, used_bytes, ctx, queue);

    // init host memory...
    for (i = 0; i < half_n_cmplx; i++) {
        source[i].x = (rand()/(float)RAND_MAX)*2-1;
        source[i].y = (rand()/(float)RAND_MAX)*2-1;
        source[i+half_n_cmplx].x = source[i].x;
        source[i+half_n_cmplx].y = source[i].y;
    }

    // alloc device memory
    allocDeviceBuffer(&work, used_bytes, ctx, queue);

    printf("Size of two-element vector: %f\n",N);

    FILE *fp;
    //change input unitl no zeros are found into output
    int sum_zeros = 0;
    int count_do_while = 0;
    do {
        sum_zeros = 0;
        copyToDevice(work, source, used_bytes, queue);

        printf("generating input and checkin output\n");

        if( (fp = fopen(input_fft, "wb" )) == 0 )
            printf( "The file %s was not opened\n",input_fft);

        //saving input
        for (i = 0; i < N; i++) {
            fwrite(&(source[i].x), 1, sizeof(double), fp);
            fwrite(&(source[i].y), 1, sizeof(double), fp);
        }
        fclose(fp);

        long long time0, time1;
        time0 = get_time();
        transform(work, n_ffts, fftKrnl, queue, 0, 1, 64);
        clFinish(queue);
        time1 = get_time();
        double kernel_time = (double) (time1-time0) / 1000000;
        double fftsz = 512;
        double Gflops = n_ffts*(5*fftsz*log2(fftsz))/kernel_time;
        printf("NFFT:%d GFLOPS:%f\n",n_ffts,Gflops);
        printf("\nkernel time: %.12f\n", kernel_time);
        copyFromDevice(result, work, used_bytes, queue);

        if( (fp = fopen(output_fft, "wb" )) == 0 )
            printf( "The file %s was not opened\n", output_fft);
        //saving output

        for (i = 0; i < N; i++) {

            if(result[i].x > AVOIDZERON && result[i].x < AVOIDZERO && i < half_n_cmplx) {
                printf("ZERO at postion %d, (%.12f, %.12f)\n",i, result[i].x, result[i].y);
                source[i].x++;
                source[i+half_n_cmplx].x++;
                source[i].y++;
                source[i+half_n_cmplx].y++;
                sum_zeros++;
            }
            if(result[i].y > AVOIDZERON && result[i].y < AVOIDZERO && i < half_n_cmplx) {
                printf("ZERO at postion %d, (%.12f, %.12f)\n",i, result[i].x, result[i].y);
                source[i].x++;
                source[i+half_n_cmplx].x++;
                source[i].y++;
                source[i+half_n_cmplx].y++;
                sum_zeros++;
            }

            fwrite(&(result[i].x), 1, sizeof(double), fp);
            fwrite(&(result[i].y), 1, sizeof(double), fp);
            source[i].x = result[i].x;
            source[i].y = result[i].y;
        }
        fclose(fp);
        printf("Number of zeros found at output: %d\n",sum_zeros);
        count_do_while++;
    } while (sum_zeros > 0 && count_do_while < 10);


    transform(work, n_ffts, ifftKrnl, queue, 0, 1, 64);
    copyFromDevice(result, work, used_bytes, queue);

    //checking inverse
    if( (fp = fopen(input_fft, "rb" )) == 0 ) {
        printf( "The file %s was not opened\n",input_fft);
    }
    double x, y;

    int num_errors = 0;
    for (i = 0; i < N; i++) {
        fread(&x, 1, sizeof(double), fp);
        fread(&y, 1, sizeof(double), fp);

        if ((fabs(x)>AVOIDZERO)&&
                ((fabs((result[i].x-x)/result[i].x)>ACCEPTDIFF)||
                 (fabs((result[i].x-x)/x)>ACCEPTDIFF))) {
            if(num_errors < 20)
            printf("Error\ne (%f, %f)\nr (%f, %f)\n", x, y, result[i].x, result[i].y);
            num_errors++;
        }
        if ((fabs(y)>AVOIDZERO)&&
                ((fabs((result[i].y-y)/result[i].y)>ACCEPTDIFF)||
                 (fabs((result[i].y-y)/y)>ACCEPTDIFF))) {
            if(num_errors < 20)
            printf("Error\ne (%f, %f)\nr (%f, %f)\n", x, y, result[i].x, result[i].y);
            num_errors++;
        }

    }
    fclose(fp);

    printf("Number of errors at inverse fft: %d\n",num_errors);

    freeDeviceBuffer(work, ctx, queue);
    freeHostBuffer(source, ctx, queue);
    freeHostBuffer(result, ctx, queue);
    deinit(queue, fftProg, fftKrnl, ifftKrnl, chkKrnl);
}


void getDevices(cl_device_type deviceType) {
    cl_uint         platforms_n = 0;
    cl_uint         devices_n   = 0;
    cl_int                  ret;

    /* The following code queries the number of platforms and devices, and
     * lists the information about both.
     */
    clGetPlatformIDs(100, platform_id, &platforms_n);
    //	if (VERBOSE)
    {
        printf("\n=== %d OpenCL platform(s) found: ===\n", platforms_n);
        for (int i = 0; i < platforms_n; i++) {
            char buffer[10240];
            printf("  -- %d --\n", i);
            clGetPlatformInfo(platform_id[i], CL_PLATFORM_PROFILE, 10240, buffer,
                              NULL);
            printf("  PROFILE = %s\n", buffer);
            clGetPlatformInfo(platform_id[i], CL_PLATFORM_VERSION, 10240, buffer,
                              NULL);
            printf("  VERSION = %s\n", buffer);
            clGetPlatformInfo(platform_id[i], CL_PLATFORM_NAME, 10240, buffer, NULL);
            printf("  NAME = %s\n", buffer);
            clGetPlatformInfo(platform_id[i], CL_PLATFORM_VENDOR, 10240, buffer, NULL);
            printf("  VENDOR = %s\n", buffer);
            clGetPlatformInfo(platform_id[i], CL_PLATFORM_EXTENSIONS, 10240, buffer,
                              NULL);
            printf("  EXTENSIONS = %s\n", buffer);
        }
    }

    clGetDeviceIDs(platform_id[0], deviceType, 100, device_id, &devices_n);
    //	if (VERBOSE)
    {
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
            clGetDeviceInfo(device_id[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                            sizeof(buf_ulong), &buf_ulong, NULL);
            printf("  CL_DEVICE_MAX_MEM_ALLOC_SIZE = %llu\n",
                   (unsigned long long) buf_ulong);
        }
        //printf("\n");
    }

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

void usage(){
	printf("Usage: fft <input_size> <cl_device_tipe> <ocl_kernel_file> \n");
	printf("  input size range from 0 to 5\n");
	printf("  cl_device_types\n");
	printf("    Default: %d\n",CL_DEVICE_TYPE_DEFAULT);
	printf("    CPU: %d\n",CL_DEVICE_TYPE_CPU);
	printf("    GPU: %d\n",CL_DEVICE_TYPE_GPU);
	printf("    ACCELERATOR: %d\n",CL_DEVICE_TYPE_ACCELERATOR);
	printf("    ALL: %d\n",CL_DEVICE_TYPE_ALL);
}

int main(int argc, char** argv) {

    int devType;
    if(argc == 4) {
        sizeIndex = atoi(argv[1]);
        devType = atoi(argv[2]);
        kernel_file = argv[3];
    } else {
        usage();
        exit(1);
    }


    getDevices(devType);

    dump<cplxdbl>(device_id[0], context, command_queue);
}
