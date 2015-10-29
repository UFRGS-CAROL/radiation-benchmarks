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
#include <omp.h>

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_ext.h>

#include "fftlib.h"
#include "kernel_fft.h"

#ifdef LOGS
#include "../../include/log_helper.h"
#endif /* LOGS */

#define MAX_ERR_ITER_LOG 500

#define NPROCESSORS 8 // OMP num threads

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

int sizeIndex;

#ifdef TIMING
inline long long timing_get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

long long setup_start, setup_end;
long long loop_start, loop_end;
long long kernel_start, kernel_end;
long long check_start, check_end;
int NFFT;
#endif

int loops_fft_iter;


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
#ifdef TIMING
    NFFT=n_ffts;
#endif
    int half_n_cmplx = half_n_ffts * 512;
    unsigned long used_bytes = half_n_cmplx * 2 * sizeof(T2);
    N = half_n_cmplx*2;

#ifdef ERR_INJ
	if(test_number == 2){
		printf("injecting error, changing input!\n");
		source[0].x = source[0].x*2;
		source[0].y = source[0].y*-3;
	}else if(test_number == 3){
		printf("injecting error, restoring input!\n");
		source[0].x = source[0].x/2;
		source[0].y = source[0].y/-3;
	} else if (test_number == 4){
		printf("get ready, infinite loop...");
		fflush(stdout);
		while(1){sleep(1000);}
	}
#endif

    // alloc device memory
    allocDeviceBuffer(&work, used_bytes, ctx, queue[0]);

    copyToDevice(work, source, used_bytes, queue[0]);


#ifdef TIMING
	kernel_start = timing_get_time();
#endif

#ifdef LOGS
    start_iteration();
#endif /* LOGS */
    int loop_ffts;
    for(loop_ffts=0;loop_ffts<loops_fft_iter;loop_ffts++){
        transform(work, n_ffts, kernelIndex == 0 ? fftKrnl : ifftKrnl, queue[0], distribution, 1, 64);
        transform(work, n_ffts, kernelIndex == 1 ? fftKrnl : ifftKrnl, queue[0], distribution, 1, 64);
        transform(work, n_ffts, kernelIndex == 0 ? fftKrnl : ifftKrnl, queue[0], distribution, 1, 64);
    }
    clFinish(queue[0]);

#ifdef LOGS
    end_iteration();
#endif /* LOGS */

#ifdef TIMING
	kernel_end = timing_get_time();
#endif



    T2 *workT = (T2*) malloc(used_bytes);
    if(workT == NULL)
    {
        printf("error alloc\n");
        exit(1);
    }

    copyFromDevice(workT, work, used_bytes, queue[0]);
    clFinish(queue[0]);

    T2* resultCPU = workT;

    
    
#ifdef TIMING
	check_start = timing_get_time();
#endif
  
	int num_errors = 0;
	int num_errors_i = 0; //complex
	omp_set_num_threads(NPROCESSORS);
        #pragma omp parallel for reduction(+:num_errors,num_errors_i)
        for (i = 0; i < (int)N/2; i++) {
            if ((fabs(gold[i].x)>AVOIDZERO)&&
                    ((fabs((resultCPU[i].x-gold[i].x)/resultCPU[i].x)>ACCEPTDIFF)||
                     (fabs((resultCPU[i].x-gold[i].x)/gold[i].x)>ACCEPTDIFF))) {
                num_errors++;
            }
            if ((fabs(gold[i].y)>AVOIDZERO)&&
                    ((fabs((resultCPU[i].y-gold[i].y)/resultCPU[i].y)>ACCEPTDIFF)||
                     (fabs((resultCPU[i].y-gold[i].y)/gold[i].y)>ACCEPTDIFF))) {
                num_errors++;
            }
 if ((fabs(gold[i + (int) N/2].x)>AVOIDZERO)&&
                    ((fabs((resultCPU[i + (int)N/2].x-gold[i +(int) N/2].x)/resultCPU[i+(int) N/2].x)>ACCEPTDIFF)||
                     (fabs((resultCPU[i +(int) N/2].x-gold[i +(int) N/2].x)/gold[i +(int) N/2].x)>ACCEPTDIFF))) {
                num_errors_i++;
            }
            if ((fabs(gold[i + (int) N/2].y)>AVOIDZERO)&&
                    ((fabs((resultCPU[i + (int)N/2].y-gold[i + (int)N/2].y)/resultCPU[i +(int) N/2].y)>ACCEPTDIFF)||
                     (fabs((resultCPU[i +(int) N/2].y-gold[i +(int) N/2].y)/gold[i + (int) N/2].y)>ACCEPTDIFF))) {
                num_errors_i++;
            }
        }

#ifdef LOGS
	int num_errors_total = num_errors + num_errors_i;
	if(num_errors_total>0){
		int err_loged=0;
		for (i = 0; i < N/2 && err_loged < MAX_ERR_ITER_LOG && err_loged < num_errors_total; i++) {

			char error_detail[150];
	
			if ((fabs(gold[i].x)>AVOIDZERO)&&
				((fabs((resultCPU[i].x-gold[i].x)/resultCPU[i].x)>ACCEPTDIFF)||
				(fabs((resultCPU[i].x-gold[i].x)/gold[i].x)>ACCEPTDIFF))) {
					err_loged++;
					snprintf(error_detail, 150, "pos:%d real r:%1.16e e:%1.16e",i, resultCPU[i].x, gold[i].x);
					log_error_detail(error_detail);
			}
			if(err_loged >= MAX_ERR_ITER_LOG) break;
			if ((fabs(gold[i].y)>AVOIDZERO)&&
				((fabs((resultCPU[i].y-gold[i].y)/resultCPU[i].y)>ACCEPTDIFF)||
				(fabs((resultCPU[i].y-gold[i].y)/gold[i].y)>ACCEPTDIFF))) {
					err_loged++;
					snprintf(error_detail, 150, "pos:%d real r:%1.16e e:%1.16e",i, resultCPU[i].y, gold[i].y);
					log_error_detail(error_detail);
			}
			if(err_loged >= MAX_ERR_ITER_LOG) break;
			if ((fabs(gold[i + (int) N/2].x)>AVOIDZERO)&&
				((fabs((resultCPU[i + (int)N/2].x-gold[i +(int) N/2].x)/resultCPU[i+(int) N/2].x)>ACCEPTDIFF)||
				(fabs((resultCPU[i +(int) N/2].x-gold[i +(int) N/2].x)/gold[i +(int) N/2].x)>ACCEPTDIFF))) {
					err_loged++;
					snprintf(error_detail, 150, "pos:%d imag r:%1.16e e:%1.16e",i, resultCPU[i+ (int)N/2].x, gold[i + (int)N/2].x);
					log_error_detail(error_detail);
			}
			if(err_loged >= MAX_ERR_ITER_LOG) break;
			if ((fabs(gold[i + (int) N/2].y)>AVOIDZERO)&&
				((fabs((resultCPU[i + (int)N/2].y-gold[i + (int)N/2].y)/resultCPU[i +(int) N/2].y)>ACCEPTDIFF)||
				(fabs((resultCPU[i +(int) N/2].y-gold[i +(int) N/2].y)/gold[i + (int) N/2].y)>ACCEPTDIFF))) {
					err_loged++;
					snprintf(error_detail, 150, "pos:%d imag r:%1.16e e:%1.16e",i, resultCPU[i+ (int)N/2].y, gold[i + (int)N/2].y);
					log_error_detail(error_detail);
			}
			if(err_loged >= MAX_ERR_ITER_LOG) break;
        	}
	}
	log_error_count(num_errors_total);
#endif /* LOGS */

#ifdef TIMING
	check_end = timing_get_time();
#endif
	
    if (test_number % 15 == 0 || num_errors>0 || num_errors_i>0)
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

void usage(){
        printf("Usage: fft <input_size> <cl_device_tipe> <FFT_kernel_repetitions> <input_file> <output_gold_file> <#iterations>\n");
        printf("  input size range from 0 to 5\n");
        printf("  cl_device_types\n");
        printf("    Default: %d\n",CL_DEVICE_TYPE_DEFAULT);
        printf("    CPU: %d\n",CL_DEVICE_TYPE_CPU);
        printf("    GPU: %d\n",CL_DEVICE_TYPE_GPU);
        printf("    ACCELERATOR: %d\n",CL_DEVICE_TYPE_ACCELERATOR);
        printf("    ALL: %d\n",CL_DEVICE_TYPE_ALL);
}

int main(int argc, char** argv) {
#ifdef TIMING
	setup_start = timing_get_time();
#endif

    int devType, iterations=1;
    //char *kernel_file;
    char *input, *output;
    if(argc == 7) {
        sizeIndex = atoi(argv[1]);
        devType = atoi(argv[2]);
        loops_fft_iter = atoi(argv[3]);
        input = argv[4];
        output = argv[5];
        iterations = atoi(argv[6]);
        distribution = 0;//atoi(argv[2]);
    } else {
        usage();
        exit(1);
    }
    if(loops_fft_iter<1){
        printf("<FFT_kernel_repetitions> should be greater than 1\n");
        usage();
        exit(1);
    }

    
    FILE *fp, *fp_gold;
    // init host memory...
    if( (fp = fopen(input, "rb" )) == 0 ) {
        printf( "error file input_fft was not opened\n");
        return 0;
    }
    if( (fp_gold = fopen(output, "rb" )) == 0 ) {
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
    start_log_file((char *)"openclfft", test_info);
    set_max_errors_iter(MAX_ERR_ITER_LOG);
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


    getDevices(devType);

    char * kernel_source = (char *)malloc(sizeof(char)*strlen(kernel_fft_ocl)+2);
    strcpy(kernel_source,kernel_fft_ocl);
    init(true, kernel_source, device_id[0], context, command_queue[0], fftProg, fftKrnl,
         ifftKrnl);



#ifdef TIMING
	setup_end = timing_get_time();
#endif
    //LOOP START
    int loop;
    for(loop=0; loop<iterations; loop++) {
#ifdef TIMING
	loop_start = timing_get_time();
#endif
        dump<cplxdbl>(device_id[0], context, command_queue, loop);
#ifdef TIMING
	loop_end = timing_get_time();
	double setup_timing = (double) (setup_end - setup_start) / 1000000;
	double loop_timing = (double) (loop_end - loop_start) / 1000000;
	double kernel_timing = (double) (kernel_end - kernel_start) / 1000000;
	double check_timing = (double) (check_end - check_start) / 1000000;
	printf("\n\tTIMING:\n");
	printf("setup: %f\n",setup_timing);
	printf("loop: %f\n",loop_timing);
	printf("kernel: %f\n",kernel_timing);
	printf("check: %f\n",check_timing);
	double fftsz = 512;
	double Gflops = NFFT*(5*fftsz*log2(fftsz))/kernel_timing;
	printf("nfft:%d\ngflops:%f\n",NFFT,Gflops);
#endif
    }
    free(source);
    free(gold);
#ifdef LOGS
    end_log_file();
#endif /* LOGS */

}

