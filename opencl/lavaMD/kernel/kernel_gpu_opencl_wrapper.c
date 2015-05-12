#ifdef __cplusplus
extern "C" {
#endif

#include <string.h>

#include <CL/cl.h>					// (in library path provided to compiler)	needed by OpenCL types and functions

#include "./../main.h"								// (in the main program folder)	needed to recognized input parameters

#include "./../util/opencl/opencl.h"				// (in library path specified to compiler)	needed by for device functions
#include "./../util/timer/timer.h"					// (in library path specified to compiler)	needed by timer


#include "./kernel_gpu_opencl_wrapper.h"				// (in the current directory)

void
kernel_gpu_opencl_wrapper(	par_str par_cpu,
                            dim_str dim_cpu,
                            box_str* box_cpu,
                            FOUR_VECTOR* rv_cpu,
                            fp* qv_cpu,
                            FOUR_VECTOR* fv_cpu)
{

    // timer
    long long time0;
    long long time1;
    long long time2;
    long long time3;
    long long time4;
    long long time5;
    long long time6;

    time0 = get_time();


    // common variables
    cl_int error;


    // Get the number of available platforms
    cl_uint num_platforms;
    error = clGetPlatformIDs(	0,
                                NULL,
                                &num_platforms);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);

    // Get the list of available platforms
    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
    error = clGetPlatformIDs(	num_platforms,
                                platforms,
                                NULL);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);

    // Select the 1st platform
    cl_platform_id platform = platforms[0];

    // Get the name of the selected platform and print it (if there are multiple platforms, choose the first one)
    char pbuf[100];
    error = clGetPlatformInfo(	platform,
                                CL_PLATFORM_VENDOR,
                                sizeof(pbuf),
                                pbuf,
                                NULL);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);
    printf("Platform: %s\n", pbuf);


    // Create context properties for selected platform
    cl_context_properties context_properties[3] = {	CL_CONTEXT_PLATFORM,
                                                    (cl_context_properties) platform,
                                                    0
                                                  };

    // Create context for selected platform being GPU
    cl_context context;
    context = clCreateContextFromType(	context_properties,
                                        CL_DEVICE_TYPE_GPU,
                                        NULL,
                                        NULL,
                                        &error);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);


    // Get the number of devices (previousely selected for the context)
    size_t devices_size;
    error = clGetContextInfo(	context,
                                CL_CONTEXT_DEVICES,
                                0,
                                NULL,
                                &devices_size);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);

    // Get the list of devices (previousely selected for the context)
    cl_device_id *devices = (cl_device_id *) malloc(devices_size);
    error = clGetContextInfo(	context,
                                CL_CONTEXT_DEVICES,
                                devices_size,
                                devices,
                                NULL);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);

    // Select the first device (previousely selected for the context) (if there are multiple devices, choose the first one)
    cl_device_id device;
    device = devices[0];

    // Get the name of the selected device (previousely selected for the context) and print it
    error = clGetDeviceInfo(device,
                            CL_DEVICE_NAME,
                            sizeof(pbuf),
                            pbuf,
                            NULL);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);
    printf("Device: %s\n", pbuf);


    // Create a command queue
    cl_command_queue command_queue;
    command_queue = clCreateCommandQueue(	context,
                                            device,
                                            0,
                                            &error);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);


    // Load kernel source code from file
    const char *source = load_kernel_source("./kernel/kernel_gpu_opencl.cl");
    size_t sourceSize = strlen(source);

    // Create the program
    cl_program program = clCreateProgramWithSource(	context,
                         1,
                         &source,
                         &sourceSize,
                         &error);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);

    // parameterized kernel dimension
    char clOptions[110];
    //  sprintf(clOptions,"-I../../src");
    sprintf(clOptions,"-I.");
#ifdef RD_WG_SIZE
    sprintf(clOptions + strlen(clOptions), " -DRD_WG_SIZE=%d", RD_WG_SIZE);
#endif
#ifdef RD_WG_SIZE_0
    sprintf(clOptions + strlen(clOptions), " -DRD_WG_SIZE_0=%d", RD_WG_SIZE_0);
#endif
#ifdef RD_WG_SIZE_0_0
    sprintf(clOptions + strlen(clOptions), " -DRD_WG_SIZE_0_0=%d", RD_WG_SIZE_0_0);
#endif


    // Compile the program
    error = clBuildProgram(	program,
                            1,
                            &device,
                            clOptions,
                            NULL,
                            NULL);
    // Print warnings and errors from compilation
    static char log[65536];
    memset(log, 0, sizeof(log));
    clGetProgramBuildInfo(	program,
                            device,
                            CL_PROGRAM_BUILD_LOG,
                            sizeof(log)-1,
                            log,
                            NULL);
    if (strstr(log,"warning:") || strstr(log, "error:"))
        printf("<<<<\n%s\n>>>>\n", log);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);

    // Create kernel
    cl_kernel kernel;
    kernel = clCreateKernel(program,
                            "kernel_gpu_opencl",
                            &error);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);


    size_t local_work_size[1];
    local_work_size[0] = NUMBER_THREADS;
    size_t global_work_size[1];
    global_work_size[0] = dim_cpu.number_boxes * local_work_size[0];

    printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", global_work_size[0]/local_work_size[0], local_work_size[0]);

    time1 = get_time();


    cl_mem d_box_gpu;
    d_box_gpu = clCreateBuffer(	context,
                                CL_MEM_READ_WRITE,
                                dim_cpu.box_mem,
                                NULL,
                                &error );
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);


    cl_mem d_rv_gpu;
    d_rv_gpu = clCreateBuffer(	context,
                                CL_MEM_READ_WRITE,
                                dim_cpu.space_mem,
                                NULL,
                                &error );
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);


    cl_mem d_qv_gpu;
    d_qv_gpu = clCreateBuffer(	context,
                                CL_MEM_READ_WRITE,
                                dim_cpu.space_mem2,
                                NULL,
                                &error );
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);



    cl_mem d_fv_gpu;
    d_fv_gpu = clCreateBuffer(	context,
                                CL_MEM_READ_WRITE,
                                dim_cpu.space_mem,
                                NULL,
                                &error );
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);

    time2 = get_time();


    error = clEnqueueWriteBuffer(	command_queue, d_box_gpu, 1,0,dim_cpu.box_mem,box_cpu,0,NULL,NULL);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);


    error = clEnqueueWriteBuffer(	command_queue,
                                    d_rv_gpu,
                                    1,
                                    0,
                                    dim_cpu.space_mem,
                                    rv_cpu,
                                    0,
                                    0,
                                    0);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);


    error = clEnqueueWriteBuffer(	command_queue,
                                    d_qv_gpu,
                                    1,
                                    0,
                                    dim_cpu.space_mem2,
                                    qv_cpu,
                                    0,
                                    0,
                                    0);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);


    error = clEnqueueWriteBuffer(	command_queue,
                                    d_fv_gpu,
                                    1,
                                    0,
                                    dim_cpu.space_mem,
                                    fv_cpu,
                                    0,
                                    0,
                                    0);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);

    time3 = get_time();

    clSetKernelArg(	kernel,
                    0,
                    sizeof(par_str),
                    (void *) &par_cpu);
    clSetKernelArg(	kernel,
                    1,
                    sizeof(dim_str),
                    (void *) &dim_cpu);
    clSetKernelArg(	kernel,
                    2,
                    sizeof(cl_mem),
                    (void *) &d_box_gpu);
    clSetKernelArg(	kernel,
                    3,
                    sizeof(cl_mem),
                    (void *) &d_rv_gpu);
    clSetKernelArg(	kernel,
                    4,
                    sizeof(cl_mem),
                    (void *) &d_qv_gpu);
    clSetKernelArg(	kernel,
                    5,
                    sizeof(cl_mem),
                    (void *) &d_fv_gpu);

    // launch kernel - all boxes
    error = clEnqueueNDRangeKernel(	command_queue,
                                    kernel,
                                    1,
                                    NULL,
                                    global_work_size,
                                    local_work_size,
                                    0,
                                    NULL,
                                    NULL);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);

    // Wait for all operations to finish NOT SURE WHERE THIS SHOULD GO
    error = clFinish(command_queue);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);

    time4 = get_time();


    error = clEnqueueReadBuffer(command_queue,d_fv_gpu,CL_TRUE,0,dim_cpu.space_mem,fv_cpu,0,NULL,NULL);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);


    time5 = get_time();


    // Release kernels...
    clReleaseKernel(kernel);

    // Now the program...
    clReleaseProgram(program);

    // Clean up the device memory...
    clReleaseMemObject(d_rv_gpu);
    clReleaseMemObject(d_qv_gpu);
    clReleaseMemObject(d_fv_gpu);
    clReleaseMemObject(d_box_gpu);

    // Flush the queue
    error = clFlush(command_queue);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);

    // ...and finally, the queue and context.
    clReleaseCommandQueue(command_queue);

    // ???
    clReleaseContext(context);

    time6 = get_time();


    printf("Time spent in different stages of GPU_CUDA KERNEL:\n");

    printf("%15.12f s, %15.12f % : GPU: SET DEVICE / DRIVER INIT\n",	(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time6-time0) * 100);
    printf("%15.12f s, %15.12f % : GPU MEM: ALO\n", 					(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time6-time0) * 100);
    printf("%15.12f s, %15.12f % : GPU MEM: COPY IN\n",					(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time6-time0) * 100);

    printf("%15.12f s, %15.12f % : GPU: KERNEL\n",						(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time6-time0) * 100);

    printf("%15.12f s, %15.12f % : GPU MEM: COPY OUT\n",				(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time6-time0) * 100);
    printf("%15.12f s, %15.12f % : GPU MEM: FRE\n", 					(float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time6-time0) * 100);

    printf("Total time:\n");
    printf("%.12f s\n", 												(float) (time6-time0) / 1000000);

}


#ifdef __cplusplus
}
#endif
