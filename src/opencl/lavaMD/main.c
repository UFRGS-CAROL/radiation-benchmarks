#include <stdio.h> // (in path known to compiler)			needed by printf
#include <stdlib.h> // (in path known to compiler)			needed by malloc
#include <stdbool.h> // (in path known to compiler)			needed by true/false

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>

#include "./main.h" // (in the current directory)
#include "kernel_lavamd.h"

char input_distance[150];
char input_charges[150];
char output_gold[150];
//char *kernel_file;
int block_size;
int devType;
void kernel_gpu_opencl_wrapper(	par_str par_cpu,
                            dim_str dim_cpu,
                            box_str* box_cpu,
                            FOUR_VECTOR* rv_cpu,
                            fp* qv_cpu,
                            FOUR_VECTOR* fv_cpu);

long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

void usage(){
        printf("Usage: lavamd_gen <input_size> <cl_device_tipe> <workgroup_block_size>\n");
        printf("  input size is the number of boxes, 15 is a reasonable number\n");
        printf("  cl_device_types\n");
        printf("    Default: %d\n",CL_DEVICE_TYPE_DEFAULT);
        printf("    CPU: %d\n",CL_DEVICE_TYPE_CPU);
        printf("    GPU: %d\n",CL_DEVICE_TYPE_GPU);
        printf("    ACCELERATOR: %d\n",CL_DEVICE_TYPE_ACCELERATOR);
        printf("    ALL: %d\n",CL_DEVICE_TYPE_ALL);
}

int number_nn=0; // total number of neighbors
int boxes = 15;

int main(int argc, char *argv []){

    if(argc == 4) {
        boxes = atoi(argv[1]);
        devType = atoi(argv[2]);
        //kernel_file = argv[3];
        block_size = atoi(argv[3]);
    } else {
        usage();
        exit(1);
    }

    snprintf(input_distance, 150, "input_distance_%d_%d",boxes, block_size);
    snprintf(input_charges, 150, "input_charges_%d_%d",boxes, block_size);
    snprintf(output_gold, 150, "output_gold_%d_%d",boxes, block_size);
    //====================================================================
    //	CPU/MCPU VARIABLES
    //=====================================================================

    // counters
    int i, j, k, l, m, n;

    // system memory
    par_str par_cpu;
    dim_str dim_cpu;
    box_str* box_cpu;
    FOUR_VECTOR* rv_cpu;
    fp* qv_cpu;
    FOUR_VECTOR* fv_cpu;
    int nh;


    printf("WG size of kernel = %d \n", block_size);


    //=====================================================================
    //	CHECK INPUT ARGUMENTS
    //=====================================================================

    // assing default values
    dim_cpu.arch_arg = 0;
    dim_cpu.cores_arg = 1;
    dim_cpu.boxes1d_arg = boxes;


    //=====================================================================
    //	INPUTS
    //=====================================================================

    par_cpu.alpha = 0.5;


    //=====================================================================
    //	DIMENSIONS
    //=====================================================================

    // total number of boxes
    dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg; // 8*8*8=512

    // how many particles space has in each direction
    dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;							//512*100=51,200
    dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
    dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);

    // box array
    dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);


    //=====================================================================
    //	SYSTEM MEMORY
    //=====================================================================

    //=====================================================================
    //	BOX
    //=====================================================================

    // allocate boxes
    box_cpu = (box_str*)malloc(dim_cpu.box_mem);

    // initialize number of home boxes
    nh = 0;
    // home boxes in z direction
    for(i=0; i<dim_cpu.boxes1d_arg; i++) {
        // home boxes in y direction
        for(j=0; j<dim_cpu.boxes1d_arg; j++) {
            // home boxes in x direction
            for(k=0; k<dim_cpu.boxes1d_arg; k++) {

                // current home box
                box_cpu[nh].x = k;
                box_cpu[nh].y = j;
                box_cpu[nh].z = i;
                box_cpu[nh].number = nh;
                box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

                // initialize number of neighbor boxes
                box_cpu[nh].nn = 0;

                // neighbor boxes in z direction
                for(l=-1; l<2; l++) {
                    // neighbor boxes in y direction
                    for(m=-1; m<2; m++) {
                        // neighbor boxes in x direction
                        for(n=-1; n<2; n++) {

                            // check if (this neighbor exists) and (it is not the same as home box)
                            if(		(((i+l)>=0 && (j+m)>=0 && (k+n)>=0)==true && ((i+l)<dim_cpu.boxes1d_arg && (j+m)<dim_cpu.boxes1d_arg && (k+n)<dim_cpu.boxes1d_arg)==true)	&&
                                    (l==0 && m==0 && n==0)==false	) {

                                // current neighbor box
                                box_cpu[nh].nei[box_cpu[nh].nn].x = (k+n);
                                box_cpu[nh].nei[box_cpu[nh].nn].y = (j+m);
                                box_cpu[nh].nei[box_cpu[nh].nn].z = (i+l);
                                box_cpu[nh].nei[box_cpu[nh].nn].number =	(box_cpu[nh].nei[box_cpu[nh].nn].z * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg) +
                                        (box_cpu[nh].nei[box_cpu[nh].nn].y * dim_cpu.boxes1d_arg) +
                                        box_cpu[nh].nei[box_cpu[nh].nn].x;
                                box_cpu[nh].nei[box_cpu[nh].nn].offset = box_cpu[nh].nei[box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;

                                // increment neighbor box
                                box_cpu[nh].nn = box_cpu[nh].nn + 1;
number_nn += box_cpu[nh].nn;
                            }

                        } // neighbor boxes in x direction
                    } // neighbor boxes in y direction
                } // neighbor boxes in z direction

                // increment home box
                nh = nh + 1;

            } // home boxes in x direction
        } // home boxes in y direction
    } // home boxes in z direction
printf("\n#nn : %d\n",number_nn);
    //=====================================================================
    //	PARAMETERS, DISTANCE, CHARGE AND FORCE
    //=====================================================================

    // random generator seed set to random value - time in this case
    srand(time(NULL));

    FILE *file;

    // input (distances)
    if( (file = fopen(input_distance, "wb" )) == 0 )
        printf( "The file 'input_distances' was not opened\n" );

    // input (distances)
    rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
    for(i=0; i<dim_cpu.space_elem; i=i+1) {
        rv_cpu[i].v = (rand()%10 + 1) / 10.0; // get a number in the range 0.1 - 1.0
        rv_cpu[i].x = (rand()%10 + 1) / 10.0; // get a number in the range 0.1 - 1.0
        rv_cpu[i].y = (rand()%10 + 1) / 10.0; // get a number in the range 0.1 - 1.0
        rv_cpu[i].z = (rand()%10 + 1) / 10.0; // get a number in the range 0.1 - 1.0
        fwrite(&(rv_cpu[i].v), 1, sizeof(double), file);
        fwrite(&(rv_cpu[i].x), 1, sizeof(double), file);
        fwrite(&(rv_cpu[i].y), 1, sizeof(double), file);
        fwrite(&(rv_cpu[i].z), 1, sizeof(double), file);
    }

    fclose(file);

    // input (charge)
    if( (file = fopen(input_charges, "wb" )) == 0 )
        printf( "The file 'input_charges' was not opened\n" );
    qv_cpu = (fp*)malloc(dim_cpu.space_mem2);
    for(i=0; i<dim_cpu.space_elem; i=i+1) {
        qv_cpu[i] = (rand()%10 + 1) / 10.0; // get a number in the range 0.1 - 1.0
        fwrite(&(qv_cpu[i]), 1, sizeof(double), file);
    }
    fclose(file);

    // output (forces)
    fv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
    for(i=0; i<dim_cpu.space_elem; i=i+1) {
        fv_cpu[i].v = 0; // set to 0, because kernels keeps adding to initial value
        fv_cpu[i].x = 0; // set to 0, because kernels keeps adding to initial value
        fv_cpu[i].y = 0; // set to 0, because kernels keeps adding to initial value
        fv_cpu[i].z = 0; // set to 0, because kernels keeps adding to initial value
    }


    //=====================================================================
    //	KERNEL
    //=====================================================================

    //=====================================================================
    //	GPU_OPENCL
    //=====================================================================
    printf("executin kernel\n");
    kernel_gpu_opencl_wrapper(	par_cpu,
                                dim_cpu,
                                box_cpu,
                                rv_cpu,
                                qv_cpu,
                                fv_cpu);


    //=====================================================================
    //	SYSTEM MEMORY DEALLOCATION
    //=====================================================================

    printf("saving results\n");
    // dump results
    if( (file = fopen(output_gold, "wb" )) == 0 )
        printf( "The file 'output_forces' was not opened\n" );
    int number_zeros = 0;
    int higher_zero = 0;
    int lower_zero = 0;
    for(i=0; i<dim_cpu.space_elem; i=i+1) {
        if(fv_cpu[i].v == 0.0)
            number_zeros++;
        if(fv_cpu[i].v > 0.0)
            higher_zero++;
        if(fv_cpu[i].v < 0.0)
            lower_zero++;

        if(fv_cpu[i].x == 0.0)
            number_zeros++;
        if(fv_cpu[i].x > 0.0)
            higher_zero++;
        if(fv_cpu[i].x < 0.0)
            lower_zero++;

        if(fv_cpu[i].y == 0.0)
            number_zeros++;
        if(fv_cpu[i].y > 0.0)
            higher_zero++;
        if(fv_cpu[i].y < 0.0)
            lower_zero++;

        if(fv_cpu[i].z == 0.0)
            number_zeros++;
        if(fv_cpu[i].z > 0.0)
            higher_zero++;
        if(fv_cpu[i].z < 0.0)
            lower_zero++;

        fwrite(&(fv_cpu[i].v), 1, sizeof(double), file);
        fwrite(&(fv_cpu[i].x), 1, sizeof(double), file);
        fwrite(&(fv_cpu[i].y), 1, sizeof(double), file);
        fwrite(&(fv_cpu[i].z), 1, sizeof(double), file);
    }
    fclose(file);

    printf("Total Number of zeros in the output is %d, from %ld numbers\n",number_zeros, (dim_cpu.space_elem*4));


    free(rv_cpu);
    free(qv_cpu);
    free(fv_cpu);
    free(box_cpu);


    return 0;	
}


void kernel_gpu_opencl_wrapper(	par_str par_cpu,
                            dim_str dim_cpu,
                            box_str* box_cpu,
                            FOUR_VECTOR* rv_cpu,
                            fp* qv_cpu,
                            FOUR_VECTOR* fv_cpu)
{

    // timer
    long long time0;
    long long time1;


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
                                        devType,
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
    //const char *source = load_kernel_source(kernel_file);
    size_t sourceSize = strlen(kernel_lavamd_ocl);

    // Create the program
    cl_program program = clCreateProgramWithSource(	context,
                         1,
                         &kernel_lavamd_ocl,
                         &sourceSize,
                         &error);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);

    // parameterized kernel dimension
    char clOptions[110];
    //  sprintf(clOptions,"-I../../src");
    sprintf(clOptions,"-I.");

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
    local_work_size[0] = block_size;
    size_t global_work_size[1];
    global_work_size[0] = dim_cpu.number_boxes * local_work_size[0];

    printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", global_work_size[0]/local_work_size[0], local_work_size[0]);

// Kernel, for each box (dim_cpu.number_boxes) 
//double flop = dim_cpu.number_boxes;
//printf("flop: %f\n",flop);
// iterate for each neighbor of a box (number_nn)
double flop =  number_nn;
// The last for iterate NUMBER_PAR_PER_BOX times 
flop *= NUMBER_PAR_PER_BOX;
// the last for uses 46 operations plus 2 exp() functions
flop *=46;


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


    clSetKernelArg(kernel, 0, sizeof(par_str), (void *) &par_cpu);
    clSetKernelArg(kernel, 1, sizeof(dim_str), (void *) &dim_cpu);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &d_box_gpu);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &d_rv_gpu);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &d_qv_gpu);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &d_fv_gpu);
    clSetKernelArg(kernel, 6, sizeof(int), (void *) &block_size);
    // launch kernel - all boxes
    time0 = get_time();
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

    time1 = get_time();

    double kernel_time = (double) (time1-time0) / 1000000;
    double flops = (double)flop/kernel_time;
    double outputpersec = (double)dim_cpu.space_elem * 4 / kernel_time;
    printf("BOXES:%d BLOCK:%d OUTPUT/S:%f FLOPS:%f\n",boxes,block_size,outputpersec,flops);
    printf("kernel_time:%f\n",kernel_time);

    error = clEnqueueReadBuffer(command_queue,d_fv_gpu,CL_TRUE,0,dim_cpu.space_mem,fv_cpu,0,NULL,NULL);
    if (error != CL_SUCCESS)
        fatal_CL(error, __LINE__);


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

}
