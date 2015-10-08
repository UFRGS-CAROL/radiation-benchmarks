#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <vector>

#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_ext.h>

#define NUM_BUFFERS 8
#define GPU_QUEUE 0
#define CPU_QUEUE 1

using namespace std;


bool scanCPU(double *data, double* reference, double* dev_result, const size_t size, int buf)
{
    bool passed = true;

    double last = 0.0f;
    for (unsigned int i = 0; i < size; ++i)
    {
        reference[i] = data[i] + last;
        last = reference[i];
    }
    
    int c = 0;
    for (unsigned int i = 0; i < size; ++i)
    {
        if (reference[i] != dev_result[i])
        {
	    if(c < 25)
		{
            cout << "[" << buf << "] " << "Mismatch at i: " << i << " ref: " << reference[i]
                 << " dev: " << dev_result[i] << endl;
            passed = false;
			c++;
		}
	}
    }

/*  for (unsigned int i = 0; i < size; ++i)
    {
        printf("%lf\n", dev_result[i]);
    }*/
return passed;

}



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

void getOpenCLCodeFileText(char* filePath, char* fileText)
{
	FILE* openCLFile = fopen(filePath, "r");
	if(openCLFile == NULL)
	{
		printf("Can't open file %s\n\n", filePath);
		exit(1);
	}

	else
	{
		fseek(openCLFile, 0, SEEK_END);
		long fileSize = ftell(openCLFile);
		rewind(openCLFile);

                fileText = (char*) malloc(sizeof(char) * fileSize);		
		if(fread(fileText, sizeof(char), fileSize, openCLFile) != fileSize)
		{
			printf("Can't read file %s\n\n", filePath);
			exit(1);
		}

		fileText[fileSize] = '\0';

		fclose(openCLFile);
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
cl_command_queue        cmd_queue[2]; //0 = gpu, 1 = cpu
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
    cmd_queue[0] = clCreateCommandQueue(context, device_id[0], 0, &ret);
    if (ret != CL_SUCCESS)
    {
        printf("\nError at clCreateCommandQueue! Error code %i\n\n", ret);
        exit(1);
    }

    cmd_queue[1] = clCreateCommandQueue(context, device_id[1], 0, &ret);
    if (ret != CL_SUCCESS)
    {
        printf("\nError at clCreateCommandQueue! Error code %i\n\n", ret);
        exit(1);
    }
   
}

cl_int createProgram(cl_context context, const char** cl_source, cl_device_id device, cl_program* prog)
{
    cl_int err = 0;    
    *prog = clCreateProgramWithSource(context,
                                      1,
                                      cl_source,
                                      NULL,
                                      &err);
    CL_CHECK_ERROR(err);

    string compileFlags;
    if (checkExtension(device, "cl_khr_fp64"))
    {
        compileFlags = "-DK_DOUBLE_PRECISION ";
    }
    else if (checkExtension(device, "cl_amd_fp64"))
    {
        compileFlags = "-DAMD_DOUBLE_PRECISION ";
    }
    // Before proceeding, make sure the kernel code compiles and
    // all kernels are valid.
    cout << "Compiling scan kernels." << endl;
    err = clBuildProgram(*prog, 1, &device, compileFlags.c_str(), NULL, NULL);
    CL_CHECK_ERROR(err);

    if (err != CL_SUCCESS)
    {
	cl_int terr = err;
        char log[20486];
        size_t retsize = 0;
        err = clGetProgramBuildInfo(*prog, device, CL_PROGRAM_BUILD_LOG, 20486
                * sizeof(char), log, &retsize);

        CL_CHECK_ERROR(err);
        cout << "Build error." << endl;
        cout << "Retsize: " << retsize << endl;
        cout << "Log: " << log << endl;
        return terr;
    }

    return CL_SUCCESS;
}

cl_int createKernels(cl_program* prog, cl_kernel* scan)
{
    cl_int err = 0;

    *scan = clCreateKernel(*prog, "scan", &err);
    CL_CHECK_ERROR(err);
}

void* createHostBuffer(cl_context context, long bytes, cl_command_queue command_queue)
{
    cl_int err = 0;

    cl_mem buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            bytes, NULL, &err);
    CL_CHECK_ERROR(err);
    
    void* retBuf = clEnqueueMapBuffer(command_queue, buf, true,
            CL_MAP_READ|CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &err);
    CL_CHECK_ERROR(err);
    clFinish(command_queue);
    return retBuf;
}

cl_mem createDeviceBuffer(cl_context context, long bytes)
{
    cl_int err = 0;

    cl_mem buf = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    CL_CHECK_ERROR(err);

    return buf;
}

void writeDeviceBuffer(cl_command_queue command_queue, cl_mem devBuf, long bytes, void* hostBuf)
{
    cl_int err = 0;

    err = clEnqueueWriteBuffer(command_queue, devBuf, true, 0, bytes, hostBuf, 0,
            NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clFinish(command_queue);
    CL_CHECK_ERROR(err);
}

void readBufferFromDevice(cl_command_queue command_queue, cl_mem devBuf, long bytes, void* hostBuf)
{
    cl_int err = 0;

    err = clEnqueueReadBuffer(command_queue, devBuf, true, 0, bytes, hostBuf,
                0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clFinish(command_queue);
    CL_CHECK_ERROR(err);

}

void executeKernel(cl_command_queue command_queue, cl_kernel kernel, const size_t global_work, const size_t local_work)
{
    cl_int err = 0;
    err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                        &global_work, &local_work, 0, NULL, NULL);
    CL_CHECK_ERROR(err);
    err = clFinish(command_queue);
    CL_CHECK_ERROR(err);
}

void setScanKernelArgs(cl_kernel* scan, cl_mem o0, cl_mem o1, cl_mem o2, cl_mem o3, cl_mem o4, cl_mem o5, cl_mem o6, cl_mem o7, cl_mem i0, cl_mem i1, cl_mem i2, cl_mem i3, cl_mem i4, cl_mem i5, cl_mem i6, cl_mem i7, int size, const int offset)
{
    cl_int err = 0;

    err = clSetKernelArg(*scan, 0, sizeof(cl_mem), (void*)&o0);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(*scan, 1, sizeof(cl_mem), (void*)&o1);
    CL_CHECK_ERROR(err);  
    err = clSetKernelArg(*scan, 2, sizeof(cl_mem), (void*)&o2);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(*scan, 3, sizeof(cl_mem), (void*)&o3);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(*scan, 4, sizeof(cl_mem), (void*)&o4);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(*scan, 5, sizeof(cl_mem), (void*)&o5);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(*scan, 6, sizeof(cl_mem), (void*)&o6);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(*scan, 7, sizeof(cl_mem), (void*)&o7);
    CL_CHECK_ERROR(err);

    err = clSetKernelArg(*scan, 8, sizeof(cl_mem), (void*)&i0);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(*scan, 9, sizeof(cl_mem), (void*)&i1);
    CL_CHECK_ERROR(err);  
    err = clSetKernelArg(*scan, 10, sizeof(cl_mem), (void*)&i2);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(*scan, 11, sizeof(cl_mem), (void*)&i3);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(*scan, 12, sizeof(cl_mem), (void*)&i4);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(*scan, 13, sizeof(cl_mem), (void*)&i5);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(*scan, 14, sizeof(cl_mem), (void*)&i6);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(*scan, 15, sizeof(cl_mem), (void*)&i7);
    CL_CHECK_ERROR(err);

    err = clSetKernelArg(*scan, 16, sizeof(cl_int), (void*)&size);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(*scan, 17, sizeof(cl_int), (void*)&offset);
    CL_CHECK_ERROR(err);
}


int main(int argc, char** argv)
{
    if(argc < 2)
    {
    	cout << "Missing parameter!\nSelect distribution:\n";
    	cout << "0 = 100,0% gpu - 0%     cpu\n";
    	cout << "1 = 87,50% gpu - 12,50% cpu\n";
    	cout << "2 = 75,00% gpu - 25,00% cpu\n";
    	cout << "3 = 62,50% gpu - 37,50% cpu\n";
    	cout << "4 = 50,00% gpu - 50,00% cpu\n";
    	cout << "5 = 37,50% gpu - 62,50% cpu\n";
    	cout << "6 = 25,00% gpu - 75,00% cpu\n";
    	cout << "7 = 12,50% gpu - 87,50% cpu\n";
    	cout << "8 = 0%     gpu - 100,0% cpu\n\n";   
    	exit(1);
    }

    cout < "\n";
    int distr = atoi(argv[1]);
    cl_int err = 0;

    // Initialize OpenCL
    getDevices(CL_DEVICE_TYPE_ALL);

    char* cl_source_scan;
    
    FILE* openCLFile = fopen("scan.cl", "r");
    if(openCLFile == NULL)
    {
           printf("Can't open file %s\n\n", "scan.cl");
           exit(1);
    }
     
    fseek(openCLFile, 0, SEEK_END);
    long fileSize = ftell(openCLFile);
    rewind(openCLFile);

    cl_source_scan = (char*) malloc(sizeof(char) * fileSize);
    if(fread(cl_source_scan, sizeof(char), fileSize, openCLFile) != fileSize)
    {
            printf("Can't read file %s\n\n", "scan.cl");
            exit(1);
    }

    cl_source_scan[fileSize] = '\0';
    fclose(openCLFile);

    // Program Setup
    cl_program gpuProg, cpuProg;

    err = createProgram(context, (const char**) &cl_source_scan, device_id[GPU_QUEUE], &gpuProg);
    CL_CHECK_ERROR(err);   
    if(err != CL_SUCCESS)
    {
	printf("Error: creating program failed!\n\n");
	exit(1);
    }

    err = createProgram(context, (const char**) &cl_source_scan, device_id[CPU_QUEUE], &cpuProg);  
    CL_CHECK_ERROR(err);
    if(err != CL_SUCCESS)
    {
	printf("Error: creating program failed!\n\n");
	exit(1);
    }


    cl_kernel gpuScan, cpuScan;
 
   // Extract out the 3 kernels
    err = createKernels(&gpuProg, &gpuScan);
    CL_CHECK_ERROR(err);
    if(err != CL_SUCCESS)
    {
        printf("Error: creating kernels failed!\n\n");
        exit(1);
    }

    err = createKernels(&cpuProg, &cpuScan);
    CL_CHECK_ERROR(err);
    if(err != CL_SUCCESS)
    {
        printf("Error: creating kernels failed!\n\n");
        exit(1);
    }

    // Problem Sizes
    int probSizes[7] = { 1, 8, 32, 64, 128, 256, 512 };
    int size = probSizes[5];

    // Convert to MB
    //size = (size * 1024 * 1024) / sizeof(double);

    // Create input data on CPU
    unsigned int bytes = size * sizeof(double);
    double* reference = new double[size];

    // Allocate pinned host memory for input data
    double* h_inputs[NUM_BUFFERS];
    cl_mem d_outputs[8];
    double* h_outputs[NUM_BUFFERS];
    cl_mem d_inputs[8];
    
    cout << "Initializing host memory." << endl;
    for(int i = 0; i < NUM_BUFFERS; i++)
    {
	h_inputs[i] = (double*)createHostBuffer(context, bytes, cmd_queue[GPU_QUEUE]);
    
    	// Allocate pinned host memory for output data
    	h_outputs[i] = (double*)createHostBuffer(context, bytes, cmd_queue[GPU_QUEUE]);
    
	// Initialize host memory
    	srand((unsigned)time(NULL));
    	for (int j = 0; j < size; j++)
    	{
        	h_inputs[i][j] = rand(); //Fill with some pattern
        	h_outputs[i][j] = -1;
    	}

    	// Allocate device memory for input array
	d_inputs[i] = createDeviceBuffer(context, bytes);

    	// Allocate device memory for output array
	d_outputs[i] = createDeviceBuffer(context, bytes);
    }

    // Number of local work items per group
    const size_t local_wsize  = 256;

    // Number of global work items
    const size_t global_wsize = size * 8; // i.e. 64 work groups
    
    // Repeat the test multiplie times to get a good measurement
    int passes = 1;
    int temp_iters = 1;
    int total_iters = 0;

    while(temp_iters < size)
    {
	temp_iters *= 2;
	total_iters++;
    }

    double tk_gpu = 0, tk_cpu = 0;
    double temp_tk_gpu, temp_tk_cpu;

    cl_mem cpu_d_inputs[8];
    cl_mem cpu_d_outputs[8];
    double* cpu_h_inputs[NUM_BUFFERS];

    double th = mysecond();

    // Copy data to GPU
    cout << "Copying input data to device." << endl;
    for(int i = 0; i < NUM_BUFFERS; i++)
    {    
	writeDeviceBuffer(cmd_queue[GPU_QUEUE], d_inputs[i], bytes, h_inputs[i]);
    	writeDeviceBuffer(cmd_queue[GPU_QUEUE], d_outputs[i], bytes, h_inputs[i]);
    }

    //Allocate device memory for input array
    for(int i = 0; i < NUM_BUFFERS; i++)
    {
	cpu_d_inputs[i] = createDeviceBuffer(context, bytes);
	cpu_d_outputs[i] = createDeviceBuffer(context, bytes);
    		
	cpu_h_inputs[i] = (double*)createHostBuffer(context, bytes, cmd_queue[CPU_QUEUE]);
    }

    cout << "\nRunning benchmark: " << NUM_BUFFERS << " vectors with size " << size << endl;
    for (int k = 0; k < passes; k++)
    {
	int final_size_gpu = size/pow(2, distr);
	temp_iters = 1;
        	
	cout << "\nGPU running:\n";
        for (int offset = 1; offset < final_size_gpu; offset *= 2)
        {
	    printf("iteration %d of %d\n", temp_iters, total_iters);
	    // Set the kernel arguments for the reduction kernel
	    setScanKernelArgs(&gpuScan,
		d_outputs[0], d_outputs[1], d_outputs[2], d_outputs[3], d_outputs[4], d_outputs[5], d_outputs[6], d_outputs[7],
		d_inputs[0], d_inputs[1], d_inputs[2], d_inputs[3], d_inputs[4], d_inputs[5], d_inputs[6], d_inputs[7],
		size, offset);

	    // For scan, we use a reduce-then-scan approach

            // Each thread block gets an equal portion of the
            // input array, and computes the sum.
            temp_tk_gpu = mysecond();
            executeKernel(cmd_queue[GPU_QUEUE], gpuScan, global_wsize, local_wsize);
            temp_tk_gpu = mysecond() - temp_tk_gpu;
   
	    tk_gpu += temp_tk_gpu;
            temp_iters++;
        }

	for (int i = 0; i < NUM_BUFFERS; i++)
	{
		readBufferFromDevice(cmd_queue[GPU_QUEUE], d_outputs[i], bytes, cpu_h_inputs[i]);

		writeDeviceBuffer(cmd_queue[CPU_QUEUE], cpu_d_inputs[i], bytes, cpu_h_inputs[i]);
		writeDeviceBuffer(cmd_queue[CPU_QUEUE], cpu_d_outputs[i], bytes, cpu_h_inputs[i]);
	}

	cout << "\nCPU running:\n";
	for (int offset = final_size_gpu; offset < size; offset *= 2)
	{
	    printf("iteration %d of %d\n", temp_iters, total_iters);
	    // Set the kernel arguments for the reduction kernel
	    setScanKernelArgs(&cpuScan,
		cpu_d_outputs[0], cpu_d_outputs[1], cpu_d_outputs[2], cpu_d_outputs[3], cpu_d_outputs[4], cpu_d_outputs[5], cpu_d_outputs[6], cpu_d_outputs[7],
		cpu_d_inputs[0], cpu_d_inputs[1], cpu_d_inputs[2], cpu_d_inputs[3], cpu_d_inputs[4], cpu_d_inputs[5], cpu_d_inputs[6], cpu_d_inputs[7],
		size, offset);

	    // For scan, we use a reduce-then-scan approach

            // Each thread block gets an equal portion of the
            // input array, and computes the sum.
            temp_tk_cpu = mysecond();
            executeKernel(cmd_queue[CPU_QUEUE], cpuScan, global_wsize, local_wsize);
            temp_tk_cpu = mysecond() - temp_tk_cpu;
   
	    tk_cpu += temp_tk_cpu;
            temp_iters++;
	}

	double totalScanTime = mysecond() - th;

	
	for(int i = 0; i < NUM_BUFFERS; i++)
        {
	        readBufferFromDevice(cmd_queue[CPU_QUEUE], cpu_d_outputs[i], bytes, h_outputs[i]);

	        // If answer is incorrect, stop test and do not report performance
	        if (! scanCPU(h_inputs[i], reference, h_outputs[i], size, i))
        	{
            		printf("Error computing scan, incorrect answer\n");
            		return 1;
        	}

	}

	if(distr == 0) tk_cpu = 0;
	if(distr == 8) tk_gpu = 0;

       	printf("\n\nGPU scan time: %f\n", tk_gpu);
	printf("CPU scan time: %f\n", tk_cpu);
	printf("Total kernel time: %f\n\n", totalScanTime);

       	char atts[1024];
       //	double avgTime = totalScanTime / (double) total_iters;
       //	double gbs = (double) (size * sizeof(double)) / (1000. * 1000. * 1000.);
       //	cout << "kernel time: " << avgTime << "\ngbs: " << gbs << " GB/s\n";
    
    	
    }

    for(int i = 0; i < NUM_BUFFERS; i++)
    {
    	// Clean up device memory
    	err = clReleaseMemObject(d_inputs[i]);
    	CL_CHECK_ERROR(err);
    	err = clReleaseMemObject(d_outputs[i]);
	CL_CHECK_ERROR(err);
	err = clReleaseMemObject(cpu_d_inputs[i]);
    	CL_CHECK_ERROR(err);
    	err = clReleaseMemObject(cpu_d_outputs[i]);
	CL_CHECK_ERROR(err);
	
    }

    // Clean up pinned host memory
    //err = clEnqueueUnmapMemObject(command_queue, h_i, h_idata, 0, NULL, NULL);
    //CL_CHECK_ERROR(err);
    //err = clEnqueueUnmapMemObject(command_queue, h_o, h_odata, 0, NULL, NULL);
    //CL_CHECK_ERROR(err);
    //err = clReleaseMemObject(h_i);
    //CL_CHECK_ERROR(err);
    //err = clReleaseMemObject(h_o);
    //CL_CHECK_ERROR(err);

    // Clean up other host memory
    delete[] reference;

    err = clReleaseProgram(cpuProg);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(cpuScan);
    CL_CHECK_ERROR(err);

    err = clReleaseProgram(gpuProg);
    CL_CHECK_ERROR(err);
    err = clReleaseKernel(gpuScan);
    CL_CHECK_ERROR(err);

}

