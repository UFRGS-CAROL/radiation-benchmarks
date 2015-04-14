#define _CRT_SECURE_NO_WARNINGS // AVOID build errors in sprintf and fopen

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <string.h>

#include <CL/cl.h>
#include <sys/time.h>

#include "/home/carol/log_helper/log_helper.h"

#define SIZE 8192
#define GOLD_MATRIX_PATH "/home/carol/TestGPU/GenerateGoldMatrix/Double_GOLD_8192.matrix"
#define LOGFILE_MATRIXNAME "openclmxm8192"

// #define PLATFORM_ID 0
#define DEVICE_ID 0

#define BLOCK_SIZE 32

#define SWITCH_CHAR  '-'


#define N_ERRORS_LOG 500

#define ITERACTIONS 100000000000000

double *h_A;
double *h_B;
double *h_GOLD;

 
void ReadMatrixFromFile()
{
	FILE *f_A, *f_B, *f_GOLD;

printf("open matrix...");
        f_A = fopen("/home/carol/TestGPU/GenerateGoldMatrix/Double_A_8192.matrix","rb");
        f_B = fopen("/home/carol/TestGPU/GenerateGoldMatrix/Double_B_8192.matrix","rb");
        f_GOLD = fopen(GOLD_MATRIX_PATH,"rb");

	if (!(f_A && f_B && f_GOLD)) { printf ("Error opening matrix.\n"); exit(-1); }

printf("read...");
        fread(h_A,sizeof(double)*SIZE*SIZE, 1, f_A);
        fread(h_B,sizeof(double)*SIZE*SIZE, 1, f_B);
        fread(h_GOLD,sizeof(double)*SIZE*SIZE, 1, f_GOLD);


        fclose(f_A);
	fclose(f_B);
	fclose(f_GOLD);
}


int main(int argc, char** argv)
{
	std::cout << "Size of double on host: " << (sizeof(double)) << " bytes\n";
 
	int size = SIZE * SIZE;
	int mem_size = size * sizeof (double);
	h_A = (double*) malloc(mem_size);
	h_B = (double*) malloc(mem_size);
	h_GOLD = (double*) malloc(mem_size);
	if (!(h_A && h_B && h_GOLD)) std::cout << "Error while trying to alloc host memory.\n";

	int wSize = SIZE;

	double time_st;

	// =========> Log vars

	int ea=0; //wrong integers in the current loop
	int t_ea=0; //total number of wrong integers
	int old_ea = 0;

	double total_time = 0.0;

	int i, j, loop2;

	int kernel_errors=0;
	int zero = 0;


	char test_info[100];
	snprintf(test_info, 100, "size:%dx%d",SIZE,SIZE);
	start_log_file(LOGFILE_MATRIXNAME, test_info);
	set_max_errors_iter(500);
	set_iter_interval_print(10);

	// =========> OpenCl vars
	cl_context clGPUContext;
	cl_command_queue clCommandQue;
	cl_program clProgram;
	cl_kernel clKernelMxM;
	cl_kernel clKernelGoldChk;
 
	size_t dataBytes;
	size_t kernelLength;
	cl_int errcode;
 
	cl_mem d_A;
	cl_mem d_B;
	cl_mem d_C;
	cl_mem d_GOLD;
	cl_mem d_KERRORS;

	cl_int numplat;
	cl_platform_id platforms[5];
 
	clGetPlatformIDs (2, platforms, (cl_uint*)&numplat);
	std::cout << "OpenCL Platforms available  : " << numplat << "\n";
	// This returns a platform list available on the system, on my system this means:
	//	[0] = Intel Core i7 (4Cores, 8Logical threads) / OpenCL1.2
	//			Intel HD4600 Integrated Graphics / OpenCL1.2
	//	[1] = NVIDIA GTX850m Dedicated Graphics / CUDA 6.5 / OpenCL1.1


	// Setup OpenCL context and device for NVIDIA GTX850m
	cl_context_properties props[3];
	props[0] = (cl_context_properties)CL_CONTEXT_PLATFORM;  // indicates that next element is platform
	props[1] = (cl_context_properties)platforms[numplat-1];  // platform is of type cl_platform_id
	props[2] = (cl_context_properties)0;   // last element must be 0

	clGPUContext = clCreateContextFromType(props,
					CL_DEVICE_TYPE_GPU,				// It could be CL_DEVICE_TYPE_ALL as on the selected platform this is the only computing device, btw...
					NULL, NULL, &errcode);
	if (errcode!=CL_SUCCESS) std::cout << "error clCreateContextFromType : " << errcode << "\n";
 
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

	if (errcode!=CL_SUCCESS) std::cout << "error clGetContextInfo : " << errcode << "\n";

	char clName[50];

	errcode = clGetDeviceInfo(	clDevices[DEVICE_ID],
						CL_DEVICE_NAME,
						sizeof(char)*50,
						clName,
						NULL);
	if (errcode!=CL_SUCCESS) std::cout << "error clGetDeviceInfo : " << errcode << "\n";
	std::cout << "Device name : " << clName << "\n";

	cl_device_fp_config clDeviceDoubleCapability;

	errcode = clGetDeviceInfo(	clDevices[DEVICE_ID],
							0x1032, // = CL_DEVICE_DOUBLE_FP_CONFIG (line commented on cl.h)
							sizeof(cl_device_fp_config),
							&clDeviceDoubleCapability,
							NULL);
	if (errcode!=CL_SUCCESS) std::cout << "error clGetDeviceInfo : " << errcode << "\n";
	std::cout << "Device fully support double extension? : ";
	if (clDeviceDoubleCapability==(CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_DENORM)) std::cout << "Yes\n";
	else std::cout << "No\n";

	clCommandQue = clCreateCommandQueue(clGPUContext, 
					clDevices[DEVICE_ID], 0, &errcode);
	if (errcode!=CL_SUCCESS) std::cout << "error clCommandQue : " << errcode << "\n";

	// Load and build OpenCL kernel
	std::cout << "Loading and building opencl kernels...";
	std::ifstream kernelfile("/home/carol/DSN15_codes/openclmxm/Kernel.cl"); // This will read the file to the memory as OpenCL needs to compile it from there
	std::string kernelstr((std::istreambuf_iterator<char>(kernelfile)),
				std::istreambuf_iterator<char>());
	const char* clMatrixMul = kernelstr.c_str();
	kernelLength=kernelstr.length();
 
	clProgram = clCreateProgramWithSource(clGPUContext, 
				1, (const char **)& clMatrixMul, 
				&kernelLength, &errcode);
	if (errcode!=CL_SUCCESS) std::cout << "error clCreateProgramWithSource : " << errcode << "\n";
 
	errcode = clBuildProgram(clProgram, 0, 
				NULL, NULL, NULL, NULL);

	if (errcode!=CL_SUCCESS) std::cout << "error clBuildProgram : " << errcode << "\n";
	if (errcode==CL_BUILD_PROGRAM_FAILURE) {
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(clProgram, clDevices[DEVICE_ID], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char *log = (char *) malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(clProgram, clDevices[DEVICE_ID], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
	}
 
	clKernelMxM = clCreateKernel(clProgram,	// Load kernel named matrixMul (you can have multiple kernels on the same source)
				"matrixMul", &errcode);
	if (errcode!=CL_SUCCESS) std::cout << "error clCreateKernel MxM : " << errcode << "\n";

	clKernelGoldChk = clCreateKernel(clProgram,
				"GoldChk", &errcode);
	if (errcode!=CL_SUCCESS) std::cout << "error clCreateKernel GoldChk : " << errcode << "\n";


	std::cout << "Load matrix from file...";
	ReadMatrixFromFile();

	for(loop2=0; loop2<ITERACTIONS; loop2++)
	{
		kernel_errors=0;

		// Setup device memory (and copy matrixes from host to device)
		d_A = clCreateBuffer(clGPUContext, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				mem_size, h_A, &errcode); // Load matrix A
		if (errcode!=CL_SUCCESS) std::cout << "error clCreateBuffer d_A : " << errcode << "\n";

		d_B = clCreateBuffer(clGPUContext, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				mem_size, h_B, &errcode); // Load matrix B
		if (errcode!=CL_SUCCESS) std::cout << "error clCreateBuffer d_B : " << errcode << "\n";

		d_C = clCreateBuffer(clGPUContext, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
				mem_size, h_B, &errcode); // Load matrix C, filled with B erase prev values.
		if (errcode!=CL_SUCCESS) std::cout << "error clCreateBuffer d_C : " << errcode << "\n";
		
 
 
		// ========> Launch OpenCL kernel
		size_t localWorkSize[2], globalWorkSize[2], local;
 
		// Set all kernel function parameters
		errcode = clSetKernelArg(clKernelMxM, 0, 
					sizeof(cl_mem), (void *)&d_A);
		errcode |= clSetKernelArg(clKernelMxM, 1, 
					sizeof(cl_mem), (void *)&d_B);
		errcode |= clSetKernelArg(clKernelMxM, 2, 
					sizeof(cl_mem), (void *)&d_C);
		errcode |= clSetKernelArg(clKernelMxM, 3, 
					sizeof(int), (void *)&wSize);
		if (errcode!=CL_SUCCESS) std::cout << "error clSetKernelArg MxM : " << errcode << "\n";

		size_t wgsize;


		localWorkSize[0] = BLOCK_SIZE;	// There will be SIZE x SIZE compute units (threads in cuda)
		localWorkSize[1] = BLOCK_SIZE;

		globalWorkSize[0] = SIZE;	// There will be SIZE x SIZE compute units (threads in cuda)
		globalWorkSize[1] = SIZE;

	
		start_iteration();
		// DETAIL: localWorkSize is set to 0 so opencl set the best partition of global worksizes automagically
		// /\ This don't work as it would, it will only set the first dimension to the max worksize (1024) and the other dims to 1.
		// Put clKernel in the execution queue of the device context
		errcode = clEnqueueNDRangeKernel(clCommandQue, 
					clKernelMxM, 2, NULL, globalWorkSize, 
					localWorkSize, 0, NULL, NULL);
		if (errcode!=CL_SUCCESS) std::cout << "error clEqnueueNDRangeKernel MxM : " << errcode << "\n";


		clFinish(clCommandQue);		// Wait for opencl finish the tasks, maybe not needed because of clEnqueueReadBuffer

		end_iteration();


		clReleaseMemObject(d_A);
		clReleaseMemObject(d_B);

		// Load GOLD on GPU to check gold
		d_GOLD = clCreateBuffer(clGPUContext, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				mem_size, h_GOLD, &errcode); // Load matrix GOLD
		if (errcode!=CL_SUCCESS) std::cout << "error clCreateBuffer d_GOLD : " << errcode << "\n";

		d_KERRORS = clCreateBuffer(clGPUContext, 
				CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
				sizeof (int), &kernel_errors, &errcode); // Load kerrors var
		if (errcode!=CL_SUCCESS) std::cout << "error clCreateBuffer d_KERRORS : " << errcode << "\n";
	
		// Set all kernel function parameters
		errcode = clSetKernelArg(clKernelGoldChk, 0, 
					sizeof(cl_mem), (void *)&d_GOLD);
		errcode |= clSetKernelArg(clKernelGoldChk, 1, 
					sizeof(cl_mem), (void *)&d_C);
		errcode |= clSetKernelArg(clKernelGoldChk, 2, 
					sizeof(int), (void *)&wSize);
		errcode |= clSetKernelArg(clKernelGoldChk, 3, 
					sizeof(cl_mem), (void *)&d_KERRORS);
		if (errcode!=CL_SUCCESS) std::cout << "error clSetKernelArg GoldChk : " << errcode << "\n";

		errcode = clEnqueueNDRangeKernel(clCommandQue, 
					clKernelGoldChk, 2, NULL, globalWorkSize, 
					localWorkSize, 0, NULL, NULL);
		if (errcode!=CL_SUCCESS) std::cout << "error clEqnueueNDRangeKernel GoldChk: " << errcode << "\n";


		clFinish(clCommandQue);		// Wait for opencl finish the tasks, maybe not needed because of clEnqueueReadBuffer


		clReleaseMemObject(d_GOLD);

		errcode = clEnqueueReadBuffer(clCommandQue, 
					d_KERRORS, CL_TRUE, 0, sizeof(int), 
					&kernel_errors, 0, NULL, NULL);
		if (errcode!=CL_SUCCESS) std::cout << "error clEnqueueReadBuffer on d_KERRORS : " << errcode << "\n";


		ea = 0;
		t_ea += kernel_errors;

		log_error_count(kernel_errors);	

		if (kernel_errors>0)
		{
			std::cout << "Error detected! kerrors = " << kernel_errors << "\n";

			// Retrieve result from device on matrix A
			errcode = clEnqueueReadBuffer(clCommandQue, 
						d_C, CL_TRUE, 0, mem_size, 
						h_A, 0, NULL, NULL);
			if (errcode!=CL_SUCCESS) std::cout << "error clEnqueueReadBuffer on d_C : " << errcode << "\n";

			char error_detail[150];

			for(i=0; (i<SIZE) && (ea < N_ERRORS_LOG); i++)
			{
				for(j=0; (j<SIZE) && (ea < N_ERRORS_LOG); j++)
				{
					if ((fabs((h_A[i+SIZE*j]-h_GOLD[i+SIZE*j])/h_A[i+SIZE*j]) > 0.0000000001)||(fabs((h_A[i+SIZE*j]-h_GOLD[i+SIZE*j])/h_GOLD[i+SIZE*j]) > 0.0000000001))
					{
						//ea++;
						//fprintf(file, "\n p: [%d, %d], r: %1.16e, e: %1.16e, error: %d\n", i, j, h_A[i + SIZE * j], h_GOLD[i + SIZE * j], ea);
						snprintf(error_detail, 150, "p: [%d, %d], r: %1.16e, e: %1.16e", i, j, h_A[i + SIZE * j], h_GOLD[i + SIZE * j]);
						log_error_detail(error_detail);
										
					}
				}
			}


			ReadMatrixFromFile();	
		}

		if(kernel_errors > 0 || (loop2 % 10 == 0))
		{
			printf("\ntest number: %d", loop2);
			printf("\nerrors: %d", kernel_errors);
		}

		clReleaseMemObject(d_C);
		clReleaseMemObject(d_KERRORS);
	}

	end_log_file();


	free(h_A);
	free(h_B);
	free(h_GOLD);
 
	free(clDevices);
	clReleaseContext(clGPUContext);
	clReleaseKernel(clKernelMxM);
	clReleaseKernel(clKernelGoldChk);
	clReleaseProgram(clProgram);
	clReleaseCommandQueue(clCommandQue);

}
