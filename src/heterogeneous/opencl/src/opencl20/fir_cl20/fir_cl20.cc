/* Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:Northeastern University Computer Architecture Research (NUCAR)
 * Group, Northeastern University, http://www.ece.neu.edu/groups/nucar/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 *  with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/
 * or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *   Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimers. Redistributions in binary
 *   form must reproduce the above copyright notice, this list of conditions and
 *   the following disclaimers in the documentation and/or other materials
 *   provided with the distribution. Neither the names of NUCAR, Northeastern
 *   University, nor the names of its contributors may be used to endorse or
 *   promote products derived from this Software without specific prior written
 *   permission.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *   DEALINGS WITH THE SOFTWARE.
 *
 * Calculate a FIR filter with OpenCL 2.0
 *
 * It requires an input signal and number of blocks and number of data as args
 *
 */

#include <stdio.h>/* for printf */
#include <stdint.h>/* for uint64 definition */
#include <stdlib.h>/* for exit() definition */
#include <time.h>/* for clock_gettime */
#include <string.h>
#include <CL/cl.h>
#include "src/opencl20/fir_cl20/include/fir_cl20.h"
// #ifdef GPUPROF
// #include "inc/GPUPerfAPI.h"
// #include <dlfcn.h>
// #endif

#define BILLION 1000000000L

#define CHECK_STATUS(status, message) \
  if (status != CL_SUCCESS) {         \
      printf(message);                \
      printf("\n");                   \
  }

/** Define custom constants*/
#define MAX_SOURCE_SIZE (0x100000)

/* radiation things */
extern "C"
{
#include "../../logHelper/logHelper.h"
}

#define ITERATIONS 10000

cl_int err;

void FIR::SaveGold() {
  char gold_file_str[64];
  sprintf(gold_file_str, "output/output_%d_%d", num_blocks, num_data);

  FILE* gold_file = fopen(gold_file_str, "wb");
  fwrite(output, num_blocks*num_data*sizeof(float), 1, gold_file);

  fclose(gold_file);

}

void FIR::CheckGold() {
  float *gold = (float*) malloc(num_blocks * num_data * sizeof(float));

  char gold_file_str[64];
  sprintf(gold_file_str, "output/output_%d_%d", num_blocks, num_data);

  FILE* gold_file = fopen(gold_file_str, "rb");
  int read = fread(gold, num_blocks*num_data*sizeof(float), 1, gold_file);
  if(read != 1)
    read = -1;

  fclose(gold_file);

  int errors = 0;
  for(unsigned int i = 0; i < num_blocks*num_data; i++)
  {
    	if(abs(gold[i] - output[i]) > 1e-5)
     	{
		errors++;
		
		char error_detail[128];
        	snprintf(error_detail, 64, "position: [%d], output: %f, gold: %f\n", i, output[i], gold[i]);
        	printf("Error: %s\n", error_detail);

#ifdef LOGS
  log_error_detail(error_detail);
#endif

	}  
  }

#ifdef LOGS
  log_error_count(errors);
#endif

  free(gold);

  //std::cout << "There were " << errors << " errors in the output!" << std::endl;
  //std::cout << std::endl;
}


void FIR::Run() {
  //uint64_t diff;
  //struct timespec start, end;

  // Define custom variables
  int i, count;
  int local;

  /** Declare the Filter Properties */
  num_tap = 1024;
  num_total_data = num_data * num_blocks;
  local = 64;

  printf("FIR Filter\n Data Samples : %d Generating inputs and gold: %s \n", num_data, gen_inputs == true ? "YES" : "NO");
  printf("num_blocks : %d \n", num_blocks);
  printf("Local Workgroups : %d\n", local);

  // Load the kernel source code into the array source_str
  FILE *fp;
  char *source_str;
  size_t source_size;

  fp = fopen("fir_cl20_kernel.cl", "r");
  if (!fp) { fp = fopen("src/opencl20/fir_cl20/fir_cl20_kernel.cl", "r"); }
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  // Get device information
  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1,
                       &device_id, &ret_num_devices);

  printf("\n No of Devices %d", ret_num_platforms);

  // Get platform information
  char *platformVendor;
  size_t platInfoSize;
  clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, 0, NULL,
                    &platInfoSize);

  platformVendor = (char*)malloc(platInfoSize);

  clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, platInfoSize,
                    platformVendor, NULL);
  printf("\tVendor: %s\n", platformVendor);
  free(platformVendor);

  // Create an OpenCL context
  cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

  // Create a command queue
  cl_command_queue queue = clCreateCommandQueueWithProperties(context,
                                                              device_id,
                                                              NULL,
                                                              &ret);
  // Allocate SVM buffers
  input = (cl_float *)clSVMAlloc(context, CL_MEM_READ_ONLY,
                                 num_total_data*sizeof(cl_float), 0);
  output = (cl_float *)clSVMAlloc(context, CL_MEM_READ_WRITE,
                                  num_total_data*sizeof(cl_float), 0);
  coeff = (cl_float *)clSVMAlloc(context, CL_MEM_READ_ONLY,
                                 num_tap*sizeof(cl_float), 0);
  temp_output = (cl_float *)clSVMAlloc(context, CL_MEM_READ_WRITE,
                                       (num_data+num_tap-1)*sizeof(cl_float), 0);

  // Map SVM buffers for writing
  err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, input,
                        num_total_data*sizeof(cl_float), 0, 0, 0);
  if (err != CL_SUCCESS) {
    printf("Error clEnqueueSVMMap input :: %i", err);
    exit(1);
  }
  err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, output,
                        num_total_data*sizeof(cl_float), 0, 0, 0);
  if (err != CL_SUCCESS) {
    printf("Error clEnqueueSVMMap output :: %i", err);
    exit(1);
  }
  err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, coeff,
                        num_tap*sizeof(cl_float), 0, 0, 0);
  if (err != CL_SUCCESS) {
    printf("Error clEnqueueSVMMap coeff :: %i", err);
    exit(1);
  }
  err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, temp_output,
                        (num_data+num_tap-1)*sizeof(cl_float), 0, 0, 0);
  if (err != CL_SUCCESS) {
    printf("Error clEnqueueSVMMap temp_output :: %i", err);
    exit(1);
  }

  char input_file_str[64], coeff_file_str[64];
  sprintf(input_file_str, "input/input_%d_%d", num_blocks, num_data);
  sprintf(coeff_file_str, "input/coeff_%d_%d", num_blocks, num_data);

  if(gen_inputs == true)
  {
  	// Initialize the input data
  	for (i = 0; (unsigned)i < num_total_data; i++) {
    	  input[i] = 8;
          output[i] = 99;
  	}

  	for (i = 0; (unsigned)i < num_tap; i++)
          coeff[i] = 1.0/num_tap;
  
	FILE* input_file = fopen(input_file_str, "wb");
	fwrite(input, num_total_data*sizeof(float), 1, input_file);
	fclose(input_file);

	FILE* coeff_file = fopen(coeff_file_str, "wb");
	fwrite(coeff, num_tap*sizeof(float), 1, coeff_file);
	fclose(coeff_file);
  }

  else
  {
	int read;

	FILE* input_file = fopen(input_file_str, "rb");
	read = fread(input, num_total_data*sizeof(float), 1, input_file);
	if(read != 1) 
		read = -1;

	fclose(input_file);

	FILE* coeff_file = fopen(coeff_file_str, "rb");
	read = fread(coeff, num_tap*sizeof(float), 1, coeff_file);
	if(read != 1)
		read = -1;

	fclose(coeff_file);
  }

  for (i=0; (unsigned)i < (num_data+num_tap-1); i++)
    temp_output[i] = 0.0;

	
 
 /* 
#if 1
  // Read the input file
  FILE *fip;
  i = 0;
  fip = fopen("temp.dat", "r");
  if (!fip) { fip = fopen("src/opencl20/fir_cl20/input/temp.dat", "r"); }
  if (!fip) { fip = fopen("input/temp.dat", "r"); }
  if (!fip) { fprintf(stderr, "Unable to locate accessory file.\n"); exit(1);}
  while ((unsigned)i < num_total_data) {
    //int res = fscanf(fip, "%f", &input[i]);
      i++;
    }
  fclose(fip);
#if 0
  printf("\n The Input:\n");
  i = 0;
  while (i < num_total_data) {
      printf("%f, ", input[i]);
      i++;
  }
#endif
#endif
*/

  // Done writing, unnmap SVM buffers
  err = clEnqueueSVMUnmap(queue, input, 0, 0, 0);
  if (err != CL_SUCCESS) {
    printf("Error clEnqueueSVMUnmap input :: %i", err);
    exit(1);
  }

  err = clEnqueueSVMUnmap(queue, output, 0, 0, 0);
  if (err != CL_SUCCESS) {
    printf("Error clEnqueueSVMUnmap output :: %i", err);
    exit(1);
  }

  err = clEnqueueSVMUnmap(queue, coeff, 0, 0, 0);
  if (err != CL_SUCCESS) {
    printf("Error clEnqueueSVMUnmap coeff :: %i", err);
    exit(1);
  }

  // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource(context, 1,
                                                 (const char **)&source_str,
                                                 (const size_t *)&source_size,
                                                 &ret);

  // Build the program
  ret = clBuildProgram(program, 1, &device_id,
                       "-I ./ -cl-std=CL2.0", NULL, NULL);

  CHECK_STATUS(ret, "Error: Build Program\n");

  // Create the OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, "FIR", &ret);
  CHECK_STATUS(ret, "Error: Create kernel. (clCreateKernel)\n");

  // Set the arguments of the kernel
  // Use clSetKernelArgSVMPointer to set SVM pointers as arguments
  ret = clSetKernelArgSVMPointer(kernel, 0, output);
  ret = clSetKernelArgSVMPointer(kernel, 1, coeff);
  ret = clSetKernelArgSVMPointer(kernel, 2, temp_output);
  ret = clSetKernelArg(kernel, 3, sizeof(cl_uint), (void *)&num_tap);
  // Not a SVM pointer

  // Decide the local group size formation
  size_t global_threads[1]={num_data};
  size_t local_threads[1]={128};
  //cl_command_type cmdType;

//initialize logs
#ifdef LOGS
  char test_info[100];
  snprintf(test_info, 100, "blocks:%d, data:%d", num_blocks, num_data);
  char test_name[100];
  snprintf(test_name, 100, "openclFIR");
  start_log_file(test_name, test_info);
  set_max_errors_iter(500);
  set_iter_interval_print(10);
#endif

//begin loop of iterations
  for(int iteration = 0; iteration < (gen_inputs == true ? 1 : ITERATIONS); iteration++)
  {
	if(iteration % 10 == 0)
		std::cout << "Iteration #" << iteration << std::endl;

//start iteration
#ifdef LOGS
  start_iteration();
#endif  

	// FIR Loop
  	//uint64_t execTimeMs = 0.0;
  	//  double execTimeMs = 0.0f;

 	/* measure monotonic time */
  	//clock_gettime(CLOCK_MONOTONIC, &start);

  	count = 0;
  	while ((unsigned)count < num_blocks) {
    		// Custom item size based on current algorithm
    		//size_t global_item_size = num_data;
    		//size_t local_item_size = num_data;
   	 	// Execute the OpenCL kernel on the list
    		cl_event event;
    		ret = clEnqueueNDRangeKernel(queue,
                	                 	kernel,
                        	        	1,
                                	 	NULL,
                                 		global_threads,
                                 		local_threads,
                                 		0,
                                 		NULL,
                                 		&event);

    		CHECK_STATUS(ret, "Error: Range kernel. (clCreateKernel)\n");
    		clFinish(queue);


      		count++;
	}

// Flush memory buffers
  ret = clFlush(queue);
  ret = clFinish(queue);

//end iteration
#ifdef LOGS
  end_iteration();
#endif

  	if(gen_inputs == true)
		SaveGold();

  	else
		CheckGold();  
  }

//end log file
#ifdef LOGS
  end_log_file();
#endif

  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  clSVMFree(context, input);
  clSVMFree(context, output);
  clSVMFree(context, coeff);
  clSVMFree(context, temp_output);
  ret = clReleaseCommandQueue(queue);
  ret = clReleaseContext(context);

  //clock_gettime(CLOCK_MONOTONIC, &end);
  //diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  //printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
}


