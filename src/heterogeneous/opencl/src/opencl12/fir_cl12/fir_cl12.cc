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
 * Calculate a FIR filter with OpenCL 1.2
 *
 * It requires an input signal and number of blocks and number of data as args
 *
 */
#include <string.h>
#include <time.h>/* for clock_gettime */
#include <stdio.h>
#include <stdint.h>/* for uint64 definition */
#include <stdlib.h>/* for exit() definition */
#include <CL/cl.h>
#include <sys/stat.h>
#include "src/opencl12/fir_cl12/include/fir_cl12.h"
#include "src/opencl12/fir_cl12/include/eventlist.h"

#define BILLION 1000000000L

#define CHECK_STATUS(status, message)           \
  if (status != CL_SUCCESS) {                   \
    printf(message);                            \
    printf("\n");                               \
  }

/** Define custom constants*/
#define MAX_SOURCE_SIZE (0x100000)

/* radiation things */
extern "C"
{
#include "../../logHelper/logHelper.h"
}

#define ITERATIONS 10000

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
  uint64_t diff;
  struct timespec start, end;

  /** Define Custom Variables */
  int i, count;
  int local;

  /** Declare the Filter Properties */
  num_tap = 1024;
  num_total_data = num_data * num_blocks;
  local = 64;

  printf("FIR Filter\n Data Samples : %d Generating inputs: %s\n", num_data, (gen_inputs == true ? "YES" : "NO"));
  printf("num_blocks : %d \n", num_blocks);
  printf("Local Workgroups : %d\n", local);
  // exit(0);
  /** Define variables here */
  input = (cl_float *) malloc( num_total_data* sizeof(cl_float) );
  output = (cl_float *) malloc( num_total_data* sizeof(cl_float) );
  coeff = (cl_float *) malloc( num_tap* sizeof(cl_float) );
  temp_output = (cl_float *) malloc( (num_data+num_tap-1) * sizeof(cl_float) );

  char input_file_str[64], coeff_file_str[64];
  sprintf(input_file_str, "input/input_%d_%d", num_blocks, num_data);
  sprintf(coeff_file_str, "input/coeff_%d_%d", num_blocks, num_data);

  if(gen_inputs == true)
  {
        // Initialize the input data
        for (i = 0; (unsigned)i < num_total_data; i++) {
          input[i] = i;
          output[i] = i;
        }

        for (i = 0; (unsigned)i < num_tap; i++)
          coeff[i] = 1.0/num_tap;

        FILE* input_file = fopen(input_file_str, "wb");
        fwrite(input, num_total_data*sizeof(float), 1, input_file);
        fclose(input_file);

        FILE* coeff_file = fopen(coeff_file_str, "wb");
        fwrite(coeff, num_tap*sizeof(float), 1, coeff_file);
        fclose(coeff_file);

        //VFNETTO
        //for(int c = 0; c < 10; c++)
        //{
        //        printf("INIT: input: %f, output: %f, temp_output: %f\n", input[c], output[c], temp_output[c]);
        //}
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


 // Event Creation
  cl_event event;
  EventList* event_list;

  // Load the kernel source code into the array source_str
  FILE *fp;
  char *source_str;
  size_t source_size;

  fp = fopen("fir_cl12_kernel.cl", "r");
  if (!fp) { fp = fopen("src/opencl12/fir_cl12/fir_cl12_kernel.cl", "r"); }
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  // Get platform and device information
  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1,
                       &device_id, &ret_num_devices);

  printf("/n No of Devices %d", ret_num_platforms);

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
  /* cl_command_queue command_queue = clCreateCommandQueueWithProperties\
     (context, device_id, NULL, &ret); */
  cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, \
                                 CL_QUEUE_PROFILING_ENABLE, 0};
  cl_command_queue command_queue = \
    clCreateCommandQueueWithProperties(context, device_id, props, &ret);
  // Create event_list for Timestamps
  event_list = new EventList(context, command_queue, device_id, true);

  // Create memory buffers on the device for each vector
  cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      sizeof(cl_float) * num_data, NULL, &ret);
  cl_mem output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       sizeof(cl_float) * num_data, NULL, &ret);
  cl_mem coeffBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      sizeof(cl_float) * num_tap, NULL, &ret);
  cl_mem temp_output_buffer = clCreateBuffer(context,
                                            CL_MEM_READ_WRITE,
                                            sizeof(cl_float)*(num_data+num_tap-1),
                                            NULL,
                                            &ret);

  // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource(context, 1,
                                                 (const char **)&source_str,
                                                 (const size_t *)&source_size,
                                                 &ret);
  // Build the program
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

  CHECK_STATUS(ret, "Error: Build Program\n");

  // Create the OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, "FIR", &ret);
  CHECK_STATUS(ret, "Error: Create kernel. (clCreateKernel)\n");

  // Set the arguments of the kernel
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&output_buffer);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&coeffBuffer);
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&temp_output_buffer);
  ret = clSetKernelArg(kernel, 3, sizeof(cl_uint), (void *)&num_tap);

  // Initialize Memory Buffer
  ret = clEnqueueWriteBuffer(command_queue,
                             coeffBuffer,
                             1,
                             0,
                             num_tap * sizeof(cl_float),
                             coeff,
                             0,
                             0,
                             &event);

  event_list->add(event);

  ret = clEnqueueWriteBuffer(command_queue,
                             temp_output_buffer,
                             1,
                             0,
                             (num_tap) *sizeof(cl_float),
                             temp_output,
                             0,
                             0,
                             &event);

  event_list->add(event);

  // Decide the local group formation
  size_t globalThreads[1]={num_data};
  size_t localThreads[1]={128};
  //cl_command_type cmdType;
  count = 0;

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

//start iteration
#ifdef LOGS
  start_iteration();
#endif

  clock_gettime(CLOCK_MONOTONIC, &start);/* mark start time */

  while ((unsigned)count < num_blocks) {
    /* fill in the temp_input buffer object */
    ret = clEnqueueWriteBuffer(command_queue,
                               temp_output_buffer,
                               1,
                               (num_tap-1)*sizeof(cl_float),
                               num_data * sizeof(cl_float),
                               input + (count * num_data),
                               0,
                               0,
                               &event);

    // (num_tap-1)*sizeof(cl_float)
    event_list->add(event);

    //size_t global_item_size = num_data;
    // GLOBAL ITEMSIZE IS CUSTOM BASED ON COMPUTAION ALGO
    //size_t local_item_size = num_data;
    // size_t local_item_size[4] = {num_data/4,num_data/4,num_data/4,num_data/4};
    // LOCAL ITEM SIZE IS CUSTOM BASED ON COMPUTATION ALGO

    // Execute the OpenCL kernel on the list
    ret = clEnqueueNDRangeKernel(command_queue,
                                 kernel,
                                 1,
                                 NULL,
                                 globalThreads,
                                 localThreads,
                                 0,
                                 NULL,
                                 &event);

    CHECK_STATUS(ret, "Error: Range kernel. (clCreateKernel)\n");
    clFinish(command_queue);
//    ret = clWaitForEvents(1, &event);
//    ret = clWaitForEvents(1, &event);

    event_list->add(event);

//end iteration
#ifdef LOGS
  end_iteration();
#endif

    /* Get the output buffer */
    ret = clEnqueueReadBuffer(command_queue,
                              output_buffer,
                              CL_TRUE,
                              0,
                              num_data * sizeof(cl_float),
                              output + count * num_data,
                              0,
                              NULL,
                              &event);
    event_list->add(event);
    count++;
  }

  clock_gettime(CLOCK_MONOTONIC, &end);/* mark the end time */

//end log file
#ifdef LOGS
  end_log_file();
#endif

  /* Uncomment to print output */
  // printf("\n The Output:\n");
  // i = 0;
  // while( i<10 )
  // {
  //   printf( "%f ", output[i] );
  //  i++;
  // }

  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(inputBuffer);
  ret = clReleaseMemObject(output_buffer);
  ret = clReleaseMemObject(coeffBuffer);
  ret = clReleaseMemObject(temp_output_buffer);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);

  free(input);
  free(output);
  free(coeff);
  free(temp_output);

  /* Ensure that eventDumps exists */
  mkdir("eventDumps", 0700);
    
  /* comment to hide timing events */
  //event_list->printEvents();
  event_list->dumpEvents((char *)"eventDumps");

  diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  printf("elapsed time = %llu nanoseconds\n", (long long unsigned int) diff);
}
