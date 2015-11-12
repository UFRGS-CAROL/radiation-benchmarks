/*
 * Hetero-mark
 *
 * Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:
 *   Northeastern University Computer Architecture Research (NUCAR) Group
 *   Northeastern University
 *   http://www.ece.neu.edu/groups/nucar/
 *
 * Author: Yifan Sun (yifansun@coe.neu.edu)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a 
 * copy of this software and associated documentation files (the "Software"), 
 * to deal with the Software without restriction, including without limitation 
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, 
 * and/or sell copies of the Software, and to permit persons to whom the 
 * Software is furnished to do so, subject to the following conditions:
 * 
 *   Redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimers.
 *
 *   Redistributions in binary form must reproduce the above copyright 
 *   notice, this list of conditions and the following disclaimers in the 
 *   documentation and/or other materials provided with the distribution.
 *
 *   Neither the names of NUCAR, Northeastern University, nor the names of 
 *   its contributors may be used to endorse or promote products derived 
 *   from this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS WITH THE SOFTWARE.
 */
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <iostream>
//#include "src/hsa/fir_hsa/kernels.h"
//#include "src/hsa/fir_hsa/fir_benchmark.h"

#include "kernels.h"
#include "fir_benchmark.h"

/* radiation things */
extern "C"
{
#include "../../logHelper/logHelper.h"
}

#define ITERATIONS 100000000

void FirBenchmark::Initialize() {
  numTap = 1024;
  numTotalData = numData * numBlocks;
  local = 64;

  input = new float[numTotalData];
  output = new float[numTotalData];
  coeff = new float[numTap];
  temp_output = new float[numData + numTap - 1];

  // CKALRA, DLOWELL
  rmt_buff_size = numData;
  P.lockBuffer = (int*)calloc(rmt_buff_size, sizeof(int));
  P.errorBuffer = (unsigned int*)calloc(rmt_buff_size, sizeof(int));
  P.groupID = (unsigned int*)calloc(1, sizeof(int));
  P.addressBuffer = (unsigned int*)calloc(rmt_buff_size, sizeof(int));
  P.valueBuffer = (unsigned long*)calloc(rmt_buff_size, sizeof(unsigned long));

  
  char input_file_str[64], coeff_file_str[64];
  sprintf(input_file_str, "input/input_%d_%d", numBlocks, numData);
  sprintf(coeff_file_str, "input/coeff_%d_%d", numBlocks, numData);

  if(gen_inputs == true)
  {
  	// Initialize input data
  	for (unsigned int i = 0; i < numTotalData; i++) {
    	  input[i] = i;
  	}

  	// Initialize coefficient
  	for (unsigned int i = 0; i < numTap; i++) {
    	  coeff[i] = 1.0 / numTap;
  	}
  
	FILE* input_file = fopen(input_file_str, "wb");
        fwrite(input, numTotalData*sizeof(float), 1, input_file);
        fclose(input_file);

        FILE* coeff_file = fopen(coeff_file_str, "wb");
        fwrite(coeff, numTap*sizeof(float), 1, coeff_file);
        fclose(coeff_file);

  }

  else
  {
        int read;

        FILE* input_file = fopen(input_file_str, "rb");
        read = fread(input, numTotalData*sizeof(float), 1, input_file);
        if(read != 1)
                read = -1;

        fclose(input_file);

        FILE* coeff_file = fopen(coeff_file_str, "rb");
        read = fread(coeff, numTap*sizeof(float), 1, coeff_file);
        if(read != 1)
                read = -1;

        fclose(coeff_file);
  }

}

void FirBenchmark::Run() {
  std::cout << "Running FIR. Generating inputs and gold: " << (gen_inputs == true ? "YES" : "NO") << std::endl;  
  std::cout << std::endl;

//initialize logs
#ifdef LOGS
  char test_info[100];
  snprintf(test_info, 100, "blocks:%d, data:%d", numBlocks, numData);
  char test_name[100];
  snprintf(test_name, 100, "hsaFIR_RMT_INTER");
  start_log_file(test_name, test_info);
  set_max_errors_iter(500);
  set_iter_interval_print(10);
#endif



//begin loop of iterations
  //for(int iteration = 0; iteration < (gen_inputs == true) ? 1 : 1; iteration++)//ITERATIONS); iteration++)
  {
  	// Initialize temp output
  	for (unsigned int i = 0; i < (numData + numTap - 1); i++) {
    	  temp_output[i] = 0.0;
  	}

   	//if(iteration % 10 == 0)
	//	std::cout << "Iteration #" << iteration << std::endl;

//start iteration
#ifdef LOGS
  start_iteration();
#endif

  for (unsigned int i = 0; i < numBlocks; i++) {
 	SNK_INIT_LPARM(lparm, 0);
 	lparm->ndim = 1;
// CKALRA, DLOWELL: double global workgroup size
	lparm->gdims[0] = 2*numData;
	lparm->ldims[0] = 128;

    memset(P.lockBuffer,0,rmt_buff_size*sizeof(int));
    memset(P.errorBuffer, 0, rmt_buff_size*sizeof(unsigned int));
    *(P.groupID) = 0;

    FIR(output, coeff, temp_output, numTap,  P.lockBuffer, P.errorBuffer, P.addressBuffer, P.valueBuffer, P.groupID, lparm);
  }

  if(gen_inputs == true)
		SaveGold();

  	else
		CheckGold();
  
//end iteration
#ifdef LOGS
  end_iteration();
#endif

  }

//end log file
#ifdef LOGS
  end_log_file();
#endif
}

void FirBenchmark::SaveGold() {
  char gold_file_str[64];
  sprintf(gold_file_str, "output/output_%d_%d", numBlocks, numData);

  FILE* gold_file = fopen(gold_file_str, "wb");
  fwrite(output, numBlocks*numData*sizeof(float), 1, gold_file);

  fclose(gold_file);

}

void FirBenchmark::CheckGold() {
  float *gold = (float*) malloc(numBlocks * numData * sizeof(float));

  char gold_file_str[64];
  sprintf(gold_file_str, "output/output_%d_%d", numBlocks, numData);

  FILE* gold_file = fopen(gold_file_str, "rb");
  if(gold_file == NULL)
  {
	printf("can't open gold file!\n");
	exit(1);
  }

  else
  {

  int read = fread(gold, numBlocks*numData*sizeof(float), 1, gold_file);
  if(read != 1)
    read = -1;

  fclose(gold_file);

  int errors = 0;
  for(unsigned int i = 0; i < numBlocks*numData; i++)
  {
    if(gold[i] != output[i])
    {
	errors++;

        char error_detail[128];
        snprintf(error_detail, 64, "position: [%d], output: %e, gold: %e\n", i, output[i], gold[i]);
        printf("Error: %s\n", error_detail);

#ifdef LOGS
  log_error_detail(error_detail);
#endif

    }
  }


char rmt_error_detail[128];

/* CKALRA, DLOWELL: Process errorBuffer, inthebeam test you can just save to file */
  unsigned int errCount = 0;
  unsigned int totErrCount = 0;
  //printf("errorBuffer size (flat global size): %u\n",ebsize);
  for(int i=0;i<rmt_buff_size;i++){
    if(P.errorBuffer[i]>0){
      errCount++;
      totErrCount+=P.errorBuffer[i];
    }
  }

  if(errCount > 0)
  {
    snprintf(rmt_error_detail, 128, "err_buf size: %u, thread_err: %u, total_err: %u\n", rmt_buff_size, errCount, totErrCount);
    printf("%s", rmt_error_detail);

#ifdef LOGS
  log_error_detail(rmt_error_detail);
#endif

  }

#ifdef LOGS
  log_error_count(errors+errCount);
#endif
  }

  free(gold);

  //std::cout << "There were " << errors << " errors in the output!" << std::endl;
  //std::cout << std::endl;
}



void FirBenchmark::Verify() {
  for (unsigned int i = 0; i < numTotalData; i++) {
    printf("output[i] = %f\n", output[i]);
  }
}

void FirBenchmark::Summarize() {
	printf("May the force be with you\n");

}

void FirBenchmark::Cleanup() {
  delete[] input;
  delete[] output;
  delete[] coeff;
  delete[] temp_output;
}
