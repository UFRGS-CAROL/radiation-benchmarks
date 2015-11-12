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

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "kernels.h"
#include "fir_benchmark.h"

/* radiation things */
extern "C"
{
#include "../../logHelper/logHelper.h"
}

#define ITERATIONS 10000000

void FirBenchmark::Initialize() {
  numTap = 1024;
  numTotalData = numData * numBlocks;
  local = 64;

  input = new float[numTotalData];
  output = new float[numTotalData];
  coeff = new float[numTap];
  temp_output = new float[numData + numTap - 1];


  char input_file_str[64], coeff_file_str[64];
  sprintf(input_file_str, "input/input_%d_%d", numBlocks, numData);
  sprintf(coeff_file_str, "input/coeff_%d_%d", numBlocks, numData);

  if(gen_inputs == true)
  {
        // Initialize input data
        for (unsigned int i = 0; i < numTotalData; i++) {
          input[i] = (float)rand()/(float)(RAND_MAX/1000);
          output[i] = 0;//(float)rand()/(float)(RAND_MAX/1000);
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

        /*for (unsigned int i = 0; i < 10; i++)
        {
          printf("input %d: %f\ncoeff %d: %f\n", i, input[i], i, coeff[i]);
        }*/
  }

  else
  {
        int read;

        FILE* input_file = fopen(input_file_str, "rb");
        if(input == NULL)
        {
                printf("can't open input file!\n");
                exit(1);
        }

        else
        {
                read = fread(input, numTotalData*sizeof(float), 1, input_file);
                if(read != 1)
                        read = -1;

                fclose(input_file);
        }

        FILE* coeff_file = fopen(coeff_file_str, "rb");
        if(coeff_file == NULL)
        {
                printf("can't open coeff file!\n");
                exit(1);
        }

        else
        {
                read = fread(coeff, numTap*sizeof(float), 1, coeff_file);
                if(read != 1)
                        read = -1;

                fclose(coeff_file);
        }

  }

    // CKALRA, DLOWELL  
  ebsize = numData;
//  if (ebsize % BLOCK_SIZE != 0)
//   ebsize = (ebsize / BLOCK_SIZE + 1) * BLOCK_SIZE;
  errorBuffer = (unsigned int*)calloc(sizeof(unsigned int), ebsize);
}

void FirBenchmark::Run() {
  std::cout << "Running FIR RMT INTRA. Generating inputs and gold: " << (gen_inputs == true ? "YES" : "NO") << std::endl;
  std::cout << std::endl;

//initialize logs
#ifdef LOGS
  char test_info[100];
  snprintf(test_info, 100, "blocks:%d, data:%d", numBlocks, numData);
  char test_name[100];
  snprintf(test_name, 100, "hsaFIR_RMT_INTRA");
  start_log_file(test_name, test_info);
  set_max_errors_iter(500);
  set_iter_interval_print(10);
#endif

//begin loop of iterations
  //for(int iteration = 0; iteration < (gen_inputs == true ? 1 : ITERATIONS); iteration++)
  {
  	// Initialize temp output
	for (unsigned int i = 0; i < (numData + numTap - 1); i++) {
    	  temp_output[i] = input[i];
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
    		lparm->gdims[0] = 2*numData;
    		lparm->ldims[0] = 2*128;

		memset(errorBuffer, 0, ebsize*sizeof(unsigned int));
    		FIR(output+i*numData, coeff, temp_output, numTap, errorBuffer, lparm);
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

  /*for(unsigned int i = numData*numBlocks-10; i < numData*numBlocks; i++)
    printf("output %d: %e\n", i, output[i]);*/
}

void FirBenchmark::CheckGold() {
  float *gold = (float*) malloc(numBlocks * numData * sizeof(float));
  
  char gold_file_str[64];
  sprintf(gold_file_str, "output/output_%d_%d", numBlocks, numData);

  FILE* gold_file = fopen(gold_file_str, "rb");
  int read = fread(gold, numBlocks*numData*sizeof(float), 1, gold_file);
  if(read != 1)
    read = -1;

  fclose(gold_file);

  int errors = 0;
  for(unsigned int i = 0; i < numBlocks*numData; i++)
  {
    if(abs(gold[i] - output[i]) > 1e-5 || output[i] == 0)
    {
        errors++;

        char error_detail[128];
        snprintf(error_detail, 128, "position: [%d], output: %f, gold: %f\n", i, output[i], gold[i]);

        if(errors < 10 ) printf("%s", error_detail);

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
  for(unsigned int i=0;i<ebsize;i++){
    if(errorBuffer[i]>0){
      errCount++;
      totErrCount+=errorBuffer[i];
    }
  }

  if(errCount > 0)
  {
    snprintf(rmt_error_detail, 128, "err_buf size: %u, thread_err: %u, total_err: %u\n", ebsize, errCount, totErrCount);
    printf("%s", rmt_error_detail);

#ifdef LOGS
  log_error_detail(rmt_error_detail);
#endif

  }
  //printf("Threads with errors: %u\n",errCount);
  //printf("Total errors: %u\n\n",totErrCount);

#ifdef LOGS
  log_error_count(errors+errCount);
#endif

  free(gold);
}

void FirBenchmark::Verify() {
  for (unsigned int i = 0; i < numTotalData; i++) {
    printf("output[i] = %f\n", output[i]);
  }
}

void FirBenchmark::Summarize() {
}

void FirBenchmark::Cleanup() {
  delete[] input;
  delete[] output;
  delete[] coeff;
  delete[] temp_output;
}
