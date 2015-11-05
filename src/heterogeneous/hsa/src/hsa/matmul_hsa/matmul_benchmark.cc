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
 * Author: Xiangyu Li (xili@ece.neu.edu)
 * Modified by: Yifan Sun (yifansun@coe.neu.edu)
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

#include "src/hsa/matmul_hsa/matmul_benchmark.h"

#include <memory>

/* radiation things */
extern "C"
{
#include "../../logHelper/logHelper.h"
}

#define ITERATIONS 10000


MatmulBenchmark::MatmulBenchmark() {
  workGroupSize = 64;
  maxIter = 1000;
}

//NOT USED!
void MatmulBenchmark::ExecKernel() {
  global_work_size[0] = nr * workGroupSize;
  local_work_size[0] = workGroupSize;

  SNK_INIT_LPARM(lparm, 0);
  lparm->ldims[0] = local_work_size[0];
  lparm->gdims[0] = nr * workGroupSize;

  for (int j = 0; j < maxIter; j++) {
    if (j % 2 == 0) {
      pageRank_kernel(nr, rowOffset, col, val, sizeof(float) * 64, vector,
                      eigenV, lparm);
    } else {
      pageRank_kernel(nr, rowOffset, col, val, sizeof(float) * 64, eigenV,
                      vector, lparm);
    }
  }
}



void MatmulBenchmark::Initialize() {
  }

void MatmulBenchmark::Run() {
  std::cout << "Running Matrix Multiplication. Generating inputs and gold: " << (gen_inputs == true ? "YES" : "NO") << std::endl;
  std::cout << std::endl;

//initialize logs
#ifdef LOGS
  char test_info[100];
  snprintf(test_info, 100, "size:%d", input_size);
  char test_name[100];
  snprintf(test_name, 100, "hsaMatmul");
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
 
  	// Execute the kernel
  	ExecKernel();

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



void MatmulBenchmark::SaveGold() {
  char gold_file_str[64];
  sprintf(gold_file_str, "output/output_%d", nr);

  FILE* gold_file = fopen(gold_file_str, "wb");
  fwrite(eigenV, nr*sizeof(float), 1, gold_file);
  
  for(int i = 0; i < 10; i++)
   printf("%f \n", eigenV[i]);   

  fclose(gold_file);

}

void MatmulBenchmark::CheckGold() {
  float *gold = (float*) malloc(nr * sizeof(float));

  char gold_file_str[64];
  sprintf(gold_file_str, "output/output_%d", nr);

  FILE* gold_file = fopen(gold_file_str, "rb");
  int read = fread(gold, nr*sizeof(float), 1, gold_file);
  if(read != 1)
    read = -1;

  fclose(gold_file);
  
  int errors = 0;
  for(int i = 0; i < nr; i++)
  {
    if(abs(gold[i] - eigenV[i]) > 1e-5)
    {
	errors++;
    	
        char error_detail[128];
	snprintf(error_detail, 64, "position: [%d], output: %f, gold: %f\n", i, eigenV[i], gold[i]);
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

void MatmulBenchmark::Summarize() {}

void MatmulBenchmark::Cleanup() { FreeBuffer(); }

int MatmulBenchmark::GetLength() { return nr; }

float MatmulBenchmark::abs(float num) {
  if (num < 0) {
    num = -num;
  }
  return num;
}


