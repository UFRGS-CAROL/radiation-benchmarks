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
//#include "src/hsa/fir_hsa/kernels.h"
//#include "src/hsa/fir_hsa/fir_benchmark.h"

#include "kernels.h"
#include "fir_benchmark.h"


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


  // Initialize input data
  for (unsigned int i = 0; i < numTotalData; i++) {
    input[i] = i;
  }

  // Initialize coefficient
  for (unsigned int i = 0; i < numTap; i++) {
    coeff[i] = 1.0 / numTap;
  }

  // Initialize temp output
  for (unsigned int i = 0; i < (numData + numTap - 1); i++) {
    temp_output[i] = 0.0;
  }
}

void FirBenchmark::Run() {
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
  unsigned int errCount = 0;
  unsigned int totErrCount = 0;
  printf("errorBuffer size (flat global size): %u\n",rmt_buff_size);
  for(int i=0; i<rmt_buff_size;i++){
    if(P.errorBuffer[i]>0){
      errCount++;
      totErrCount+=P.errorBuffer[i];
    }
  }
  printf("Threads with errors: %u\n",errCount);
  printf("Total errors: %u\n\n",totErrCount);

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
