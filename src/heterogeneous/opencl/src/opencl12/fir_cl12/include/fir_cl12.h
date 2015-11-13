#ifndef FIR_H
#define FIR_H

#include "src/common/cl_util/cl_util.h"
#include <CL/cl.h>
#include "src/common/benchmark/benchmark.h"

using namespace clHelper;

class FIR : public Benchmark
{
private:
    cl_uint num_tap = 0;
    cl_uint num_data = 0;  // Block size
    cl_uint num_total_data = 0;
    cl_uint num_blocks = 0;
    cl_float* input = NULL;
    cl_float* output = NULL;
    cl_float* coeff = NULL;
    cl_float* temp_output = NULL;
 
    bool gen_inputs;   
public:
        FIR() {};
        ~FIR() {};

    void SetInitialParameters(int data, int blocks, bool g_inputs) { num_blocks = blocks; num_data = data; gen_inputs = g_inputs; }
    void Initialize() override {}
	void Run() override;
	void Verify() override {}
	void Cleanup() override {}
	void Summarize() override {}

    void SaveGold();
    void CheckGold();
};

#endif
