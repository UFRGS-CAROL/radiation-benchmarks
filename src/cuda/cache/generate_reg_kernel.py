#!/usr/bin/python

MAXREGS = 186
LINEWIDTH = 2

with open("src/register_kernel.h", "w") as fp:
    fp.write("#ifndef REGISTER_KERNEL_H_\n")
    fp.write("#define REGISTER_KERNEL_H_\n")
    fp.write("\n__device__ uint64 register_file_errors;\n")
    fp.write("\n__global__ void test_register_file_kernel(uint32 *output_rf, const uint32 reg_data, const uint64 sleep_cycles) {\n")
    fp.write("\tconst uint32 i = blockIdx.x * blockIdx.y + threadIdx.x;\n")
    fp.write("\tuint32 reg_errors = 0;\n")
    
    #setting the registers
    for i in range(0, MAXREGS, LINEWIDTH):
        fp.write("\tregister uint32 ")
        
        for t in range(0, LINEWIDTH):
            fp.write("r{} = output_rf[i + {}]".format(i + t, i + t))
            if t < LINEWIDTH - 1:
                fp.write(", ");
        
        fp.write(";\n")
        
    fp.write("\n\tsleep_cuda(sleep_cycles);\n\n")
    
    #saving the results
    for i in range(0, MAXREGS, LINEWIDTH):
        for t in range(0, LINEWIDTH):
            fp.write("\toutput_rf[i + {}] = r{};\n".format(i + t, i + t))
            fp.write("\tif (r" + str(i + t) + " != reg_data) {\n")
            fp.write("\t\treg_errors++;\n")
            fp.write("\t}\n")
    # ~ for i in range(0, MAXREGS, LINEWIDTH):
        # ~ for t in range(0, LINEWIDTH):
            # ~ fp.write("\toutput_rf[i + {}] = r{};\n".format(i + t, i + t))
            # ~ fp.write("\treg_errors |= (r" + str(i + t) + " ^ reg_data);\n\n")
            
    
    fp.write("\tif (reg_errors != 0) {\n")
    fp.write("\t\tatomicAdd(&register_file_errors, reg_errors);\n")
    fp.write("\t}\n")
    fp.write("\t__syncthreads();\n")
    fp.write("}\n\n")
    fp.write("#endif /* REGISTER_KERNEL_H_ */\n")
