#!/usr/bin/python

MAXREGS = 186

with open("src/register_kernel.h", "w") as fp:
    fp.write("#ifndef REGISTER_KERNEL_H_\n")
    fp.write("#define REGISTER_KERNEL_H_\n")
    fp.write("\n#include "'"utils.h"'"\n")

    ####################################################################
    #kernel for FFFF which must be or
    fp.write("\n__global__ void test_register_file_kernel_or(uint32 *output_rf1, uint32 *output_rf2, uint32 *output_rf3, uint32 *trip_mem1, uint32 *trip_mem2, uint32 *trip_mem3, uint32 reg_data, const uint64 sleep_cycles) {\n")
    fp.write("\tconst uint32 i =  blockDim.x * blockIdx.x + threadIdx.x;\n")
    #fp.write("\tregister uint32 reg_errors = 0;\n")
    
    #setting the registers
    for i in range(0, MAXREGS):
        fp.write("\tregister uint32 r{} = reg_data | trip_mem1[{}] | trip_mem2[{}] | trip_mem3[{}]".format(i, i, i, i)) # output_rf[i + {}]
        fp.write(";\n")
    
   
          
    fp.write("\n\tsleep_cuda(sleep_cycles);\n\n")
           
        
    #saving the results
    for i in range(0, MAXREGS):
        fp.write("\toutput_rf1[i + {}] = r{};\n".format(i, i))
        fp.write("\toutput_rf2[i + {}] = r{};\n".format(i, i))
        fp.write("\toutput_rf3[i + {}] = r{};\n\n".format(i, i))
            
    for i in range(MAXREGS, MAXREGS + (256 - MAXREGS)):
        fp.write("\toutput_rf1[i + {}] = reg_data;\n".format(i, i))
        fp.write("\toutput_rf2[i + {}] = reg_data;\n".format(i, i))
        fp.write("\toutput_rf3[i + {}] = reg_data;\n\n".format(i, i))
  
    fp.write("}\n\n")
    
    ####################################################################
    # Kernel for 0000 must be AND
    fp.write("\n__global__ void test_register_file_kernel_and(uint32 *output_rf1, uint32 *output_rf2, uint32 *output_rf3, uint32 *trip_mem1, uint32 *trip_mem2, uint32 *trip_mem3, uint32 reg_data, const uint64 sleep_cycles) {\n")
    fp.write("\tconst uint32 i = blockIdx.x * blockIdx.y + threadIdx.x;\n")
    #fp.write("\tregister uint32 reg_errors = 0;\n")
    
    #setting the registers
    for i in range(0, MAXREGS):
        fp.write("\tregister uint32 r{} = reg_data & trip_mem1[{}] & trip_mem2[{}] & trip_mem3[{}]".format(i, i, i, i)) # output_rf[i + {}]
        fp.write(";\n")
    
   
          
    fp.write("\n\tsleep_cuda(sleep_cycles);\n\n")
           
        
    #saving the results
    for i in range(0, MAXREGS):
        fp.write("\toutput_rf1[i + {}] = r{};\n".format(i, i))
        fp.write("\toutput_rf2[i + {}] = r{};\n".format(i, i))
        fp.write("\toutput_rf3[i + {}] = r{};\n\n".format(i, i))
            
    for i in range(MAXREGS, MAXREGS + (256 - MAXREGS)):
        fp.write("\toutput_rf1[i + {}] = reg_data;\n".format(i, i))
        fp.write("\toutput_rf2[i + {}] = reg_data;\n".format(i, i))
        fp.write("\toutput_rf3[i + {}] = reg_data;\n\n".format(i, i))
  
    fp.write("}\n\n")
    
    
    
    fp.write("#endif /* REGISTER_KERNEL_H_ */\n")
