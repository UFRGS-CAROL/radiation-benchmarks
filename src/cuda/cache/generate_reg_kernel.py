#!/usr/bin/python

MAXREGS = 256

with open("src/register_kernel.h", "w") as fp:
    fp.write("#ifndef REGISTER_KERNEL_H_\n")
    fp.write("#define REGISTER_KERNEL_H_\n")
    fp.write("\n#include "'"utils.h"'"\n")

    ####################################################################
    #kernel for FFFF which must be or
    fp.write("\n__global__ void test_register_file_kernel_or(uint32 *rf1, uint32 *rf2, uint32 *rf3, uint32 *mem1, uint32 *mem2, uint32 *mem3, uint32 reg_data, const uint64 sleep_cycles) {\n")
    fp.write("\tconst uint32 i =  blockDim.x * blockIdx.x * 256 + threadIdx.x;\n")
    #fp.write("\tregister uint32 reg_errors = 0;\n")
    
    #setting the registers
    for i in range(0, MAXREGS):
        fp.write("\tregister uint32 r{} = mem1[{}]".format(i, i))  #| mem1[{}] | mem2[{}] | mem3[{}]".format(i, i, i, i))
        fp.write(";\n")
    
   
          
    fp.write("\n\tsleep_cuda(sleep_cycles);\n\n")
           
        
    #saving the results
    for i in range(0, MAXREGS):
        fp.write("\trf1[i + {}] = r{};\n".format(i, i))
        fp.write("\trf2[i + {}] = r{};\n".format(i, i))
        fp.write("\trf3[i + {}] = r{};\n\n".format(i, i))
            
    #for i in range(MAXREGS, MAXREGS + (256 - MAXREGS)):
        #fp.write("\trf1[i + {}] = reg_data;\n".format(i, i))
        #fp.write("\trf2[i + {}] = reg_data;\n".format(i, i))
        #fp.write("\trf3[i + {}] = reg_data;\n\n".format(i, i))
  
    fp.write("}\n\n")
    fp.write("#endif /* REGISTER_KERNEL_H_ */\n")
    exit(1)
    
    
    ####################################################################
    # Kernel for 0000 must be AND
    fp.write("\n__global__ void test_register_file_kernel_and(uint32 *rf1, uint32 *rf2, uint32 *rf3, uint32 *mem1, uint32 *mem2, uint32 *mem3, uint32 reg_data, const uint64 sleep_cycles) {\n")
    fp.write("\tconst uint32 i = blockIdx.x * blockIdx.y + threadIdx.x;\n")
    #fp.write("\tregister uint32 reg_errors = 0;\n")
    
    #setting the registers
    for i in range(0, MAXREGS):
        fp.write("\tregister uint32 r{} = reg_data & mem1[{}] & mem2[{}] & mem3[{}]".format(i, i, i, i))
        fp.write(";\n")
    
   
          
    fp.write("\n\tsleep_cuda(sleep_cycles);\n\n")
           
        
    #saving the results
    for i in range(0, MAXREGS):
        fp.write("\trf1[i + {}] = r{};\n".format(i, i))
        fp.write("\trf2[i + {}] = r{};\n".format(i, i))
        fp.write("\trf3[i + {}] = r{};\n\n".format(i, i))
            
    for i in range(MAXREGS, MAXREGS + (256 - MAXREGS)):
        fp.write("\trf1[i + {}] = reg_data;\n".format(i, i))
        fp.write("\trf2[i + {}] = reg_data;\n".format(i, i))
        fp.write("\trf3[i + {}] = reg_data;\n\n".format(i, i))
  
    fp.write("}\n\n")
    
    
    
    fp.write("#endif /* REGISTER_KERNEL_H_ */\n")
