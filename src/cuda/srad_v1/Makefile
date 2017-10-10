TARGET=srad_v1
# CUDA toolkit installation path
CUDA_DIR = /usr/local/cuda

# CUDA toolkit libraries
CUDA_LIB_DIR := $(CUDA_DIR)/lib
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -d $(CUDA_DIR)/lib64; then echo T; else echo F; fi), T)
     	CUDA_LIB_DIR := $(CUDA_DIR)/lib64
     endif
endif

# CUDA SDK installation path
SDK_DIR = /usr/local/cuda-8.0/samples/

ARCH= -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 \
	  -gencode=arch=compute_60,code=sm_60 \
      -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_62,code=sm_62
      
NVCC_FLAGS= $(ARCH) -std=c++11


CC :=gcc
NVCC= $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

ifeq ($(LOGS), 1)
NVCC_FLAGS+= -I../../include/ -L../../include -logHelper -DLOGS
endif


#all
all: $(TARGET)



# link objects(binaries) together
$(TARGET):		main.o
	$(NVCC)	main.o \
				-I$(INCLUDE) \
				-L$(CUDA_LIB_DIR)  \
				-lm -lcuda -lcudart -o $(TARGET) $(NVCC_FLAGS)

# compile main function file into object (binary)
main.o: 	main.cu \
				define.c \
				graphics.c \
				extract_kernel.cu \
				prepare_kernel.cu \
				reduce_kernel.cu \
				srad_kernel.cu \
				srad2_kernel.cu \
				compress_kernel.cu
	$(NVCC)	main.cu -c -O3 $(NVCC_FLAGS)
	
test:
	./$(TARGET) 10 0.5 image.pgm 1 image_out.pgm

generate:
	./$(TARGET) 1 0.5 image.pgm 0 image_out.pgm


# delete all object files
clean:
	rm -rf *.o $(TARGET)
