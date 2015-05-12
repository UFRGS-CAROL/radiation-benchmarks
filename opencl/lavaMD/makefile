#C_C = gcc
#OMP_LIB = -lgomp
#OMP_FLAG = -fopenmp
#INC=/opt/AMDAPPSDK-2.9-1/include
#LIB=/opt/AMDAPPSDK-2.9-1/lib/x86_64
#OCL_LIB =	-L$(LIB) \
#			-lOpenCL
#OCL_INC = -I$(INC)
CC=gcc

NVIDIA_INC=-I/usr/local/cuda-6.5/include/
NVIDIA_LIB=-L/usr/local/cuda-6.5/lib64

INTEL_INC=-I/opt/intel/opencl-1.2-sdk-4.6.0.92/include/
INTEL_LIB=-L/opt/intel/opencl-1.2-4.5.0.8/lib64/

AMD_INC=-I/opt/AMDAPPSDK-3.0-0-Beta/include/
AMD_LIB=-L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/

SOURCES_GEN=main.c ./kernel/kernel_gpu_opencl_wrapper.c ./util/timer/timer.c ./util/opencl/opencl.c


amd: amd_gen

amd_gen: $(SOURCES_GEN)
	$(CC) $(SOURCES_GEN) -lm -lgomp -fopenmp -lOpenCL $(AMD_LIB) $(AMD_INC) -o lavamd_gen

intel: intel_gen

intel_gen: $(SOURCES_GEN)
	$(CC) $(SOURCES_GEN) -lm -lgomp -fopenmp -lOpenCL $(INTEL_LIB) $(INTEL_INC) -o lavamd_gen

nvidia: nvidia_gen

nvidia_gen: $(SOURCES_GEN)
	$(CC) $(SOURCES_GEN) -lm -lgomp -fopenmp -lOpenCL $(NVIDIA_LIB) $(NVIDIA_INC) -o lavamd_gen

# link objects together
#a.out:		main.o \
#			./kernel/kernel_gpu_opencl_wrapper.o \
#			./util/timer/timer.o \
#			./util/opencl/opencl.o
#	$(C_C)	main.o \
#			./kernel/kernel_gpu_opencl_wrapper.o \
#			./util/timer/timer.o \
#			./util/opencl/opencl.o \
#			-lm \
#			$(OCL_LIB) \
#			-o lavaMD
#
## compile function files into objects
#main.o:		main.h \
#			main.c \
#			./kernel/kernel_gpu_opencl_wrapper.h \
#			./kernel/kernel_gpu_opencl_wrapper.c \
#			./util/timer/timer.h \
#			./util/timer/timer.c \
#			./util/opencl/opencl.h \
#			./util/opencl/opencl.c
#	$(C_C)	$(OUTPUT) $(KERNEL_DIM) main.c \
#			-c \
#			-o main.o \
#			-O3
#
#./kernel/kernel_gpu_opencl_wrapper.o:	./kernel/kernel_gpu_opencl_wrapper.h \
#										./kernel/kernel_gpu_opencl_wrapper.c
#	$(C_C)	$(KERNEL_DIM)						./kernel/kernel_gpu_opencl_wrapper.c \
#										$(OCL_INC) \
#										-c \
#										-o ./kernel/kernel_gpu_opencl_wrapper.o \
#										-O3
#
#
#./util/timer/timer.o:	./util/timer/timer.h \
#						./util/timer/timer.c
#	$(C_C)				./util/timer/timer.c \
#						-c \
#						-o ./util/timer/timer.o \
#						-O3
#
#./util/opencl/opencl.o:	./util/opencl/opencl.h \
#						./util/opencl/opencl.c
#	$(C_C)				./util/opencl/opencl.c \
#						$(OCL_INC) \
#						-c \
#						-o ./util/opencl/opencl.o \
#						-O3

# delete all object and executable files
clean:
	rm	*.o lavaMD \
		./kernel/*.o \
		./util/timer/*.o \
		./util/opencl/*.o \
