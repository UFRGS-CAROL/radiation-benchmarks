################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/ConvolutionalLayerKernel.cu \
../src/FullyConnectedLayerKernel.cu \
../src/MaxpoolingLayerKernel.cu \
../src/OutputLayerKernel.cu 

CPP_SRCS += \
../src/ConvNet.cpp \
../src/ConvolutionalLayer.cpp \
../src/FullyConnectedLayer.cpp \
../src/Layer.cpp \
../src/LogsProcessing.cpp \
../src/MNISTParser.cpp \
../src/MaxpoolingLayer.cpp \
../src/OutputLayer.cpp \
../src/Util.cpp \
../src/compare_layers.cpp \
../src/cudaUtil.cpp \
../src/main.cpp 

OBJS += \
./src/ConvNet.o \
./src/ConvolutionalLayer.o \
./src/ConvolutionalLayerKernel.o \
./src/FullyConnectedLayer.o \
./src/FullyConnectedLayerKernel.o \
./src/Layer.o \
./src/LogsProcessing.o \
./src/MNISTParser.o \
./src/MaxpoolingLayer.o \
./src/MaxpoolingLayerKernel.o \
./src/OutputLayer.o \
./src/OutputLayerKernel.o \
./src/Util.o \
./src/compare_layers.o \
./src/cudaUtil.o \
./src/main.o 

CU_DEPS += \
./src/ConvolutionalLayerKernel.d \
./src/FullyConnectedLayerKernel.d \
./src/MaxpoolingLayerKernel.d \
./src/OutputLayerKernel.d 

CPP_DEPS += \
./src/ConvNet.d \
./src/ConvolutionalLayer.d \
./src/FullyConnectedLayer.d \
./src/Layer.d \
./src/LogsProcessing.d \
./src/MNISTParser.d \
./src/MaxpoolingLayer.d \
./src/OutputLayer.d \
./src/Util.d \
./src/compare_layers.d \
./src/cudaUtil.d \
./src/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_32,code=compute_32 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_37,code=compute_37 -gencode arch=compute_50,code=compute_50 -gencode arch=compute_52,code=compute_52 -gencode arch=compute_53,code=compute_53 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


