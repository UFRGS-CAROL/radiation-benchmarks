#!/bin/bash

set -ex

/usr/local/cuda/bin/nvcc -ccbin g++  -I../../../common/include -I../../Common -I../../common/inc \
			-I./include -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc \
			-I/src/cuda/common/include -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 \
			-gencode arch=compute_75,code=compute_75 cudaTensorCoreGEMMTest.cu  -o cudaTensorCoreGEMMTest   -L/usr/local/cuda/lib64 -lcublas

