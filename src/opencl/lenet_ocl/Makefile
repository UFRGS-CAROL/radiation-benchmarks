all: lenetOCL lenetCUDA

CXX=g++

NVCC=nvcc

INCLUDE=./inc/

CFLAGS=-std=c++11 -lOpenCL


lenetOCL:
	$(CXX) convnet/main.cpp -I$(INCLUDE) -o lenetOCL $(CFLAGS)

lenetCUDA:
	echo "not yet"


clean:
	rm -rf *.o lenetOCL lenetCUDA
