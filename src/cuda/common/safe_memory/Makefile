# make static lib

CXX=gcc

CUDALIB= -L/usr/local/cuda/lib64 -lcudart
CUDAINC= -I/usr/local/cuda/include

CXXFLAGS=-fPIC -Wall -Wextra -shared  $(CUDALIB) $(CUDAINC) 
CXXFLAGS+= -Wno-unknown-pragmas -std=c++11 -O3 

OBJ=safe_memory.o
LIBOUT=libSafeMemory.so

all: clean safe_memory.cpp
	$(CXX) -c $(CXXFLAGS) safe_memory.cpp -o $(OBJ)
	ar cvr $(LIBOUT) $(OBJ)


clean:
	rm -rf $(LIBOUT) *.o *.a 
