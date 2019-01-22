
INC=-I/home/carol/radiation-benchmarks/src/include/
LIB=-L/usr/lib/  -lm -fopenmp -L../../../include/ -lLogHelper

SRC=new_verify_mem.cpp #/home/carol/radiation-benchmarks/src/include/log_helper.cpp
EXE=MicroBench_ADD

all: 
	g++ -o $(EXE) soma.cpp -DLOGS $(INC) $(LIB)
clean: 
	rm -f $(EXE) *.o *.d
