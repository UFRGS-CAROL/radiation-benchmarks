
INC=-I/home/gabriel/radiation-benchmarks/src/include/
LIB=-L/usr/lib/  -lm -fopenmp -L../../include/ -lLogHelper

SRC=new_verify_mem.cpp #/home/carol/radiation-benchmarks/src/include/log_helper.cpp
EXE=mem_test

all: 
	g++ -o $(EXE) new_verify_mem.cpp -DLOGS $(INC) $(LIB)
clean: 
	rm -f $(EXE) *.o *.d
