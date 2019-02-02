
INC=-I/home/gabriel/radiation-benchmarks/src/include/
LIB=-L/usr/lib/  -lm -fopenmp -L../../../../../include/ -lLogHelper

SRC=new_verify_mem.cpp 
EXE=MicroBench_MUL

all: 
	#g++ -o $(EXE)_INT_CPU mul.cpp -DLOGS  $(INC) $(LIB)  -DINT -fpermissive -g
	g++ -o $(EXE)_FLOAT_CPU mul.cpp -DLOGS  $(INC) $(LIB)  -DFLOAT
assembly:
	#g++ -save-temps -fverbose-asm -o assembly_int mul.cpp -DLOGS -DINT $(INC) $(LIB)
	g++ -save-temps -fverbose-asm -o assembly_float mul.cpp -DLOGS -DFLOAT $(INC) $(LIB)
clean: 
	rm -f $(EXE)_INT_CPU $(EXE)_FLOAT_CPU  *.o *.d *.ii *.s teste assembly_float assembly_int
