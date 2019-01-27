
INC=-I/home/gabriel/radiation-benchmarks/src/include/
LIB=-L/usr/lib/  -lm -fopenmp -L../../../../../include/ -lLogHelper

SRC=new_verify_mem.cpp 
EXE=MicroBench_ADD

all: 
	g++ -o $(EXE)_INT_CPU soma.cpp -DLOGS  $(INC) $(LIB)  -DINT
	g++ -o $(EXE)_FLOAT_CPU soma.cpp -DLOGS  $(INC) $(LIB)  -DFLOAT
assembly:
	g++ -save-temps -fverbose-asm -o assembly_int soma.cpp -DLOGS -DINT $(INC) $(LIB)
	g++ -save-temps -fverbose-asm -o assembly_float soma.cpp -DLOGS -DFLOAT $(INC) $(LIB)
clean: 
	rm -f $(EXE)_INT_CPU $(EXE)_FLOAT_CPU  *.o *.d *.ii *.s teste
