# C compiler
CC = gcc
DEFAULT_INPUT=8192

EXEC=gemm

NAME_SUFFIX=_not_hardened

#default is float
CC_FLAGS = -std=c99 -O0 -DDEFAULT_INPUT_SIZE=$(DEFAULT_INPUT) -DPRECISION=32 -fopenmp -DBOFFSET=12


INCLUDE+= -I../../../../../include
LIBRARY+= -L../../../../../include -lLogHelper -DLOGS

all: $(EXEC)_gen $(EXEC)_check $(EXEC)_err_inj $(EXEC)_timing 

$(EXEC)_gen: gemm_gen.c 
	$(CC) $(CC_FLAGS)  gemm_gen.c -o $(EXEC)_gen $(INCLUDE) $(LIBRARY) 

$(EXEC)_check: gemm.c 
	$(CC) $(CC_FLAGS) gemm.c -o $(EXEC)_check$(NAME_SUFFIX) $(INCLUDE) $(LIBRARY)

$(EXEC)_err_inj: gemm.c  
	$(CC) $(CC_FLAGS) gemm.c -DERR_INJ -o $(EXEC)_err_inj$(NAME_SUFFIX) $(INCLUDE) $(LIBRARY)

$(EXEC)_timing: gemm.c
	$(CC) $(CC_FLAGS) gemm.c -DTIMING -o $(EXEC)_timing$(NAME_SUFFIX) $(INCLUDE) $(LIBRARY)

generate:
	./$(EXEC)_gen 4 1024 8 /tmp/testa /tmp/testb /tmp/gold

test:
	./$(EXEC)_check 4 1024 8 /tmp/testa /tmp/testb /tmp/gold 10

clean:
	rm -f gemm_gen gemm_check$(NAME_SUFFIX) gemm_err_inj$(NAME_SUFFIX) gemm_timing$(NAME_SUFFIX)
