#include ../../common/make.config
#C compiler
CC = g++

#OPENCL_DIR = /home/sc5nf/ati-stream-sdk-v2.2-lnx64/
#INC = $(OPENCL_DIR)/include/ 
#LIB = $(OPENCL_DIR)/lib/x86_64/
OPENCL_INC=/usr/local/cuda/include/
OPENCL_LIB=/usr/local/cuda/lib64/


FLAGS = -O3
SRC = euler3d.cpp

EXE = euler3d

release:$(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL $(FLAGS) 
	$(CC) $(KERNEL_DIM) euler3d_check.cpp -o euler3d_check -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL $(FLAGS) 

errmsg:$(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -D ERRMSG $(FLAGS)

ptx:$(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB)  -lOpenCL -D PTX_MSG $(FLAGS)

res:$(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB)  -lOpenCL -D RES_MSG $(FLAGS)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
