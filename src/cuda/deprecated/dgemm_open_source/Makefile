NVCC=nvcc
TARGET=dgemm
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

all: clean $(TARGET)
INCLUDE= -I../../include/
GENCODE=-gencode=arch=compute_35,code=sm_35

ifeq ($(dbg), 1)
	DEBUG_FLAGS=-g -G
endif

ifeq ($(logs), 1)
	LOG_FLAGS=-I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda -DLOGS
endif 

$(TARGET):
	$(NVCC) $(DEBUG_FLAGS) $(GENCODE) $(INCLUDE) dgemm.cu -o $(TARGET) $(LOG_FLAGS) -std=c++11 

generate:
	$(NVCC)  $(DEBUG_FLAGS) dgemm_generate.cu -o $(TARGET)_generate -std=c++11 


test:
	./dgemm -size=16 -input_a=../gemm/dgemm_a_8192 -input_b=../gemm/dgemm_b_8192 -gold=dgemm_gold_16 -iterations=2 -verbose

clean:
	rm -f *.o $(TARGET) *~ $(TARGET)_generate
