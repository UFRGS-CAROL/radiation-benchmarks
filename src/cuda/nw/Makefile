NVCC=nvcc
TARGET=nw
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

all: clean $(TARGET)

ifeq ($(dbg), 1)
	DEBUG_FLAG= -g -G
endif

ifeq ($(logs), 1)
	LOG_FLAGS=-I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda -DLOGS
endif 

$(TARGET):
	$(NVCC) $(DEBUG_FLAGS) needle.cu -o $(TARGET) $(LOG_FLAGS) -std=c++11 

generate:
	$(NVCC)  $(DEBUG_FLAGS) needle_generate.cu -o $(TARGET)_generate -std=c++11 

clean:
	rm -f *.o $(TARGET) *~ $(TARGET)_generate
