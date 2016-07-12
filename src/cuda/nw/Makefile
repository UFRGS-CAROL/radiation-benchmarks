NVCC=nvcc
TARGET=nw
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

all: clean $(TARGET) generate

ifeq ($(dbg), 1)
	DEBUG_FLAG= -g -G
endif

LOG_FLAGS=
ifeq ($(logs), 1)
	LOG_FLAGS += -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda -DLOGS
endif 

$(TARGET):
	$(NVCC) $(DEBUG_FLAGS) $(LOG_FLAGS) needle.cu -o $(TARGET)  -std=c++11 

generate:
	$(NVCC)  $(DEBUG_FLAGS) needle_generate.cu -o $(TARGET)_generate -std=c++11 

gen_gold:nw_generate
	./nw_generate 2048 10

test:
	./nw 2048 10 input_2048 gold_2048 1

clean:
	rm -f *.o $(TARGET) *~ $(TARGET)_generate
