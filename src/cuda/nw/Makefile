NVCC=nvcc
TARGET=nw
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

all: clean $(TARGET)

$(TARGET):
	$(NVCC) needle.cu -o $(TARGET) -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda $(LOGS)

generate: clean
	$(NVCC) needle_generate.cu -o $(TARGET)_generate -std=c++11 

clean:
	rm -f *.o $(TARGET) *~ $(TARGET)_generate
