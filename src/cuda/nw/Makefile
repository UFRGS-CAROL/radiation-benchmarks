NVCC=nvcc
TARGET=nw
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

all: clean $(TARGET)

$(TARGET):
	$(NVCC) needle.cu -o $(TARGET) -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda $(LOGS)


clean:
	rm -f *.o $(TARGET) *~
