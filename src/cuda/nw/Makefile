NVCC=nvcc
TARGET=nw

all: clean $(TARGET)

$(TARGET):
	$(NVCC) needle.cu -o $(TARGET)


clean:
	rm *.o $(TARGET) *~
