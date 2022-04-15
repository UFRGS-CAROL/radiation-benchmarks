
ARCH?=arm
CC=arm-linux-gnueabihf-g++
EXEC=matmul
MATRIXSIZE?=500
ITER=100
CCFLAGS= -O3 -Wno-unused-result
LDFLAGS= -lm -lcblas



INCLUDE= -I/usr/local/include
LIBRARY= -L/usr/local/lib/aarch64-linux-gnu  -L/usr/local/lib -lLogHelper
CCFLAGS+= -DLOGS

ifeq ($(ARCH), arm64)
CC=aarch64-linux-gnu-g++
AS=aarch64-linux-gnu-as
LD=aarch64-linux-gnu-ld
AR=aarch64-linux-gnu-ar
endif

ifeq ($(ARCH), arm)
AS=arm-linux-gnueabihf-as
LD=arm-linux-gnueabihf-ld
AR=arm-linux-gnueabihf-ar
CCFLAGS+= -mcpu=cortex-a9 -mfpu=vfpv3 -mfloat-abi=hard
endif

ifeq ($(ARCH), x86)
CC=g++
endif

DEFAULTGOLD=./matmul_gold_$(MATRIXSIZE).bin
DEFAULTINPUTA=./matmul_input_$(MATRIXSIZE).bin
BINARYDEFAULT=./$(EXEC)

all: clean $(EXEC)

$(EXEC): matmul.cpp
	$(CC) $(CCFLAGS) -o $(EXEC) matmul.cpp $(LDFLAGS) $(INCLUDE) $(LIBRARY)

clean:
	rm -f $(EXEC) $(DEFAULTGOLD) $(DEFAULTINPUT)



generate: $(EXEC)
	$(BINARYDEFAULT) -i $(DEFAULTINPUTA) -g $(DEFAULTGOLD) -s $(MATRIXSIZE) -c

test: $(EXEC) generate
	$(BINARYDEFAULT) -i $(DEFAULTINPUTA) -g $(DEFAULTGOLD) -s $(MATRIXSIZE) -l $(ITER)

mkdir:
	mkdir -p $(RADIATIONPATH)/data/mxm
