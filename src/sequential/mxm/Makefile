LOGS?=0
ARCH?=arm
CC=arm-linux-gnueabihf-gcc
EXEC=matmul
MATRIXSIZE?=500

CCFLAGS= -O3 -Wno-unused-result 
LDFLAGS= -static -lm


ifeq ($(LOGS), 1)
INCLUDE= -I../../include
LIBRARY= -L../../include -lLogHelper
CCFLAGS+= -DLOGS
endif

ifeq ($(ARCH), arm64)
CC=aarch64-linux-gnu-gcc
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
CC=gcc
endif

DEFAULTGOLD=./matmul_gold.txt
DEFAULTINPUT=./matmul_input.dat

all: clean $(EXEC)

$(EXEC): matmul.c
	$(CC) $(CCFLAGS) -o $(EXEC) matmul.c $(LDFLAGS) $(INCLUDE) $(LIBRARY)

clean:
	rm -f $(EXEC) $(DEFAULTGOLD) $(DEFAULTINPUT)
	
generate: $(EXEC)
	./input_matmul.py $(DEFAULTINPUT) $(MATRIXSIZE)
	./$(EXEC) $(DEFAULTINPUT) $(DEFAULTGOLD) 1 $(MATRIXSIZE)

test: $(EXEC) generate 
	./$(EXEC) $(DEFAULTINPUT) $(DEFAULTGOLD) 0 $(MATRIXSIZE)
