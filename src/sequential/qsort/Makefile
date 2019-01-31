LOGS?=0
ARCH?=arm
CC=arm-linux-gnueabihf-gcc
EXEC=qsort
MATRIXSIZE?=500

CCFLAGS= -O3 -Wno-unused-result
LDFLAGS= -static -lm -fopenmp


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

DEFAULTGOLD=./quicksort_gold_$(MATRIXSIZE).txt
DEFAULTINPUT=./quicksort_input_$(MATRIXSIZE).txt

all: clean $(EXEC)

$(EXEC): qsort.c
	$(CC) $(CCFLAGS) -o $(EXEC) qsort.c $(LDFLAGS) $(INCLUDE) $(LIBRARY)

clean:
	rm -f $(EXEC) $(DEFAULTGOLD) $(DEFAULTINPUT)

generate: $(EXEC)
	./$(EXEC) $(MATRIXSIZE) 1 $(DEFAULTGOLD) $(DEFAULTINPUT) 

test: $(EXEC) generate
	./$(EXEC) $(MATRIXSIZE) 0 $(DEFAULTGOLD) $(DEFAULTINPUT) 
