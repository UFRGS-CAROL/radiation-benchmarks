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

RADIATIONPATH=/home/carol/radiation-benchmarks
DEFAULTGOLD=$(RADIATIONPATH)/data/mxm/matmul_gold_$(MATRIXSIZE).bin
DEFAULTINPUTA=$(RADIATIONPATH)/data/mxm/matmul_inputa_$(MATRIXSIZE).bin
DEFAULTINPUTB=$(RADIATIONPATH)/data/mxm/matmul_inputb_$(MATRIXSIZE).bin
BINARYDEFAULT=$(RADIATIONPATH)/bin/$(EXEC)

DEFAULTIP=127.0.0.1
DEFAULTPORT=9999

all: clean $(EXEC)

$(EXEC): matmul.c
	$(CC) $(CCFLAGS) -o $(EXEC) matmul.c $(LDFLAGS) $(INCLUDE) $(LIBRARY)

clean:
	rm -f $(EXEC) $(DEFAULTGOLD) $(DEFAULTINPUT)



generate: $(EXEC)
	$(BINARYDEFAULT) $(DEFAULTIP) $(DEFAULTPORT) $(DEFAULTINPUTA)  $(DEFAULTINPUTB) $(DEFAULTGOLD) 1 $(MATRIXSIZE)

test: $(EXEC) generate
	$(BINARYDEFAULT) $(DEFAULTIP) $(DEFAULTPORT) $(DEFAULTINPUTA)  $(DEFAULTINPUTB) $(DEFAULTGOLD) 0 $(MATRIXSIZE)

mkdir:
	mkdir -p $(RADIATIONPATH)/data/mxm
