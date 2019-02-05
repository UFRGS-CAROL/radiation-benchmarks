LOGS?=0
ARCH?=arm
CC=arm-linux-gnueabihf-gcc
EXEC=fft
PORT=999
ADDR=127.0.0.1

OBJ = main.o fftmisc.o  fourier.o
FILE = main.c fftmisc.c  fourier.c

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


OBJDIR=./
OBJ= fftmisc.o  fourier.o  main.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard *.h) Makefile 

DEFAULTINPUT=./input.txt
DEFAULTGOLD=./gold.txt

all: clean $(EXEC)

$(EXEC): $(OBJS)  
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(INCLUDE) $(LIBRARY)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(CCFLAGS) -c $< -o $@ $(LDFLAGS) $(INCLUDE) 

clean:
	rm -f $(EXEC) *.o


#	unsigned int port = atoi(argv[2]);
#	setup_socket(argv[1], port);
#	char *fin_path = argv[3];
#	char *f_golden_real_path = argv[4];
#	int generate = atoi(argv[5]);
#	MAXSIZE = atoi(argv[6]);
#	MAXWAVES = atoi(argv[7]);
generate: $(EXEC)
	./$(EXEC) $(ADDR) $(PORT) $(DEFAULTINPUT) $(DEFAULTGOLD) 262144 8 1


test: $(EXEC) 
	./$(EXEC) $(ADDR) $(PORT) $(DEFAULTINPUT) $(DEFAULTGOLD) 262144 8 0

