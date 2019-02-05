LOGS?=0
ARCH?=arm
CC=arm-linux-gnueabihf-gcc
EXEC=rijndael
MATRIXSIZE?=500

CCFLAGS= -O3 -Wno-unused-result
LDFLAGS= -static -lm -fopenmp


ifeq ($(LOGS), 1)
INCLUDE= -I../../include
LIBRARY= -L../../include -lLogHelper
CCFLAGS+= -DLOGS
IPADDR=127.0.0.1
PORT=999
endif

ifeq ($(LOGS), 0)
IPADDR=127.0.0.1
PORT=999
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

SOURCES= 

RADDATA=/home/fernando/git_pesquisa/radiation-benchmarks/data/aes
DEC=$(RADDATA)/input_large.asc
ENC=$(RADDATA)/input_large.enc
OUTPUT=$(RADDATA)/tmp.file

all: clean $(EXEC)

$(EXEC): aes.c aes.h aestab.h aesxam.c aes.c aes.h aestab.h aesxam.c
	$(CC) $(CCFLAGS) -o $(EXEC) aes.c aesxam.c   $(LDFLAGS) $(INCLUDE) $(LIBRARY)

clean:
	rm -f $(EXEC) $(DEFAULTGOLD) $(DEFAULTINPUT) rijndael_enc

#  IF LOGS=1 the ip and port could be a random value
#	char *ip_addr = argv[1];
#	unsigned int port = atoi(argv[2]);
#	char *input_file_path = argv[3];
#	char *output_file_path = argv[4];
#	char *golden_file_path = argv[5];
#	char mode = toupper(argv[6][0]);
#   char chave


test_enc: $(EXEC)
	./$(EXEC) $(IPADDR) $(PORT) $(DEC) $(OUTPUT) $(ENC) E 1234567890abcdeffedcba09876543211234567890abcdeffedcba0987654321

test_dec: $(EXEC)
	./$(EXEC) $(IPADDR) $(PORT) $(ENC) $(OUTPUT) $(DEC) D 1234567890abcdeffedcba09876543211234567890abcdeffedcba0987654321
	
	
	
	