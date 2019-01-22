arm64:CFLAGS = -O3
arm64:LDFLAGS = -static -lm
arm64:CC=aarch64-linux-gnu-gcc
arm64:AS=aarch64-linux-gnu-as
arm64:LD=aarch64-linux-gnu-ld
arm64:AR=aarch64-linux-gnu-ar

arm:CFLAGS = -O3 -mcpu=cortex-a9 -mfpu=vfpv3 -mfloat-abi=hard
arm:LDFLAGS = -static -lm
arm:CC=arm-linux-gnueabihf-gcc
arm:AS=arm-linux-gnueabihf-as
arm:LD=arm-linux-gnueabihf-ld
arm:AR=arm-linux-gnueabihf-ar

arm64: rijndael_dec rijndael_enc
arm: rijndael_dec rijndael_enc

rijndael_dec: aes.c aes.h aestab.h aesxam.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o rijndael_dec aes.c aesxam.c
rijndael_enc: aes.c aes.h aestab.h aesxam.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o rijndael_enc aes.c aesxam.c
clean:
	rm -rf rijndael_enc rijndael_dec
