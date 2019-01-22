OBJ = main.o fftmisc.o  fourier.o
FILE = main.c fftmisc.c  fourier.c

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

x86:CFLAGS = -O3
x86:LDFLAGS = -static -lm
x86:CC=gcc
x86: ${OBJ}
	$(CC)  ${CFLAGS} ${OBJ} -o fft -lm

arm: ${OBJ}
	$(CC)  ${CFLAGS} ${OBJ} -o fft -lm
arm64: ${OBJ}
	$(CC)  ${CFLAGS} ${OBJ} -o fft -lm
fftmisc.o: fftmisc.c
	$(CC) ${CFLAGS} -c fftmisc.c
fourierf.o: fourierf.c
	$(CC) ${CFLAGS} -c fourierf.c
main.o: main.c
	$(CC) ${CFLAGS} -c main.c

clean:
	rm -rf *.o fft output*
