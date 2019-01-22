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

arm64:
	$(CC)  ${CFLAGS} -o matmul_500 matmul.c -lm
arm:
	$(CC)  ${CFLAGS} -o matmul_500 matmul.c -lm
