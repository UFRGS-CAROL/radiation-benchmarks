CC=gcc
OMP=-fopenmp

intel:CC=icc -mmic -DMIC_NATIVE
intel:OMP=-qopenmp

SOURCES_CHECK=main_check.c kernel/kernel_cpu.c  ../../include/log_helper.c
SOURCES_GEN=main.c kernel/kernel_cpu.c
APP_NAME=lavamd

armv8:CC=aarch64-linux-gnu-gcc
armv8:CC_FLAGS=-O3 -static -fopenmp -lm

armv8: lava_generate lavamd lavamd_err_inj lavamd_timing

general: lava_generate lavamd lavamd_err_inj lavamd_timing

intel:  lava_generate lavamd lavamd_err_inj lavamd_timing


lava_generate: $(SOURCES_GEN)
	$(CC) $(SOURCES_GEN) -lm $(OMP) -o $(APP_NAME)_gen

lavamd: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -lm $(OMP) -DLOGS -o $(APP_NAME)_check

lavamd_timing: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -lm $(OMP) -DTIMING -o $(APP_NAME)_timing

lavamd_err_inj: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -lm $(OMP) -DLOGS -DERR_INJ -o $(APP_NAME)_err_inj

clean:
	rm -f input_* output_* $(APP_NAME)_gen $(APP_NAME)_check $(APP_NAME)_timing $(APP_NAME)_err_inj
