CC=gcc
OMP=-fopenmp

intel:CC=icc -mmic -DMIC_NATIVE 
intel:OMP=-qopenmp

SOURCES_CHECK=main_check.c kernel/kernel_cpu.c  ../../../include/log_helper.c
SOURCES_CHECK_HARDENED=main_check.c kernel/kernel_cpu_hardened.c  ../../../include/log_helper.c
SOURCES_GEN=main.c kernel/kernel_cpu.c 
APP_NAME=lavamd

general: lava_generate lavamd lavamd_hardened lavamd_err_inj lavamd_timing lavamd_timing_hardened

intel:  lava_generate lavamd lavamd_hardened lavamd_err_inj lavamd_timing lavamd_timing_hardened


lava_generate: $(SOURCES_GEN)
	$(CC) $(SOURCES_GEN) -lm $(OMP) -o $(APP_NAME)_gen

lavamd: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -lm $(OMP) -DLOGS -o $(APP_NAME)_check

lavamd_hardened: $(SOURCES_CHECK_HARDENED)
	$(CC) $(SOURCES_CHECK_HARDENED) -lm $(OMP) -DHARDENING -DLOGS -o $(APP_NAME)_check_hardened_2

lavamd_timing: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -lm $(OMP) -DTIMING -o $(APP_NAME)_timing

lavamd_timing_hardened: $(SOURCES_CHECK_HARDENED)
	$(CC) $(SOURCES_CHECK_HARDENED) -lm $(OMP) -DHARDENING -DTIMING -o $(APP_NAME)_timing_hardened_2

lavamd_err_inj: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -lm $(OMP) -DLOGS -DERR_INJ -o $(APP_NAME)_err_inj

clean:
	rm -f input_* output_* $(APP_NAME)_gen $(APP_NAME)_check $(APP_NAME)_check_hardened_2 $(APP_NAME)_timing $(APP_NAME)_timing_hardened_2 $(APP_NAME)_err_inj


