CC=gcc
OMP=-fopenmp

intel:CC=icc -mmic -DMIC_NATIVE 
intel:OMP=-O2 -qopenmp

SOURCES_CHECK=main_check.c kernel/kernel_cpu.c  ../../../include/log_helper.c

HARDENING_NUMBER=3
SOURCES_CHECK_HARDENED=main_check.c kernel/kernel_cpu_hardened.c  ../../../include/log_helper.c ../../selective_hardening/hardening.c ../../selective_hardening/xor_asm.c

SOURCES_ASM=kernel/kernel_cpu_hardened.c

SOURCES_GEN=main.c kernel/kernel_cpu.c 
APP_NAME=lavamd

general: lava_generate lavamd lavamd_hardened lavamd_err_inj lavamd_timing lavamd_timing_hardened

intel:  lava_generate lavamd lavamd_hardened lavamd_err_inj lavamd_timing lavamd_timing_hardened lavamd_asm


lava_generate: $(SOURCES_GEN)
	$(CC) $(SOURCES_GEN) -lm $(OMP) -o $(APP_NAME)_gen

lavamd: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -lm $(OMP) -DLOGS -o $(APP_NAME)_check

lavamd_hardened: $(SOURCES_CHECK_HARDENED)
	$(CC) $(SOURCES_CHECK_HARDENED) -lm $(OMP) -DHARDENING -DLOGS -o $(APP_NAME)_check_hardened_$(HARDENING_NUMBER)

lavamd_timing: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -lm $(OMP) -DTIMING -o $(APP_NAME)_timing

lavamd_timing_hardened: $(SOURCES_CHECK_HARDENED)
	$(CC) $(SOURCES_CHECK_HARDENED) -lm $(OMP) -DHARDENING -DTIMING -o $(APP_NAME)_timing_hardened_$(HARDENING_NUMBER)

lavamd_err_inj: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -lm $(OMP) -DLOGS -DERR_INJ -o $(APP_NAME)_err_inj

lavamd_asm: $(SOURCES_ASM)
	$(CC) -S $(SOURCES_ASM) $(OMP) -o $(APP_NAME)_asm_pure_$(HARDENING_NUMBER).s -masm=intel
	$(CC) -S $(SOURCES_ASM) -DTAGS $(OMP) -o $(APP_NAME)_asm_tagged_$(HARDENING_NUMBER).s -masm=intel
	diff $(APP_NAME)_asm_pure_$(HARDENING_NUMBER).s $(APP_NAME)_asm_tagged_$(HARDENING_NUMBER).s | dd of=asm_diff_$(HARDENING_NUMBER) status=none

clean:
	rm -f input_* output_* $(APP_NAME)_gen $(APP_NAME)_check $(APP_NAME)_check_hardened_$(HARDENING_NUMBER) $(APP_NAME)_timing $(APP_NAME)_timing_hardened_$(HARDENING_NUMBER) $(APP_NAME)_err_inj $(APP_NAME)_asm_pure_$(HARDENING_NUMBER).s $(APP_NAME)_asm_tagged_$(HARDENING_NUMBER).s asm_diff_$(HARDENING_NUMBER)


