# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3 -lm

intel:CC=icc -mmic -DMIC_NATIVE
intel:CC_FLAGS=-O2 -qopenmp

SOURCES_CHECK=lud_check.c lud_omp.c ../../../include/log_helper.c

HARDENING_NUMBER=2
SOURCES_CHECK_HARDENED=lud_check.c lud_omp_hardened.c ../../../include/log_helper.c ../../selective_hardening/hardening.c ../../selective_hardening/xor_asm.c

SOURCES_ASM=lud_omp.c

general: lud_gen lud_check lud_check_hardened lud_err_inj lud_timing lud_timing_hardened

intel: lud_gen lud_check lud_check_hardened lud_err_inj lud_timing lud_timing_hardened lud_asm

lud_gen: lud_gen.c common.c
	$(CC) common.c lud_omp.c lud_gen.c -o lud_gen  $(CC_FLAGS)

lud_check: $(SOURCES_CHECK)
	$(CC) common.c $(SOURCES_CHECK) -DLOGS -o lud_check $(CC_FLAGS)

lud_check_hardened: $(SOURCES_CHECK_HARDENED)
	$(CC) common.c $(SOURCES_CHECK_HARDENED) -DLOGS -DFLOAT_XOR -o lud_check_hardened_$(HARDENING_NUMBER) $(CC_FLAGS)

lud_err_inj: $(SOURCES_CHECK)
	$(CC) common.c $(SOURCES_CHECK) -DLOGS -DERR_INJ -o lud_err_inj $(CC_FLAGS)

lud_timing: $(SOURCES_CHECK)
	$(CC) common.c $(SOURCES_CHECK) -DTIMING -o lud_timing $(CC_FLAGS)

lud_timing_hardened: $(SOURCES_CHECK_HARDENED)
	$(CC) common.c $(SOURCES_CHECK_HARDENED) -DTIMING -DFLOAT_XOR -o lud_timing_hardened_$(HARDENING_NUMBER) $(CC_FLAGS)

lud_asm: $(SOURCES_ASM)
	$(CC) -S $(SOURCES_ASM) $(CC_FLAGS) -o lud_asm_pure_$(HARDENING_NUMBER).s -masm=intel
	$(CC) -S $(SOURCES_ASM) -DTAGS $(CC_FLAGS) -o lud_asm_tagged_$(HARDENING_NUMBER).s -masm=intel
	diff lud_asm_pure_$(HARDENING_NUMBER).s lud_asm_tagged_$(HARDENING_NUMBER).s | dd of=asm_diff_$(HARDENING_NUMBER) status=none

clean:
	rm -f lud_gen lud_check lud_err_inj lud_timing lud_check_hardened_$(HARDENING_NUMBER) lud_timing_hardened_$(HARDENING_NUMBER) lud_asm_pure_$(HARDENING_NUMBER).s lud_asm_tagged_$(HARDENING_NUMBER).s asm_diff_$(HARDENING_NUMBER)
