# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3 -DBOFFSET=12

intel:CC=icc -mmic -DMIC_NATIVE
intel:CC_FLAGS=-O2 -qopenmp -DBOFFSET=12

SOURCES_CHECK = ../../../include/log_helper.c dgemm.c dgemm_kernel.c

HARDENING_NUMBER = 4
SOURCES_CHECK_HARDENED = ../../../include/log_helper.c dgemm_hardened.c dgemm_kernel_hardened.c ../../selective_hardening/hardening.c ../../selective_hardening/xor_asm.c
SOURCES_GEN = ../../../include/log_helper.c dgemm_gen.c dgemm_kernel.c

SOURCES_ASM = dgemm_kernel.c

general: dgemm_gen dgemm_check dgemm_check_hardened dgemm_err_inj dgemm_timing dgemm_timing_hardened genMatrices

intel: dgemm_gen dgemm_check dgemm_check_hardened dgemm_err_inj dgemm_timing dgemm_timing_hardened genMatrices genMatricesXeon dgemm_asm

dgemm_gen: $(SOURCES_GEN)
	$(CC) $(SOURCES_GEN) -o dgemm_gen $(CC_FLAGS)

dgemm_check: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -DLOGS -o dgemm_check $(CC_FLAGS)

dgemm_check_hardened: $(SOURCES_CHECK_HARDENED)
	$(CC) $(SOURCES_CHECK_HARDENED) -DLOGS -o dgemm_check_hardened_$(HARDENING_NUMBER) $(CC_FLAGS)

dgemm_err_inj: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -DLOGS -DERR_INJ -o dgemm_err_inj $(CC_FLAGS)

dgemm_timing: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -DTIMING -o dgemm_timing $(CC_FLAGS)

dgemm_timing_hardened: $(SOURCES_CHECK_HARDENED)
	$(CC) $(SOURCES_CHECK_HARDENED) -DTIMING -o dgemm_timing_hardened_$(HARDENING_NUMBER) $(CC_FLAGS)

genMatrices: generateMatrices.cpp
	g++ generateMatrices.cpp -o genMatrices

genMatricesXeon: generateMatrices.cpp
	$(CC) generateMatrices.cpp -o genMatricesXeon

dgemm_asm: $(SOURCES_ASM)
	$(CC) -S $(SOURCES_ASM) $(CC_FLAGS) -o dgemm_asm_pure_$(HARDENING_NUMBER).s -masm=intel
	$(CC) -S $(SOURCES_ASM) -DTAGS $(CC_FLAGS) -o dgemm_asm_tagged_$(HARDENING_NUMBER).s -masm=intel
	diff dgemm_asm_pure_$(HARDENING_NUMBER).s dgemm_asm_tagged_$(HARDENING_NUMBER).s | dd of=asm_diff_$(HARDENING_NUMBER) status=none

clean:
	rm -f dgemm_gen dgemm_check dgemm_err_inj dgemm_timing genMatrices genMatricesXeon dgemm_check_hardened_$(HARDENING_NUMBER) dgemm_timing_hardened_$(HARDENING_NUMBER) dgemm_asm_pure_$(HARDENING_NUMBER).s dgemm_asm_tagged_$(HARDENING_NUMBER).s asm_diff_$(HARDENING_NUMBER)
