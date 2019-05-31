# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3 -lm

intel:CC=icc -mmic -DMIC_NATIVE
intel:CC_FLAGS=-O2 -qopenmp

SOURCES_CHECK=hotspot_check.c ../../../include/log_helper.c

HARDENING_NUMBER=3
SOURCES_CHECK_HARDENED=hotspot_check_hardened.c ../../../include/log_helper.c ../../selective_hardening/hardening.c ../../selective_hardening/xor_asm.c

SOURCES_ASM=hotspot_check.c

general: hotspot hotspot_check hotspot_check_hardened hotspot_err_inj hotspot_timing hotspot_timing_hardened

intel: hotspot hotspot_check hotspot_check_hardened hotspot_err_inj hotspot_timing hotspot_timing_hardened hotspot_asm

hotspot: hotspot.c
	$(CC) $(CC_FLAGS) hotspot.c -o hotspot_gen 

hotspot_check: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -o hotspot_check

hotspot_check_hardened: $(SOURCES_CHECK_HARDENED)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK_HARDENED) -DLOGS -o hotspot_check_hardened_$(HARDENING_NUMBER)

hotspot_err_inj: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -DERR_INJ -o hotspot_err_inj

hotspot_timing: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DTIMING -o hotspot_timing

hotspot_timing_hardened: $(SOURCES_CHECK_HARDENED)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK_HARDENED) -DTIMING -o hotspot_timing_hardened_$(HARDENING_NUMBER)

hotspot_asm: $(SOURCES_ASM)
	$(CC) -S $(SOURCES_ASM) $(CC_FLAGS) -o hotspot_asm_pure_$(HARDENING_NUMBER).s -masm=intel
	$(CC) -S $(SOURCES_ASM) -DTAGS $(CC_FLAGS) -o hotspot_asm_tagged_$(HARDENING_NUMBER).s -masm=intel
	diff hotspot_asm_pure_$(HARDENING_NUMBER).s hotspot_asm_tagged_$(HARDENING_NUMBER).s | dd of=asm_diff_$(HARDENING_NUMBER) status=none

clean:
	rm -f hotspot_gen hotspot_check hotspot_err_inj hotspot_timing hotspot_check_hardened_$(HARDENING_NUMBER) hotspot_timing_hardened_$(HARDENING_NUMBER) hotspot_asm_pure_$(HARDENING_NUMBER).s hotspot_asm_tagged_$(HARDENING_NUMBER).s asm_diff_$(HARDENING_NUMBER)
