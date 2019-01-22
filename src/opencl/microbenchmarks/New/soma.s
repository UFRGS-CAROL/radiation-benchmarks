	.file	"soma.cpp"
# GNU C++ (Ubuntu 4.8.4-2ubuntu1~14.04.4) version 4.8.4 (x86_64-linux-gnu)
#	compiled by GNU C version 4.8.4, GMP version 5.1.3, MPFR version 3.1.2-p3, MPC version 1.0.1
# GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
# options passed:  -fpreprocessed soma.ii -mtune=generic -march=x86-64
# -fverbose-asm -fopenmp -fstack-protector -Wformat -Wformat-security
# options enabled:  -faggressive-loop-optimizations
# -fasynchronous-unwind-tables -fauto-inc-dec -fbranch-count-reg -fcommon
# -fdelete-null-pointer-checks -fdwarf2-cfi-asm -fearly-inlining
# -feliminate-unused-debug-types -fexceptions -ffunction-cse -fgcse-lm
# -fgnu-runtime -fgnu-unique -fident -finline-atomics -fira-hoist-pressure
# -fira-share-save-slots -fira-share-spill-slots -fivopts
# -fkeep-static-consts -fleading-underscore -fmath-errno
# -fmerge-debug-strings -fmove-loop-invariants -fpeephole
# -fprefetch-loop-arrays -freg-struct-return
# -fsched-critical-path-heuristic -fsched-dep-count-heuristic
# -fsched-group-heuristic -fsched-interblock -fsched-last-insn-heuristic
# -fsched-rank-heuristic -fsched-spec -fsched-spec-insn-heuristic
# -fsched-stalled-insns-dep -fshow-column -fsigned-zeros
# -fsplit-ivs-in-unroller -fstack-protector -fstrict-volatile-bitfields
# -fsync-libcalls -ftrapping-math -ftree-coalesce-vars -ftree-cselim
# -ftree-forwprop -ftree-loop-if-convert -ftree-loop-im -ftree-loop-ivcanon
# -ftree-loop-optimize -ftree-parallelize-loops= -ftree-phiprop -ftree-pta
# -ftree-reassoc -ftree-scev-cprop -ftree-slp-vectorize
# -ftree-vect-loop-version -funit-at-a-time -funwind-tables -fverbose-asm
# -fzero-initialized-in-bss -m128bit-long-double -m64 -m80387
# -maccumulate-outgoing-args -malign-stringops -mfancy-math-387
# -mfp-ret-in-387 -mfxsr -mglibc -mieee-fp -mlong-double-80 -mmmx -mno-sse4
# -mpush-args -mred-zone -msse -msse2 -mtls-direct-seg-refs

	.section	.rodata
.LC0:
	.string	"Deu ruim"
.LC1:
	.string	"%llu\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
	.cfi_startproc
	pushq	%rbp	#
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp	#,
	.cfi_def_cfa_register 6
	subq	$64, %rsp	#,
	movl	%edi, -52(%rbp)	# argc, argc
	movq	%rsi, -64(%rbp)	# argv, argv
	movq	$1000000000, -40(%rbp)	#, MAX
	movl	$0, -44(%rbp)	#, i
	movq	$0, -32(%rbp)	#, re
	movabsq	$5000000000, %rax	#, tmp70
	movq	%rax, -24(%rbp)	# tmp70, gold
	movq	-32(%rbp), %rax	# re, tmp61
	movq	%rax, -8(%rbp)	# tmp61, .omp_data_o.1.re
	movq	-40(%rbp), %rax	# MAX, tmp62
	movq	%rax, -16(%rbp)	# tmp62, .omp_data_o.1.MAX
	leaq	-16(%rbp), %rax	#, tmp63
	movl	$0, %edx	#,
	movq	%rax, %rsi	# tmp63,
	movl	$main._omp_fn.0, %edi	#,
	call	GOMP_parallel_start	#
	leaq	-16(%rbp), %rax	#, tmp64
	movq	%rax, %rdi	# tmp64,
	call	main._omp_fn.0	#
	call	GOMP_parallel_end	#
	movq	-8(%rbp), %rax	# .omp_data_o.1.re, tmp65
	movq	%rax, -32(%rbp)	# tmp65, re
	movq	-16(%rbp), %rax	# .omp_data_o.1.MAX, tmp66
	movq	%rax, -40(%rbp)	# tmp66, MAX
	movq	-32(%rbp), %rax	# re, tmp67
	cmpq	-24(%rbp), %rax	# gold, tmp67
	je	.L2	#,
	movl	$.LC0, %edi	#,
	movl	$0, %eax	#,
	call	printf	#
	jmp	.L3	#
.L2:
	movq	-32(%rbp), %rax	# re, tmp68
	movq	%rax, %rsi	# tmp68,
	movl	$.LC1, %edi	#,
	movl	$0, %eax	#,
	call	printf	#
.L3:
	movl	$0, %eax	#, D.4284
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	main, .-main
	.type	main._omp_fn.0, @function
main._omp_fn.0:
.LFB3:
	.cfi_startproc
	pushq	%rbp	#
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp	#,
	.cfi_def_cfa_register 6
	pushq	%r12	#
	pushq	%rbx	#
	subq	$32, %rsp	#,
	.cfi_offset 12, -24
	.cfi_offset 3, -32
	movq	%rdi, -40(%rbp)	# .omp_data_i, .omp_data_i
	movq	$0, -24(%rbp)	#, re
	movq	-40(%rbp), %rax	# .omp_data_i, tmp69
	movq	(%rax), %rax	# .omp_data_i_7(D)->MAX, D.4287
	movl	%eax, %ebx	# D.4287, D.4288
	call	omp_get_num_threads	#
	movl	%eax, %r12d	#, D.4288
	call	omp_get_thread_num	#
	movl	%eax, %esi	#, D.4288
	movl	%ebx, %eax	# D.4288, tmp70
	cltd
	idivl	%r12d	# D.4288
	movl	%eax, %ecx	# tmp70, q.2
	movl	%ebx, %eax	# D.4288, tmp73
	cltd
	idivl	%r12d	# D.4288
	movl	%edx, %eax	# tmp72, tt.3
	cmpl	%eax, %esi	# tt.3, D.4288
	jl	.L6	#,
.L9:
	imull	%ecx, %esi	# q.2, D.4288
	movl	%esi, %edx	# D.4288, D.4288
	addl	%edx, %eax	# D.4288, D.4288
	leal	(%rax,%rcx), %edx	#, D.4288
	cmpl	%edx, %eax	# D.4288, D.4288
	jge	.L7	#,
	movl	%eax, -28(%rbp)	# D.4288, i
.L8:
	addq	$1, -24(%rbp)	#, re
	addq	$1, -24(%rbp)	#, re
	addq	$1, -24(%rbp)	#, re
	addq	$1, -24(%rbp)	#, re
	addq	$1, -24(%rbp)	#, re
#APP
# 32 "soma.cpp" 1
	movl $10, %eax;movl $20, %ebx;addl %ebx, %eax;
# 0 "" 2
#NO_APP
	addl	$1, -28(%rbp)	#, i
	cmpl	%edx, -28(%rbp)	# D.4288, i
	jl	.L8	#,
.L7:
	movq	-40(%rbp), %rax	# .omp_data_i, tmp74
	leaq	8(%rax), %rdx	#, D.4289
	movq	-24(%rbp), %rax	# re, tmp75
	lock addq	%rax, (%rdx)	#, tmp75,* D.4289
	jmp	.L10	#
.L6:
	movl	$0, %eax	#, tt.3
	addl	$1, %ecx	#, q.2
	jmp	.L9	#
.L10:
	addq	$32, %rsp	#,
	popq	%rbx	#
	popq	%r12	#
	popq	%rbp	#
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main._omp_fn.0, .-main._omp_fn.0
	.ident	"GCC: (Ubuntu 4.8.4-2ubuntu1~14.04.4) 4.8.4"
	.section	.note.GNU-stack,"",@progbits
