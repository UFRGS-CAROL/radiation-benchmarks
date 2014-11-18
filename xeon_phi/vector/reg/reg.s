# mark_description "Intel(R) C Intel(R) 64 Compiler XE for applications running on Intel(R) 64, Version 14.0.3.174 Build 2014042";
# mark_description "2";
# mark_description "-openmp -S -";
	.file "reg.c"
	.text
..TXTST0:
# -- Begin  main, L_main_52__par_loop1_2.26, L_main_52__tree_reduce1_2.73, L_main_50__par_region0_2.74
# mark_begin;
       .align    16,0x90
	.globl main
main:
# parameter 1: %edi
# parameter 2: %rsi
..B1.1:                         # Preds ..B1.0
..___tag_value_main.1:                                          #20.34
        pushq     %rbp                                          #20.34
..___tag_value_main.3:                                          #
        movq      %rsp, %rbp                                    #20.34
..___tag_value_main.4:                                          #
        andq      $-128, %rsp                                   #20.34
        subq      $512, %rsp                                    #20.34
        movq      %rbx, 432(%rsp)                               #20.34
..___tag_value_main.6:                                          #
        movq      %rsi, %rbx                                    #20.34
        movq      %r12, 424(%rsp)                               #20.34
..___tag_value_main.7:                                          #
        movl      %edi, %r12d                                   #20.34
        movq      $0x000000000, %rsi                            #20.34
        movl      $3, %edi                                      #20.34
        movq      %r15, 400(%rsp)                               #20.34
        movq      %r14, 408(%rsp)                               #20.34
        movq      %r13, 416(%rsp)                               #20.34
        call      __intel_new_feature_proc_init                 #20.34
..___tag_value_main.8:                                          #
                                # LOE rbx r12d
..B1.74:                        # Preds ..B1.1
        stmxcsr   (%rsp)                                        #20.34
        movl      $.2.13_2_kmpc_loc_struct_pack.8, %edi         #20.34
        xorl      %esi, %esi                                    #20.34
        orl       $32832, (%rsp)                                #20.34
        xorl      %eax, %eax                                    #20.34
        ldmxcsr   (%rsp)                                        #20.34
..___tag_value_main.11:                                         #20.34
        call      __kmpc_begin                                  #20.34
..___tag_value_main.12:                                         #
                                # LOE rbx r12d
..B1.2:                         # Preds ..B1.74
        cmpl      $3, %r12d                                     #25.16
        jne       ..B1.31       # Prob 22%                      #25.16
                                # LOE rbx
..B1.3:                         # Preds ..B1.2
        movq      8(%rbx), %rax                                 #31.36
        xorl      %edx, %edx                                    #31.19
        lea       1(%rax), %rcx                                 #
        movb      (%rax), %al                                   #31.19
        xorb      $48, %al                                      #31.19
        cmpb      $9, %al                                       #31.19
        jg        ..B1.8        # Prob 10%                      #31.19
                                # LOE rdx rcx rbx al
..B1.5:                         # Preds ..B1.3 ..B1.6
        testb     %al, %al                                      #31.19
        jl        ..B1.8        # Prob 20%                      #31.19
                                # LOE rdx rcx rbx al
..B1.6:                         # Preds ..B1.5
        movsbq    %al, %rax                                     #31.19
        lea       (%rdx,%rdx,4), %rdx                           #31.19
        lea       (%rax,%rdx,2), %rdx                           #31.19
        movb      (%rcx), %al                                   #31.19
        xorb      $48, %al                                      #31.19
        incq      %rcx                                          #31.19
        cmpb      $9, %al                                       #31.19
        jle       ..B1.5        # Prob 82%                      #31.19
                                # LOE rdx rcx rbx al
..B1.8:                         # Preds ..B1.5 ..B1.6 ..B1.3
        movq      16(%rbx), %rdi                                #32.28
        movq      %rdx, 440(%rsp)                               #31.5
        call      atol                                          #32.28
                                # LOE rax
..B1.9:                         # Preds ..B1.8
        cmpl      $4, %eax                                      #32.16
        ja        ..B1.89       # Prob 28%                      #32.16
                                # LOE rax r12d
..B1.10:                        # Preds ..B1.9
        movl      %eax, %eax                                    #32.16
        jmp       *..1..TPKT.get_refword.2_0.0.2.11(,%rax,8)    #32.16
                                # LOE
..1.2_0.TAG.04.0.2.11:
..B1.12:                        # Preds ..B1.10
        movq      lfsr.19.0.5(%rip), %rdx                       #32.16
        movq      %rdx, %rcx                                    #32.16
        shrq      $10, %rcx                                     #32.16
        movq      %rdx, %rax                                    #32.16
        shrq      $11, %rax                                     #32.16
        xorq      %rdx, %rcx                                    #32.16
        movq      %rdx, %rsi                                    #32.16
        xorq      %rax, %rcx                                    #32.16
        shrq      $30, %rdx                                     #32.16
        xorl      %ebx, %ebx                                    #32.16
        xorq      %rdx, %rcx                                    #32.16
        notq      %rcx                                          #32.16
        andq      $1, %rcx                                      #32.16
        shrq      $1, %rsi                                      #32.16
        shlq      $31, %rcx                                     #32.16
        orq       %rcx, %rsi                                    #32.16
        movq      %rsi, lfsr.19.0.5(%rip)                       #32.16
        movl      %esi, %r12d                                   #32.16
        jmp       ..B1.21       # Prob 100%                     #32.16
                                # LOE ebx r12d
..1.2_0.TAG.03.0.2.11:
..B1.14:                        # Preds ..B1.10
        movl      $-1, %r12d                                    #32.16
        xorl      %ebx, %ebx                                    #32.16
        jmp       ..B1.21       # Prob 100%                     #32.16
                                # LOE ebx r12d
..1.2_0.TAG.02.0.2.11:
..B1.16:                        # Preds ..B1.10
        movl      $1431655765, %r12d                            #32.16
        xorl      %ebx, %ebx                                    #32.16
        jmp       ..B1.21       # Prob 100%                     #32.16
                                # LOE ebx r12d
..1.2_0.TAG.01.0.2.11:
..B1.18:                        # Preds ..B1.10
        movl      $1, %r12d                                     #32.16
        xorl      %ebx, %ebx                                    #32.16
        jmp       ..B1.21       # Prob 100%                     #32.16
                                # LOE ebx r12d
..1.2_0.TAG.00.0.2.11:
..B1.20:                        # Preds ..B1.10
        xorl      %r12d, %r12d                                  #32.16
        xorl      %ebx, %ebx                                    #32.16
        jmp       ..B1.21       # Prob 100%                     #32.16
                                # LOE ebx r12d
..B1.89:                        # Preds ..B1.9
        xorl      %ebx, %ebx                                    #32.16
                                # LOE ebx r12d
..B1.21:                        # Preds ..B1.20 ..B1.18 ..B1.16 ..B1.14 ..B1.12
                                #       ..B1.89
        movl      $.L_2__STRING.7, %edi                         #34.5
        xorl      %eax, %eax                                    #34.5
        movq      440(%rsp), %rsi                               #34.5
..___tag_value_main.13:                                         #34.5
        call      printf                                        #34.5
..___tag_value_main.14:                                         #
                                # LOE ebx r12d
..B1.22:                        # Preds ..B1.21
        movl      $.L_2__STRING.8, %edi                         #35.5
        movl      %r12d, %esi                                   #35.5
        xorl      %eax, %eax                                    #35.5
..___tag_value_main.15:                                         #35.5
        call      printf                                        #35.5
..___tag_value_main.16:                                         #
                                # LOE ebx r12d
..B1.23:                        # Preds ..B1.22
        movl      $4, %edi                                      #37.5
..___tag_value_main.17:                                         #37.5
        call      omp_set_num_threads                           #37.5
..___tag_value_main.18:                                         #
                                # LOE ebx r12d
..B1.24:                        # Preds ..B1.23
        movl      $.L_2__STRING.9, %edi                         #38.5
        movl      $4, %esi                                      #38.5
        xorl      %eax, %eax                                    #38.5
..___tag_value_main.19:                                         #38.5
        call      printf                                        #38.5
..___tag_value_main.20:                                         #
                                # LOE ebx r12d
..B1.25:                        # Preds ..B1.24
        movl      %ebx, 456(%rsp)                               #43.20
        movl      $.2.13_2_kmpc_loc_struct_pack.19, %edi        #50.5
        movq      $0, 448(%rsp)                                 #44.16
        movl      %ebx, 460(%rsp)                               #45.26
        movl      %r12d, %eax                                   #48.0
        movl      %eax, 464(%rsp)                               #48.0
        call      __kmpc_global_thread_num                      #50.5
                                # LOE eax
..B1.77:                        # Preds ..B1.25
        xorl      %edx, %edx                                    #50.5
        movl      $2, %edi                                      #50.5
        movl      %eax, 468(%rsp)                               #50.5
        movl      $-1, %esi                                     #50.5
        xorl      %ecx, %ecx                                    #50.5
        movl      $__sd_2inst_string.1, %r8d                    #50.5
        movl      $50, %r9d                                     #50.5
        xorl      %eax, %eax                                    #50.5
..___tag_value_main.21:                                         #50.5
        call      __offload_target_acquire                      #50.5
..___tag_value_main.22:                                         #
                                # LOE rax
..B1.76:                        # Preds ..B1.77
        movq      %rax, %r12                                    #50.5
                                # LOE r12
..B1.26:                        # Preds ..B1.76
        testq     %r12, %r12                                    #50.5
        je        ..B1.28       # Prob 50%                      #50.5
                                # LOE r12
..B1.27:                        # Preds ..B1.26
        movl      $320, %edx                                    #50.5
        lea       80(%rsp), %rbx                                #50.5
        movq      %rbx, %rdi                                    #50.5
        lea       .2.13_2__offload_var_desc1_p.92(%rip), %rsi   #50.5
        call      _intel_fast_memcpy                            #50.5
                                # LOE rbx r12
..B1.78:                        # Preds ..B1.27
        movl      $80, %r11d                                    #50.5
        lea       (%rsp), %r9                                   #50.5
                                # LOE rbx r9 r11 r12
..B1.73:                        # Preds ..B1.73 ..B1.78
        movq      -8+.2.13_2__offload_var_desc2_p.97(%r11), %rax #50.5
        movq      -16+.2.13_2__offload_var_desc2_p.97(%r11), %rdx #50.5
        movq      -24+.2.13_2__offload_var_desc2_p.97(%r11), %rcx #50.5
        movq      -32+.2.13_2__offload_var_desc2_p.97(%r11), %rsi #50.5
        movq      -40+.2.13_2__offload_var_desc2_p.97(%r11), %r10 #50.5
        movq      %rax, -8(%r9,%r11)                            #50.5
        movq      %rdx, -16(%r9,%r11)                           #50.5
        movq      %rcx, -24(%r9,%r11)                           #50.5
        movq      %rsi, -32(%r9,%r11)                           #50.5
        movq      %r10, -40(%r9,%r11)                           #50.5
        subq      $40, %r11                                     #50.5
        jne       ..B1.73       # Prob 50%                      #50.5
                                # LOE rbx r9 r11 r12
..B1.72:                        # Preds ..B1.73
        xorl      %r15d, %r15d                                  #50.5
        lea       448(%rsp), %r10                               #50.5
        movq      %r10, 200(%rsp)                               #50.5
        lea       464(%rsp), %r11                               #50.5
        movq      %r11, 264(%rsp)                               #50.5
        lea       440(%rsp), %r13                               #50.5
        movq      %r13, 328(%rsp)                               #50.5
        lea       460(%rsp), %r14                               #50.5
        movq      %r14, 392(%rsp)                               #50.5
        lea       456(%rsp), %rax                               #50.5
        addq      $-32, %rsp                                    #50.5
        movl      $__sd_2inst_string.2, %esi                    #50.5
        movq      %rax, 56(%rbx)                                #50.5
        movq      %r12, %rdi                                    #50.5
        xorl      %edx, %edx                                    #50.5
        movl      $5, %ecx                                      #50.5
        movq      %rbx, %r8                                     #50.5
        movq      %r15, (%rsp)                                  #50.5
        movq      %r15, 8(%rsp)                                 #50.5
        movq      %r15, 16(%rsp)                                #50.5
        xorl      %eax, %eax                                    #50.5
..___tag_value_main.23:                                         #50.5
        call      __offload_offload1                            #50.5
..___tag_value_main.24:                                         #
                                # LOE
..B1.79:                        # Preds ..B1.72
        addq      $32, %rsp                                     #50.5
        jmp       ..B1.29       # Prob 100%                     #50.5
                                # LOE
..B1.28:                        # Preds ..B1.26
        addq      $-16, %rsp                                    #50.5
        movl      $___kmpv_zeromain_0, %esi                     #50.5
        lea       484(%rsp), %rdi                               #50.5
        lea       472(%rsp), %rdx                               #50.5
        lea       464(%rsp), %rcx                               #50.5
        lea       480(%rsp), %r8                                #50.5
        lea       456(%rsp), %r9                                #50.5
        lea       476(%rsp), %rax                               #50.5
        movq      %rax, (%rsp)                                  #50.5
..___tag_value_main.25:                                         #50.5
        call      L_main_50__par_region0_2.74                   #50.5
..___tag_value_main.26:                                         #
                                # LOE
..B1.80:                        # Preds ..B1.28
        addq      $16, %rsp                                     #50.5
                                # LOE
..B1.29:                        # Preds ..B1.79 ..B1.80
        movl      $.L_2__STRING.11, %edi                        #90.5
        xorl      %eax, %eax                                    #90.5
        movl      460(%rsp), %esi                               #90.5
..___tag_value_main.27:                                         #90.5
        call      printf                                        #90.5
..___tag_value_main.28:                                         #
                                # LOE
..B1.30:                        # Preds ..B1.29
        xorl      %edi, %edi                                    #91.5
        call      exit                                          #91.5
                                # LOE
..B1.31:                        # Preds ..B1.2
        movl      $il0_peep_printf_format_0, %edi               #26.9
        movq      $0, 440(%rsp)                                 #22.26
        call      puts                                          #26.9
                                # LOE
..B1.32:                        # Preds ..B1.31
        movl      $il0_peep_printf_format_1, %edi               #27.9
        call      puts                                          #27.9
                                # LOE
..B1.33:                        # Preds ..B1.32
        movl      $.L_2__STRING.1, %edi                         #27.9
        xorl      %esi, %esi                                    #27.9
        xorl      %eax, %eax                                    #27.9
..___tag_value_main.29:                                         #27.9
        call      printf                                        #27.9
..___tag_value_main.30:                                         #
                                # LOE
..B1.34:                        # Preds ..B1.33
        movl      $.L_2__STRING.2, %edi                         #27.9
        movl      $1, %esi                                      #27.9
        xorl      %eax, %eax                                    #27.9
..___tag_value_main.31:                                         #27.9
        call      printf                                        #27.9
..___tag_value_main.32:                                         #
                                # LOE
..B1.35:                        # Preds ..B1.34
        movl      $.L_2__STRING.3, %edi                         #27.9
        movl      $1431655765, %esi                             #27.9
        xorl      %eax, %eax                                    #27.9
..___tag_value_main.33:                                         #27.9
        call      printf                                        #27.9
..___tag_value_main.34:                                         #
                                # LOE
..B1.36:                        # Preds ..B1.35
        movl      $.L_2__STRING.4, %edi                         #27.9
        movl      $-1, %esi                                     #27.9
        xorl      %eax, %eax                                    #27.9
..___tag_value_main.35:                                         #27.9
        call      printf                                        #27.9
..___tag_value_main.36:                                         #
                                # LOE
..B1.37:                        # Preds ..B1.36
        movl      $il0_peep_printf_format_2, %edi               #27.9
        call      puts                                          #27.9
                                # LOE
..B1.38:                        # Preds ..B1.37
        movl      $1, %edi                                      #28.9
        call      exit                                          #28.9
..___tag_value_main.37:                                         #
                                # LOE
L_main_52__tree_reduce1_2.73:
# parameter 1: %rdi
# parameter 2: %rsi
..B1.39:                        # Preds ..B1.0
        pushq     %rbp                                          #52.9
..___tag_value_main.44:                                         #
        movq      %rsp, %rbp                                    #52.9
..___tag_value_main.45:                                         #
        andq      $-128, %rsp                                   #52.9
        subq      $512, %rsp                                    #52.9
        movq      %rbx, 432(%rsp)                               #52.9
        movq      %r15, 400(%rsp)                               #52.9
        movq      %r14, 408(%rsp)                               #52.9
        movq      %r13, 416(%rsp)                               #52.9
        movq      %r12, 424(%rsp)                               #52.9
        movl      (%rsi), %eax                                  #52.9
        movq      400(%rsp), %r15                               #52.9
..___tag_value_main.47:                                         #
        movq      408(%rsp), %r14                               #52.9
..___tag_value_main.51:                                         #
        movq      416(%rsp), %r13                               #52.9
..___tag_value_main.52:                                         #
        movq      424(%rsp), %r12                               #52.9
..___tag_value_main.53:                                         #
        movq      432(%rsp), %rbx                               #52.9
..___tag_value_main.54:                                         #
        addl      %eax, (%rdi)                                  #52.9
        xorl      %eax, %eax                                    #52.9
        movq      %rbp, %rsp                                    #52.9
        popq      %rbp                                          #52.9
..___tag_value_main.55:                                         #
        ret                                                     #52.9
..___tag_value_main.57:                                         #
                                # LOE
L_main_52__par_loop1_2.26:
# parameter 1: %rdi
# parameter 2: %rsi
# parameter 3: %rdx
# parameter 4: %rcx
# parameter 5: %r8
..B1.40:                        # Preds ..B1.0
        pushq     %rbp                                          #52.9
..___tag_value_main.59:                                         #
        movq      %rsp, %rbp                                    #52.9
..___tag_value_main.60:                                         #
        andq      $-128, %rsp                                   #52.9
        subq      $512, %rsp                                    #52.9
        xorl      %r10d, %r10d                                  #52.9
        movl      $1, %eax                                      #52.9
        movq      %rbx, 432(%rsp)                               #52.9
        movq      %r15, 400(%rsp)                               #52.9
        movq      %r14, 408(%rsp)                               #52.9
        movq      %r13, 416(%rsp)                               #52.9
..___tag_value_main.62:                                         #
        movq      %r8, %r13                                     #52.9
        movq      %r12, 424(%rsp)                               #52.9
        movl      %r10d, 56(%rsp)                               #52.9
        movl      %r10d, (%rsp)                                 #52.9
        movl      $3, 4(%rsp)                                   #52.9
        movl      %r10d, 8(%rsp)                                #52.9
        movl      %eax, 12(%rsp)                                #52.9
        addq      $-32, %rsp                                    #52.9
..___tag_value_main.66:                                         #
        movl      (%rdi), %ebx                                  #52.9
        lea       44(%rsp), %r11                                #52.9
        movq      (%rcx), %r15                                  #52.9
        movl      $.2.13_2_kmpc_loc_struct_pack.28, %edi        #52.9
        movl      (%rdx), %r14d                                 #52.9
        movl      %ebx, %esi                                    #52.9
        movq      %r11, (%rsp)                                  #52.9
        movl      $34, %edx                                     #52.9
        movl      %eax, 8(%rsp)                                 #52.9
        lea       40(%rsp), %rcx                                #52.9
        movl      %eax, 16(%rsp)                                #52.9
        lea       32(%rsp), %r8                                 #52.9
        xorl      %eax, %eax                                    #52.9
        lea       36(%rsp), %r9                                 #52.9
..___tag_value_main.67:                                         #52.9
        call      __kmpc_for_static_init_4u                     #52.9
..___tag_value_main.68:                                         #
                                # LOE r13 r15 ebx r14d
..B1.81:                        # Preds ..B1.40
        addq      $32, %rsp                                     #52.9
                                # LOE r13 r15 ebx r14d
..B1.41:                        # Preds ..B1.81
        movl      (%rsp), %ecx                                  #52.9
        cmpl      $3, %ecx                                      #52.9
        movl      4(%rsp), %eax                                 #52.9
        ja        ..B1.55       # Prob 50%                      #52.9
                                # LOE r13 r15 eax ecx ebx r14d
..B1.42:                        # Preds ..B1.41
        movd      %r14d, %xmm0                                  #61.13
        movl      $3, %r12d                                     #52.9
        cmpl      $3, %eax                                      #52.9
        pshufd    $0, %xmm0, %xmm1                              #61.13
        movdqa    .L_2il0floatpacket.140(%rip), %xmm0           #70.54
        cmovb     %eax, %r12d                                   #52.9
        movdqa    %xmm1, 32(%rsp)                               #70.54
        movq      %r13, 48(%rsp)                                #70.54
        movl      %ecx, %r13d                                   #70.54
                                # LOE r15 ebx r12d r13d r14d
..B1.43:                        # Preds ..B1.86 ..B1.42
# Begin ASM
        nop                                                     #55.0
        nop                                                     #56.0
        nop                                                     #57.0
# End ASM
                                # LOE r15 ebx r12d r13d r14d
..B1.82:                        # Preds ..B1.43
        movdqa    32(%rsp), %xmm0                               #62.52
        cmpq      $1, %r15                                      #66.28
        movdqa    %xmm0, 272(%rsp)                              #62.52
        movdqa    %xmm0, 80(%rsp)                               #62.45
        movdqa    %xmm0, 144(%rsp)                              #62.38
        movdqa    %xmm0, 208(%rsp)                              #62.24
        movdqa    %xmm0, 288(%rsp)                              #62.52
        movdqa    %xmm0, 96(%rsp)                               #62.45
        movdqa    %xmm0, 160(%rsp)                              #62.38
        movdqa    %xmm0, 224(%rsp)                              #62.24
        movdqa    %xmm0, 304(%rsp)                              #62.52
        movdqa    %xmm0, 112(%rsp)                              #62.45
        movdqa    %xmm0, 176(%rsp)                              #62.38
        movdqa    %xmm0, 240(%rsp)                              #62.24
        movdqa    %xmm0, 320(%rsp)                              #62.52
        movdqa    %xmm0, 128(%rsp)                              #62.45
        movdqa    %xmm0, 192(%rsp)                              #62.38
        movdqa    %xmm0, 256(%rsp)                              #62.24
        jbe       ..B1.53       # Prob 10%                      #66.28
                                # LOE r15 ebx r12d r13d r14d
..B1.44:                        # Preds ..B1.82
        movl      %r12d, 16(%rsp)                               #
        movl      %ebx, 24(%rsp)                                #
                                # LOE r15 r13d r14d
..B1.45:                        # Preds ..B1.51 ..B1.44
# Begin ASM
        nop                                                     #67.0
        nop                                                     #68.0
# End ASM
                                # LOE r15 r13d r14d
..B1.85:                        # Preds ..B1.45
        movdqa    80(%rsp), %xmm0                               #70.54
        movdqa    96(%rsp), %xmm1                               #70.54
        movdqa    112(%rsp), %xmm2                              #70.54
        movdqa    128(%rsp), %xmm4                              #70.54
        pcmpeqd   144(%rsp), %xmm0                              #70.54
        pcmpeqd   160(%rsp), %xmm1                              #70.54
        pcmpeqd   176(%rsp), %xmm2                              #70.54
        pcmpeqd   192(%rsp), %xmm4                              #70.54
        movdqa    .L_2il0floatpacket.140(%rip), %xmm3           #70.54
        pand      %xmm3, %xmm0                                  #70.54
        pand      %xmm3, %xmm1                                  #70.54
        pand      %xmm3, %xmm2                                  #70.54
        pand      %xmm3, %xmm4                                  #70.54
        movdqa    %xmm0, 272(%rsp)                              #70.21
        movdqa    %xmm1, 288(%rsp)                              #70.21
        movdqa    %xmm2, 304(%rsp)                              #70.21
        movdqa    %xmm4, 320(%rsp)                              #70.21
                                # LOE r15 r13d r14d
..B1.84:                        # Preds ..B1.85
# Begin ASM
        nop                                                     #72.0
        nop                                                     #73.0
# End ASM
                                # LOE r15 r13d r14d
..B1.83:                        # Preds ..B1.84
        movl      56(%rsp), %r12d                               #77.25
        xorl      %ebx, %ebx                                    #75.21
                                # LOE rbx r15 r12d r13d r14d
..B1.46:                        # Preds ..B1.50 ..B1.83
        cmpl      272(%rsp,%rbx,4), %r14d                       #76.33
        jne       ..B1.48       # Prob 50%                      #76.33
                                # LOE rbx r15 r12d r13d r14d
..B1.47:                        # Preds ..B1.46
        cmpl      208(%rsp,%rbx,4), %r14d                       #76.48
        je        ..B1.50       # Prob 78%                      #76.48
                                # LOE rbx r15 r12d r13d r14d
..B1.48:                        # Preds ..B1.46 ..B1.47
        movl      $.L_2__STRING.10, %edi                        #78.25
        movq      %rbx, %rsi                                    #78.25
        xorl      %edx, %edx                                    #78.25
        movl      %r13d, %ecx                                   #78.25
        xorl      %eax, %eax                                    #78.25
        incl      %r12d                                         #77.25
        movl      272(%rsp,%rbx,4), %r8d                        #78.25
        movl      %r12d, 56(%rsp)                               #77.25
..___tag_value_main.69:                                         #78.25
        call      printf                                        #78.25
..___tag_value_main.70:                                         #
                                # LOE rbx r15 r12d r13d r14d
..B1.49:                        # Preds ..B1.48
        movl      %r14d, 272(%rsp,%rbx,4)                       #79.60
        movl      %r14d, 80(%rsp,%rbx,4)                        #79.53
        movl      %r14d, 144(%rsp,%rbx,4)                       #79.46
        movl      %r14d, 208(%rsp,%rbx,4)                       #79.32
                                # LOE rbx r15 r12d r13d r14d
..B1.50:                        # Preds ..B1.47 ..B1.49
        incq      %rbx                                          #75.40
        cmpq      $16, %rbx                                     #75.30
        jb        ..B1.46       # Prob 93%                      #75.30
                                # LOE rbx r15 r12d r13d r14d
..B1.51:                        # Preds ..B1.50
        .byte     15                                            #66.41
        .byte     31                                            #66.41
        .byte     132                                           #66.41
        .byte     0                                             #66.41
        .byte     0                                             #66.41
        .byte     0                                             #66.41
        .byte     0                                             #66.41
        .byte     0                                             #66.41
        incq      %rbx                                          #66.41
        cmpq      %rbx, %r15                                    #66.28
        ja        ..B1.45       # Prob 93%                      #66.28
                                # LOE r15 r13d r14d
..B1.52:                        # Preds ..B1.51
        movl      16(%rsp), %r12d                               #
        movl      24(%rsp), %ebx                                #
                                # LOE r15 ebx r12d r13d r14d
..B1.53:                        # Preds ..B1.52 ..B1.82
# Begin ASM
        nop                                                     #84.0
        nop                                                     #85.0
        nop                                                     #86.0
# End ASM
                                # LOE r15 ebx r12d r13d r14d
..B1.86:                        # Preds ..B1.53
        incl      %r13d                                         #53.49
        cmpl      %r12d, %r13d                                  #54.9
        jbe       ..B1.43       # Prob 82%                      #54.9
                                # LOE r15 ebx r12d r13d r14d
..B1.54:                        # Preds ..B1.86
        movq      48(%rsp), %r13                                #
                                # LOE r13 ebx
..B1.55:                        # Preds ..B1.54 ..B1.41
        movl      $.2.13_2_kmpc_loc_struct_pack.28, %edi        #52.9
        movl      %ebx, %esi                                    #52.9
        xorl      %eax, %eax                                    #52.9
..___tag_value_main.71:                                         #52.9
        call      __kmpc_for_static_fini                        #52.9
..___tag_value_main.72:                                         #
                                # LOE r13 ebx
..B1.56:                        # Preds ..B1.55
        addq      $-16, %rsp                                    #52.9
        movl      $.2.13_2_kmpc_loc_struct_pack.66, %r12d       #52.9
        movl      $L_main_52__tree_reduce1_2.73, %r9d           #52.9
        lea       72(%rsp), %r8                                 #52.9
        movq      %r12, %rdi                                    #52.9
        movl      %ebx, %esi                                    #52.9
        xorl      %edx, %edx                                    #52.9
        incl      %edx                                          #52.9
        movl      $4, %ecx                                      #52.9
        xorl      %eax, %eax                                    #52.9
        movq      $main_kmpc_tree_reduct_lock_0, (%rsp)         #52.9
..___tag_value_main.73:                                         #52.9
        call      __kmpc_reduce_nowait                          #52.9
..___tag_value_main.74:                                         #
                                # LOE r12 r13 eax ebx
..B1.87:                        # Preds ..B1.56
        addq      $16, %rsp                                     #52.9
                                # LOE r12 r13 eax ebx
..B1.57:                        # Preds ..B1.87
        cmpl      $1, %eax                                      #52.9
        jne       ..B1.59       # Prob 50%                      #52.9
                                # LOE r12 r13 eax ebx
..B1.58:                        # Preds ..B1.57
        movl      56(%rsp), %eax                                #52.9
        movl      $main_kmpc_tree_reduct_lock_0, %edx           #52.9
        addl      %eax, (%r13)                                  #52.9
        movq      %r12, %rdi                                    #52.9
        movl      %ebx, %esi                                    #52.9
        xorl      %eax, %eax                                    #52.9
..___tag_value_main.75:                                         #52.9
        call      __kmpc_end_reduce_nowait                      #52.9
..___tag_value_main.76:                                         #
        jmp       ..B1.61       # Prob 100%                     #52.9
                                # LOE
..B1.59:                        # Preds ..B1.57
        cmpl      $2, %eax                                      #52.9
        jne       ..B1.61       # Prob 50%                      #52.9
                                # LOE r12 r13 ebx
..B1.60:                        # Preds ..B1.59
        movq      %r12, %rdi                                    #52.9
        movl      %ebx, %esi                                    #52.9
        movq      %r13, %rdx                                    #52.9
        xorl      %eax, %eax                                    #52.9
        movl      56(%rsp), %ecx                                #52.9
..___tag_value_main.77:                                         #52.9
        call      __kmpc_atomic_fixed4_add                      #52.9
..___tag_value_main.78:                                         #
                                # LOE
..B1.61:                        # Preds ..B1.58 ..B1.60 ..B1.59
        movq      400(%rsp), %r15                               #52.9
..___tag_value_main.79:                                         #
        xorl      %eax, %eax                                    #52.9
        movq      408(%rsp), %r14                               #52.9
..___tag_value_main.80:                                         #
        movq      416(%rsp), %r13                               #52.9
..___tag_value_main.81:                                         #
        movq      424(%rsp), %r12                               #52.9
..___tag_value_main.82:                                         #
        movq      432(%rsp), %rbx                               #52.9
..___tag_value_main.83:                                         #
        movq      %rbp, %rsp                                    #52.9
        popq      %rbp                                          #52.9
..___tag_value_main.84:                                         #
        ret                                                     #52.9
..___tag_value_main.86:                                         #
                                # LOE
L_main_50__par_region0_2.74:
# parameter 1: %rdi
# parameter 2: %rsi
# parameter 3: %rdx
# parameter 4: %rcx
# parameter 5: %r8
# parameter 6: %r9
# parameter 7: 16 + %rbp
..B1.62:                        # Preds ..B1.0
        pushq     %rbp                                          #50.5
..___tag_value_main.88:                                         #
        movq      %rsp, %rbp                                    #50.5
..___tag_value_main.89:                                         #
        andq      $-128, %rsp                                   #50.5
        subq      $512, %rsp                                    #50.5
        xorl      %eax, %eax                                    #52.9
        movq      %rbx, 432(%rsp)                               #50.5
..___tag_value_main.91:                                         #
        movq      %rdi, %rbx                                    #50.5
        movl      $.2.13_2_kmpc_loc_struct_pack.28, %edi        #52.9
        movq      %r12, 424(%rsp)                               #50.5
        movq      %r15, 400(%rsp)                               #50.5
        movq      %r14, 408(%rsp)                               #50.5
..___tag_value_main.92:                                         #
        movq      %r9, %r14                                     #50.5
        movq      %r13, 416(%rsp)                               #50.5
..___tag_value_main.95:                                         #
        movq      %r8, %r13                                     #50.5
        movl      (%rbx), %r12d                                 #50.5
..___tag_value_main.96:                                         #52.9
        call      __kmpc_ok_to_fork                             #52.9
..___tag_value_main.97:                                         #
                                # LOE rbx r13 r14 eax r12d
..B1.63:                        # Preds ..B1.62
        testl     %eax, %eax                                    #52.9
        je        ..B1.65       # Prob 50%                      #52.9
                                # LOE rbx r13 r14 r12d
..B1.64:                        # Preds ..B1.63
        movl      $L_main_52__par_loop1_2.26, %edx              #52.9
        movl      $.2.13_2_kmpc_loc_struct_pack.28, %edi        #52.9
        movl      $3, %esi                                      #52.9
        movq      %r13, %rcx                                    #52.9
        movq      %r14, %r8                                     #52.9
        xorl      %eax, %eax                                    #52.9
        movq      16(%rbp), %r9                                 #52.9
..___tag_value_main.98:                                         #52.9
        call      __kmpc_fork_call                              #52.9
..___tag_value_main.99:                                         #
        jmp       ..B1.68       # Prob 100%                     #52.9
                                # LOE
..B1.65:                        # Preds ..B1.63
        movl      $.2.13_2_kmpc_loc_struct_pack.28, %edi        #52.9
        movl      %r12d, %esi                                   #52.9
        xorl      %eax, %eax                                    #52.9
..___tag_value_main.100:                                        #52.9
        call      __kmpc_serialized_parallel                    #52.9
..___tag_value_main.101:                                        #
                                # LOE rbx r13 r14 r12d
..B1.66:                        # Preds ..B1.65
        movl      $___kmpv_zeromain_1, %esi                     #52.9
        movq      %rbx, %rdi                                    #52.9
        movq      %r13, %rdx                                    #52.9
        movq      %r14, %rcx                                    #52.9
        movq      16(%rbp), %r8                                 #52.9
..___tag_value_main.102:                                        #52.9
        call      L_main_52__par_loop1_2.26                     #52.9
..___tag_value_main.103:                                        #
                                # LOE r12d
..B1.67:                        # Preds ..B1.66
        movl      $.2.13_2_kmpc_loc_struct_pack.28, %edi        #52.9
        movl      %r12d, %esi                                   #52.9
        xorl      %eax, %eax                                    #52.9
..___tag_value_main.104:                                        #52.9
        call      __kmpc_end_serialized_parallel                #52.9
..___tag_value_main.105:                                        #
                                # LOE
..B1.68:                        # Preds ..B1.64 ..B1.67
        movq      400(%rsp), %r15                               #50.5
..___tag_value_main.106:                                        #
        xorl      %eax, %eax                                    #50.5
        movq      408(%rsp), %r14                               #50.5
..___tag_value_main.107:                                        #
        movq      416(%rsp), %r13                               #50.5
..___tag_value_main.108:                                        #
        movq      424(%rsp), %r12                               #50.5
..___tag_value_main.109:                                        #
        movq      432(%rsp), %rbx                               #50.5
..___tag_value_main.110:                                        #
        movq      %rbp, %rsp                                    #50.5
        popq      %rbp                                          #50.5
..___tag_value_main.111:                                        #
        ret                                                     #50.5
        .align    16,0x90
..___tag_value_main.113:                                        #
                                # LOE
# mark_end;
	.type	main,@function
	.size	main,.-main
	.data
	.align 8
	.align 8
.2.13_2__offload_var_desc1_p.92:
	.byte	17
	.byte	3
	.byte	1
	.byte	1
	.long	4
	.long	0
	.long	0
	.long	0x00000000,0x00000000
	.long	0x00000004,0x00000000
	.long	0x00000001,0x00000000
	.long	0x00000000,0x00000000
	.long	0x00000000,0x00000000
	.long	0x00000000,0x00000000
	.byte	17
	.byte	3
	.byte	1
	.byte	1
	.long	8
	.long	0
	.long	0
	.long	0x00000000,0x00000000
	.long	0x00000008,0x00000000
	.long	0x00000001,0x00000000
	.long	0x00000000,0x00000000
	.long	0x00000000,0x00000000
	.long	0x00000000,0x00000000
	.byte	17
	.byte	3
	.byte	1
	.byte	1
	.long	4
	.long	0
	.long	0
	.long	0x00000000,0x00000000
	.long	0x00000004,0x00000000
	.long	0x00000001,0x00000000
	.long	0x00000000,0x00000000
	.long	0x00000000,0x00000000
	.long	0x00000000,0x00000000
	.byte	17
	.byte	3
	.byte	1
	.byte	1
	.long	8
	.long	0
	.long	0
	.long	0x00000000,0x00000000
	.long	0x00000008,0x00000000
	.long	0x00000001,0x00000000
	.long	0x00000000,0x00000000
	.long	0x00000000,0x00000000
	.long	0x00000000,0x00000000
	.byte	17
	.byte	3
	.byte	1
	.byte	1
	.long	4
	.long	0
	.long	0
	.long	0x00000000,0x00000000
	.long	0x00000004,0x00000000
	.long	0x00000001,0x00000000
	.long	0x00000000,0x00000000
	.long	0x00000000,0x00000000
	.long	0x00000000,0x00000000
	.align 8
.2.13_2__offload_var_desc2_p.97:
	.quad	__sd_2inst_string.3
	.long	0x00000000,0x00000000
	.quad	__sd_2inst_string.4
	.long	0x00000000,0x00000000
	.quad	__sd_2inst_string.5
	.long	0x00000000,0x00000000
	.quad	__sd_2inst_string.6
	.long	0x00000000,0x00000000
	.quad	__sd_2inst_string.7
	.long	0x00000000,0x00000000
	.align 4
.2.13_2_kmpc_loc_struct_pack.8:
	.long	0
	.long	2
	.long	0
	.long	0
	.quad	.2.13_2__kmpc_loc_pack.7
	.align 4
.2.13_2__kmpc_loc_pack.7:
	.byte	59
	.byte	117
	.byte	110
	.byte	107
	.byte	110
	.byte	111
	.byte	119
	.byte	110
	.byte	59
	.byte	109
	.byte	97
	.byte	105
	.byte	110
	.byte	59
	.byte	50
	.byte	48
	.byte	59
	.byte	50
	.byte	48
	.byte	59
	.byte	59
	.space 3, 0x00 	# pad
	.align 4
.2.13_2_kmpc_loc_struct_pack.19:
	.long	0
	.long	2
	.long	0
	.long	0
	.quad	.2.13_2__kmpc_loc_pack.18
	.align 4
.2.13_2__kmpc_loc_pack.18:
	.byte	59
	.byte	117
	.byte	110
	.byte	107
	.byte	110
	.byte	111
	.byte	119
	.byte	110
	.byte	59
	.byte	109
	.byte	97
	.byte	105
	.byte	110
	.byte	59
	.byte	53
	.byte	48
	.byte	59
	.byte	53
	.byte	48
	.byte	59
	.byte	59
	.space 3, 0x00 	# pad
	.align 4
.2.13_2_kmpc_loc_struct_pack.28:
	.long	0
	.long	2
	.long	0
	.long	0
	.quad	.2.13_2__kmpc_loc_pack.27
	.align 4
.2.13_2__kmpc_loc_pack.27:
	.byte	59
	.byte	117
	.byte	110
	.byte	107
	.byte	110
	.byte	111
	.byte	119
	.byte	110
	.byte	59
	.byte	109
	.byte	97
	.byte	105
	.byte	110
	.byte	59
	.byte	53
	.byte	50
	.byte	59
	.byte	56
	.byte	55
	.byte	59
	.byte	59
	.space 3, 0x00 	# pad
	.align 4
.2.13_2_kmpc_loc_struct_pack.66:
	.long	0
	.long	18
	.long	0
	.long	0
	.quad	.2.13_2__kmpc_loc_pack.65
	.align 4
.2.13_2__kmpc_loc_pack.65:
	.byte	59
	.byte	117
	.byte	110
	.byte	107
	.byte	110
	.byte	111
	.byte	119
	.byte	110
	.byte	59
	.byte	109
	.byte	97
	.byte	105
	.byte	110
	.byte	59
	.byte	53
	.byte	50
	.byte	59
	.byte	53
	.byte	50
	.byte	59
	.byte	59
	.section .rodata, "a"
	.align 16
	.align 8
..1..TPKT.get_refword.2_0.0.2.11:
	.quad	..1.2_0.TAG.00.0.2.11
	.quad	..1.2_0.TAG.01.0.2.11
	.quad	..1.2_0.TAG.02.0.2.11
	.quad	..1.2_0.TAG.03.0.2.11
	.quad	..1.2_0.TAG.04.0.2.11
	.align 4
__sd_2inst_string.1:
	.byte	114
	.byte	101
	.byte	103
	.byte	46
	.byte	99
	.byte	0
	.space 2, 0x00 	# pad
	.align 4
__sd_2inst_string.3:
	.byte	116
	.byte	104
	.byte	95
	.byte	105
	.byte	100
	.byte	0
	.space 2, 0x00 	# pad
	.align 4
__sd_2inst_string.4:
	.byte	105
	.byte	0
	.space 2, 0x00 	# pad
	.align 4
__sd_2inst_string.5:
	.byte	114
	.byte	101
	.byte	102
	.byte	0
	.align 4
__sd_2inst_string.6:
	.byte	114
	.byte	101
	.byte	112
	.byte	101
	.byte	116
	.byte	105
	.byte	116
	.byte	105
	.byte	111
	.byte	110
	.byte	115
	.byte	0
	.align 4
__sd_2inst_string.7:
	.byte	101
	.byte	114
	.byte	114
	.byte	111
	.byte	114
	.byte	95
	.byte	99
	.byte	111
	.byte	117
	.byte	110
	.byte	116
	.byte	0
	.align 4
__sd_2inst_string.2:
	.byte	95
	.byte	95
	.byte	111
	.byte	102
	.byte	102
	.byte	108
	.byte	111
	.byte	97
	.byte	100
	.byte	95
	.byte	101
	.byte	110
	.byte	116
	.byte	114
	.byte	121
	.byte	95
	.byte	114
	.byte	101
	.byte	103
	.byte	95
	.byte	99
	.byte	95
	.byte	53
	.byte	48
	.byte	109
	.byte	97
	.byte	105
	.byte	110
	.byte	105
	.byte	99
	.byte	99
	.byte	50
	.byte	51
	.byte	50
	.byte	56
	.byte	50
	.byte	49
	.byte	54
	.byte	54
	.byte	52
	.byte	57
	.byte	85
	.byte	107
	.byte	65
	.byte	53
	.byte	105
	.byte	0
	.space 1, 0x00 	# pad
	.align 4
__sd_2inst_string.0:
	.byte	95
	.byte	95
	.byte	111
	.byte	102
	.byte	102
	.byte	108
	.byte	111
	.byte	97
	.byte	100
	.byte	95
	.byte	101
	.byte	110
	.byte	116
	.byte	114
	.byte	121
	.byte	95
	.byte	114
	.byte	101
	.byte	103
	.byte	95
	.byte	99
	.byte	95
	.byte	53
	.byte	48
	.byte	109
	.byte	97
	.byte	105
	.byte	110
	.byte	105
	.byte	99
	.byte	99
	.byte	50
	.byte	51
	.byte	50
	.byte	56
	.byte	50
	.byte	49
	.byte	54
	.byte	54
	.byte	52
	.byte	57
	.byte	85
	.byte	107
	.byte	65
	.byte	53
	.byte	105
	.byte	0
	.section .rodata.str1.4, "aMS",@progbits,1
	.align 4
	.align 4
il0_peep_printf_format_1:
	.long	2003199314
	.long	543453807
	.long	1769238639
	.long	3829359
	.align 4
il0_peep_printf_format_2:
	.long	540286985
	.long	1179394109
	.long	673206867
	.long	1145979218
	.long	2706767
	.section .rodata.str1.32, "aMS",@progbits,1
	.align 32
	.align 32
il0_peep_printf_format_0:
	.long	1634036816
	.long	1881171315
	.long	1769369458
	.long	1948280164
	.long	1847616872
	.long	1700949365
	.long	1718558834
	.long	1701985312
	.long	1769235824
	.long	1852795252
	.long	1629503091
	.long	1008755822
	.long	2003199346
	.long	543453807
	.long	1769238639
	.long	775843439
	.byte	0
	.section .OffloadEntryTable., "waG",@progbits,__offload_entry_reg_c_50mainicc2328216649UkA5i_$entry,comdat
	.align 16
__offload_entry_reg_c_50mainicc2328216649UkA5i_$entry:
	.type	__offload_entry_reg_c_50mainicc2328216649UkA5i_$entry,@object
	.size	__offload_entry_reg_c_50mainicc2328216649UkA5i_$entry,16
	.quad	__sd_2inst_string.0
	.quad	__sd_2inst_string.0
	.data
# -- End  main, L_main_52__par_loop1_2.26, L_main_52__tree_reduce1_2.73, L_main_50__par_region0_2.74
	.text
# -- Begin  string_to_uint64
# mark_begin;
       .align    16,0x90
	.globl string_to_uint64
string_to_uint64:
# parameter 1: %rdi
..B2.1:                         # Preds ..B2.0
..___tag_value_string_to_uint64.114:                            #10.41
        xorl      %eax, %eax                                    #11.21
        movb      (%rdi), %dl                                   #14.20
        xorb      $48, %dl                                      #14.29
        cmpb      $9, %dl                                       #14.37
        jg        ..B2.6        # Prob 10%                      #14.37
                                # LOE rax rbx rbp rdi r12 r13 r14 r15 dl
..B2.3:                         # Preds ..B2.1 ..B2.4
        testb     %dl, %dl                                      #14.47
        jl        ..B2.6        # Prob 20%                      #14.47
                                # LOE rax rbx rbp rdi r12 r13 r14 r15 dl
..B2.4:                         # Preds ..B2.3
        movsbq    %dl, %rdx                                     #15.32
        incq      %rdi                                          #14.52
        lea       (%rax,%rax,4), %rax                           #15.27
        lea       (%rdx,%rax,2), %rax                           #15.32
        movb      (%rdi), %dl                                   #14.20
        xorb      $48, %dl                                      #14.29
        cmpb      $9, %dl                                       #14.37
        jle       ..B2.3        # Prob 82%                      #14.37
                                # LOE rax rbx rbp rdi r12 r13 r14 r15 dl
..B2.6:                         # Preds ..B2.3 ..B2.4 ..B2.1
        ret                                                     #17.12
        .align    16,0x90
..___tag_value_string_to_uint64.116:                            #
                                # LOE
# mark_end;
	.type	string_to_uint64,@function
	.size	string_to_uint64,.-string_to_uint64
	.data
# -- End  string_to_uint64
	.text
# -- Begin  lfsr
# mark_begin;
       .align    16,0x90
	.globl lfsr
lfsr:
..B3.1:                         # Preds ..B3.0
..___tag_value_lfsr.117:                                        #25.16
        movq      lfsr.19.0.5(%rip), %rdx                       #30.14
        movq      %rdx, %rcx                                    #29.37
        shrq      $10, %rcx                                     #29.37
        movq      %rdx, %rax                                    #29.52
        shrq      $11, %rax                                     #29.52
        xorq      %rdx, %rcx                                    #29.37
        movq      %rdx, %rsi                                    #30.22
        xorq      %rax, %rcx                                    #29.52
        shrq      $30, %rdx                                     #29.67
        xorq      %rdx, %rcx                                    #29.67
        notq      %rcx                                          #29.67
        andq      $1, %rcx                                      #29.75
        shrq      $1, %rsi                                      #30.22
        shlq      $31, %rcx                                     #30.35
        orq       %rcx, %rsi                                    #30.35
        movq      %rsi, lfsr.19.0.5(%rip)                       #30.5
        movl      %esi, %eax                                    #32.12
        ret                                                     #32.12
        .align    16,0x90
..___tag_value_lfsr.119:                                        #
                                # LOE
# mark_end;
	.type	lfsr,@function
	.size	lfsr,.-lfsr
	.data
# -- End  lfsr
	.section .text.print_refword, "xaG",@progbits,print_refword,comdat
..TXTST1:
# -- Begin  print_refword
# mark_begin;
       .align    16,0x90
	.weak print_refword
print_refword:
..B4.1:                         # Preds ..B4.0
..___tag_value_print_refword.120:                               #36.29
        pushq     %rsi                                          #36.29
..___tag_value_print_refword.122:                               #
        movl      $il0_peep_printf_format_3, %edi               #37.5
        call      puts                                          #37.5
                                # LOE rbx rbp r12 r13 r14 r15
..B4.2:                         # Preds ..B4.1
        movl      $.L_2__STRING.1, %edi                         #38.5
        xorl      %esi, %esi                                    #38.5
        xorl      %eax, %eax                                    #38.5
..___tag_value_print_refword.123:                               #38.5
        call      printf                                        #38.5
..___tag_value_print_refword.124:                               #
                                # LOE rbx rbp r12 r13 r14 r15
..B4.3:                         # Preds ..B4.2
        movl      $.L_2__STRING.2, %edi                         #39.5
        movl      $1, %esi                                      #39.5
        xorl      %eax, %eax                                    #39.5
..___tag_value_print_refword.125:                               #39.5
        call      printf                                        #39.5
..___tag_value_print_refword.126:                               #
                                # LOE rbx rbp r12 r13 r14 r15
..B4.4:                         # Preds ..B4.3
        movl      $.L_2__STRING.3, %edi                         #40.5
        movl      $1431655765, %esi                             #40.5
        xorl      %eax, %eax                                    #40.5
..___tag_value_print_refword.127:                               #40.5
        call      printf                                        #40.5
..___tag_value_print_refword.128:                               #
                                # LOE rbx rbp r12 r13 r14 r15
..B4.5:                         # Preds ..B4.4
        movl      $.L_2__STRING.4, %edi                         #41.5
        movl      $-1, %esi                                     #41.5
        xorl      %eax, %eax                                    #41.5
..___tag_value_print_refword.129:                               #41.5
        call      printf                                        #41.5
..___tag_value_print_refword.130:                               #
                                # LOE rbx rbp r12 r13 r14 r15
..B4.6:                         # Preds ..B4.5
        movl      $il0_peep_printf_format_4, %edi               #42.5
        addq      $8, %rsp                                      #42.5
..___tag_value_print_refword.131:                               #
        jmp       puts                                          #42.5
        .align    16,0x90
..___tag_value_print_refword.132:                               #
                                # LOE
# mark_end;
	.type	print_refword,@function
	.size	print_refword,.-print_refword
	.section .rodata.str1.4, "aMS",@progbits,1
	.align 4
il0_peep_printf_format_3:
	.long	2003199314
	.long	543453807
	.long	1769238639
	.long	3829359
	.align 4
il0_peep_printf_format_4:
	.long	540286985
	.long	1179394109
	.long	673206867
	.long	1145979218
	.long	2706767
	.data
# -- End  print_refword
	.section .text.get_refword, "xaG",@progbits,get_refword,comdat
..TXTST2:
# -- Begin  get_refword
# mark_begin;
       .align    16,0x90
	.weak get_refword
get_refword:
# parameter 1: %edi
..B5.1:                         # Preds ..B5.0
..___tag_value_get_refword.133:                                 #46.43
        cmpl      $4, %edi                                      #47.5
        ja        ..B5.13       # Prob 28%                      #47.5
                                # LOE rbx rbp r12 r13 r14 r15 edi
..B5.2:                         # Preds ..B5.1
        movl      %edi, %edi                                    #47.5
        jmp       *..1..TPKT.get_refword.2_0.0.2(,%rdi,8)       #47.5
                                # LOE rbx rbp r12 r13 r14 r15
..1.2_0.TAG.04.0.2:
..B5.4:                         # Preds ..B5.2
        movq      lfsr.19.0.5(%rip), %rdx                       #52.24
        movq      %rdx, %rcx                                    #52.24
        shrq      $10, %rcx                                     #52.24
        movq      %rdx, %rax                                    #52.24
        shrq      $11, %rax                                     #52.24
        xorq      %rdx, %rcx                                    #52.24
        movq      %rdx, %rsi                                    #52.24
        xorq      %rax, %rcx                                    #52.24
        shrq      $30, %rdx                                     #52.24
        xorq      %rdx, %rcx                                    #52.24
        notq      %rcx                                          #52.24
        andq      $1, %rcx                                      #52.24
        shrq      $1, %rsi                                      #52.24
        shlq      $31, %rcx                                     #52.24
        orq       %rcx, %rsi                                    #52.24
        movq      %rsi, lfsr.19.0.5(%rip)                       #52.24
        movl      %esi, %eax                                    #52.24
        ret                                                     #52.24
                                # LOE
..1.2_0.TAG.03.0.2:
..B5.6:                         # Preds ..B5.2
        movl      $-1, %eax                                     #51.24
        ret                                                     #51.24
                                # LOE
..1.2_0.TAG.02.0.2:
..B5.8:                         # Preds ..B5.2
        movl      $1431655765, %eax                             #50.24
        ret                                                     #50.24
                                # LOE
..1.2_0.TAG.01.0.2:
..B5.10:                        # Preds ..B5.2
        movl      $1, %eax                                      #49.24
        ret                                                     #49.24
                                # LOE
..1.2_0.TAG.00.0.2:
..B5.12:                        # Preds ..B5.2
        xorl      %eax, %eax                                    #48.24
        ret                                                     #48.24
                                # LOE
..B5.13:                        # Preds ..B5.1
        ret                                                     #54.1
        .align    16,0x90
..___tag_value_get_refword.135:                                 #
                                # LOE
# mark_end;
	.type	get_refword,@function
	.size	get_refword,.-get_refword
	.section .data...1..TPKT.get_refword.2_0.0.2, "waG",@progbits,get_refword,comdat
	.align 8
..1..TPKT.get_refword.2_0.0.2:
	.type	..1..TPKT.get_refword.2_0.0.2,@object
	.size	..1..TPKT.get_refword.2_0.0.2,40
	.quad	..1.2_0.TAG.00.0.2
	.quad	..1.2_0.TAG.01.0.2
	.quad	..1.2_0.TAG.02.0.2
	.quad	..1.2_0.TAG.03.0.2
	.quad	..1.2_0.TAG.04.0.2
	.data
# -- End  get_refword
	.bss
	.align 4
	.align 4
___kmpv_zeromain_0:
	.type	___kmpv_zeromain_0,@object
	.size	___kmpv_zeromain_0,4
	.space 4	# pad
	.align 4
___kmpv_zeromain_1:
	.type	___kmpv_zeromain_1,@object
	.size	___kmpv_zeromain_1,4
	.space 4	# pad
	.data
	.space 3, 0x00 	# pad
	.align 8
lfsr.19.0.5:
	.long	0x80000000,0x00000000
	.type	lfsr.19.0.5,@object
	.size	lfsr.19.0.5,8
	.section .rodata, "a"
	.space 9, 0x00 	# pad
	.align 16
.L_2il0floatpacket.140:
	.long	0x00000001,0x00000001,0x00000001,0x00000001
	.type	.L_2il0floatpacket.140,@object
	.size	.L_2il0floatpacket.140,16
	.section .rodata.str1.4, "aMS",@progbits,1
	.align 4
.L_2__STRING.1:
	.long	540024841
	.long	2016419901
	.long	2016948261
	.word	10
	.type	.L_2__STRING.1,@object
	.size	.L_2__STRING.1,14
	.space 2, 0x00 	# pad
	.align 4
.L_2__STRING.2:
	.long	540090377
	.long	2016419901
	.long	2016948261
	.word	10
	.type	.L_2__STRING.2,@object
	.size	.L_2__STRING.2,14
	.space 2, 0x00 	# pad
	.align 4
.L_2__STRING.3:
	.long	540155913
	.long	2016419901
	.long	2016948261
	.word	10
	.type	.L_2__STRING.3,@object
	.size	.L_2__STRING.3,14
	.space 2, 0x00 	# pad
	.align 4
.L_2__STRING.4:
	.long	540221449
	.long	2016419901
	.long	2016948261
	.word	10
	.type	.L_2__STRING.4,@object
	.size	.L_2__STRING.4,14
	.space 2, 0x00 	# pad
	.align 4
.L_2__STRING.7:
	.long	1701864786
	.long	1769236852
	.long	980643439
	.long	175467557
	.byte	0
	.type	.L_2__STRING.7,@object
	.size	.L_2__STRING.7,17
	.space 3, 0x00 	# pad
	.align 4
.L_2__STRING.8:
	.long	543581522
	.long	1685221207
	.long	628633658
	.long	175650864
	.byte	0
	.type	.L_2__STRING.8,@object
	.size	.L_2__STRING.8,17
	.space 3, 0x00 	# pad
	.align 4
.L_2__STRING.9:
	.long	1701996628
	.long	980640865
	.long	685349
	.type	.L_2__STRING.9,@object
	.size	.L_2__STRING.9,12
	.align 4
.L_2__STRING.11:
	.long	1869771333
	.long	540701554
	.long	685349
	.type	.L_2__STRING.11,@object
	.size	.L_2__STRING.11,12
	.align 4
.L_2__STRING.10:
	.long	1763730469
	.long	622865524
	.long	1869619300
	.long	622865523
	.long	1752440932
	.long	1684104562
	.long	2016419884
	.long	2016948261
	.long	1853453088
	.long	1836020324
	.word	2661
	.byte	0
	.type	.L_2__STRING.10,@object
	.size	.L_2__STRING.10,43
	.data
	.comm main_kmpc_tree_reduct_lock_0,32,8
# mark_proc_addr_taken L_main_52__tree_reduce1_2.73;
# mark_proc_addr_taken L_main_52__par_loop1_2.26;
	.section .note.GNU-stack, ""
// -- Begin DWARF2 SEGMENT .eh_frame
	.section .eh_frame,"a",@progbits
.eh_frame_seg:
	.align 8
	.4byte 0x00000014
	.8byte 0x7801000100000000
	.8byte 0x0000019008070c10
	.4byte 0x00000000
	.4byte 0x0000025c
	.4byte 0x0000001c
	.8byte ..___tag_value_main.1
	.8byte ..___tag_value_main.113-..___tag_value_main.1
	.byte 0x04
	.4byte ..___tag_value_main.3-..___tag_value_main.1
	.2byte 0x100e
	.byte 0x04
	.4byte ..___tag_value_main.4-..___tag_value_main.3
	.4byte 0x8610060c
	.2byte 0x0402
	.4byte ..___tag_value_main.6-..___tag_value_main.4
	.8byte 0xff800d1c380e0310
	.8byte 0xffffffb00d1affff
	.2byte 0x0422
	.4byte ..___tag_value_main.7-..___tag_value_main.6
	.8byte 0xff800d1c380e0c10
	.8byte 0xffffffa80d1affff
	.2byte 0x0422
	.4byte ..___tag_value_main.8-..___tag_value_main.7
	.8byte 0xff800d1c380e0d10
	.8byte 0xffffffa00d1affff
	.8byte 0x800d1c380e0e1022
	.8byte 0xffff980d1affffff
	.8byte 0x0d1c380e0f1022ff
	.8byte 0xff900d1affffff80
	.4byte 0x0422ffff
	.4byte ..___tag_value_main.37-..___tag_value_main.8
	.4byte 0xcdccc6c3
	.2byte 0xcfce
	.byte 0x04
	.4byte ..___tag_value_main.44-..___tag_value_main.37
	.4byte 0x0410070c
	.4byte ..___tag_value_main.45-..___tag_value_main.44
	.4byte 0x8610060c
	.2byte 0x0402
	.4byte ..___tag_value_main.47-..___tag_value_main.45
	.8byte 0xff800d1c380e0310
	.8byte 0xffffffb00d1affff
	.8byte 0x800d1c380e0c1022
	.8byte 0xffffa80d1affffff
	.8byte 0x0d1c380e0d1022ff
	.8byte 0xffa00d1affffff80
	.8byte 0x1c380e0e1022ffff
	.8byte 0x980d1affffff800d
	.4byte 0x22ffffff
	.byte 0x04
	.4byte ..___tag_value_main.51-..___tag_value_main.47
	.2byte 0x04ce
	.4byte ..___tag_value_main.52-..___tag_value_main.51
	.2byte 0x04cd
	.4byte ..___tag_value_main.53-..___tag_value_main.52
	.2byte 0x04cc
	.4byte ..___tag_value_main.54-..___tag_value_main.53
	.2byte 0x04c3
	.4byte ..___tag_value_main.55-..___tag_value_main.54
	.4byte 0xc608070c
	.byte 0x04
	.4byte ..___tag_value_main.57-..___tag_value_main.55
	.4byte 0x0410060c
	.4byte ..___tag_value_main.59-..___tag_value_main.57
	.4byte 0x0410070c
	.4byte ..___tag_value_main.60-..___tag_value_main.59
	.4byte 0x8610060c
	.2byte 0x0402
	.4byte ..___tag_value_main.62-..___tag_value_main.60
	.8byte 0xff800d1c380e0310
	.8byte 0xffffffb00d1affff
	.8byte 0x800d1c380e0d1022
	.8byte 0xffffa00d1affffff
	.8byte 0x0d1c380e0e1022ff
	.8byte 0xff980d1affffff80
	.8byte 0x1c380e0f1022ffff
	.8byte 0x900d1affffff800d
	.4byte 0x22ffffff
	.byte 0x04
	.4byte ..___tag_value_main.66-..___tag_value_main.62
	.8byte 0xff800d1c380e0c10
	.8byte 0xffffffa80d1affff
	.2byte 0x0422
	.4byte ..___tag_value_main.79-..___tag_value_main.66
	.2byte 0x04cf
	.4byte ..___tag_value_main.80-..___tag_value_main.79
	.2byte 0x04ce
	.4byte ..___tag_value_main.81-..___tag_value_main.80
	.2byte 0x04cd
	.4byte ..___tag_value_main.82-..___tag_value_main.81
	.2byte 0x04cc
	.4byte ..___tag_value_main.83-..___tag_value_main.82
	.2byte 0x04c3
	.4byte ..___tag_value_main.84-..___tag_value_main.83
	.4byte 0xc608070c
	.byte 0x04
	.4byte ..___tag_value_main.86-..___tag_value_main.84
	.4byte 0x0410060c
	.4byte ..___tag_value_main.88-..___tag_value_main.86
	.4byte 0x0410070c
	.4byte ..___tag_value_main.89-..___tag_value_main.88
	.4byte 0x8610060c
	.2byte 0x0402
	.4byte ..___tag_value_main.91-..___tag_value_main.89
	.8byte 0xff800d1c380e0310
	.8byte 0xffffffb00d1affff
	.2byte 0x0422
	.4byte ..___tag_value_main.92-..___tag_value_main.91
	.8byte 0xff800d1c380e0c10
	.8byte 0xffffffa80d1affff
	.8byte 0x800d1c380e0e1022
	.8byte 0xffff980d1affffff
	.8byte 0x0d1c380e0f1022ff
	.8byte 0xff900d1affffff80
	.4byte 0x0422ffff
	.4byte ..___tag_value_main.95-..___tag_value_main.92
	.8byte 0xff800d1c380e0d10
	.8byte 0xffffffa00d1affff
	.2byte 0x0422
	.4byte ..___tag_value_main.106-..___tag_value_main.95
	.2byte 0x04cf
	.4byte ..___tag_value_main.107-..___tag_value_main.106
	.2byte 0x04ce
	.4byte ..___tag_value_main.108-..___tag_value_main.107
	.2byte 0x04cd
	.4byte ..___tag_value_main.109-..___tag_value_main.108
	.2byte 0x04cc
	.4byte ..___tag_value_main.110-..___tag_value_main.109
	.2byte 0x04c3
	.4byte ..___tag_value_main.111-..___tag_value_main.110
	.8byte 0x00000000c608070c
	.2byte 0x0000
	.byte 0x00
	.4byte 0x00000014
	.4byte 0x0000027c
	.8byte ..___tag_value_string_to_uint64.114
	.8byte ..___tag_value_string_to_uint64.116-..___tag_value_string_to_uint64.114
	.4byte 0x00000014
	.4byte 0x00000294
	.8byte ..___tag_value_lfsr.117
	.8byte ..___tag_value_lfsr.119-..___tag_value_lfsr.117
	.4byte 0x00000024
	.4byte 0x000002ac
	.8byte ..___tag_value_print_refword.120
	.8byte ..___tag_value_print_refword.132-..___tag_value_print_refword.120
	.byte 0x04
	.4byte ..___tag_value_print_refword.122-..___tag_value_print_refword.120
	.2byte 0x100e
	.byte 0x04
	.4byte ..___tag_value_print_refword.131-..___tag_value_print_refword.122
	.4byte 0x0000080e
	.4byte 0x00000014
	.4byte 0x000002d4
	.8byte ..___tag_value_get_refword.133
	.8byte ..___tag_value_get_refword.135-..___tag_value_get_refword.133
# End
