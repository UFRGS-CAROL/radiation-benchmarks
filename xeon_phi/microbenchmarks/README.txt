
                            MICROBENCHMARKS

================================================================================
    CACHE
================================================================================

This benchmark will allocate an array dynamically and send it to the target
(XeonPhi) on every iteration. Inside the target, each thread will write the
same reference word in all the memory slice. On every iteration the
reference word alternates between: 0x0, 0xFFFFFFFF or 0x55555555.

After the write procedure, an busy wait formed with a large number of nops,
will ensure that the processor will wait for about 3 seconds before it
checks the memory content.  During this wait time, we expect the memory to
suffer some radiation effect.

After the busy wait period the memory array is checked for the reference
word, to check that all the elements still the same.

This benchmark uses only 1 thread per physical processor core. In order to
place one thread per physical core, the user must make sure to use the
environment variables:
    MIC_ENV_PREFIX=PHI
    PHI_KMP_AFFINITY='granularity=fine,scatter'

Location:
    ./cache/cache.c

Usage:
    ./cache <repetitions> <total_size>

If the number of repetitions set to 0, it means infinite loop.
The total size, will be split by the threads.

Examples:
    ./cache 0 $((28672 * 56))     # L1 Cache (28KB per core)
    ./cache 0 $((491520 * 56))    # L2 Cache (480KB per core)



================================================================================
    LOGIC + ARITHMETICS
================================================================================

The benchmarks AND, OR, ROL and SLR all implements the 3 basic arithmetic
operations ADD, MUL and DIV, plus one specific logic operation. The
operations are implemented using "asm volatile" in order to the compiler not
to simplify them.

Each thread will execute only one specific operation, in order to have each
thread inside the physical processor core stressing a different component.
The logic operations supported are AND, OR, shift left and right (SLR),
shift and rotate left (ROL)

In order to ensure that one thread executing each operation will be placed
to the physical core, the user must make sure to use the environment
variables:
    MIC_ENV_PREFIX=PHI
    PHI_KMP_AFFINITY='granularity=fine,compact'


Location:
    scalar/and/and.c
    scalar/or/or.c
    scalar/slr/slr.c
    scalar/rol/rol.c

    vector/and/and_int.c
    vector/and/and_fpd.c
    vector/or/or_int.c
    vector/or/or_fpd.c
    vector/slr/slr_int.c
    vector/slr/slr_fpd.c


Usage:
    ./and <repetitions>     ##  Repetitions set to 0, means infinite loop.



================================================================================
    REGISTERS
================================================================================

The benchmark will perform MOV operations storing the reference word into (8
scalar) or (32 vector) different registers per time. On every iteration the
reference word alternates between: 0x0, 0xFFFFFFFF or 0x55555555.

After the move procedure, an busy wait formed with a large number of nops,
will ensure that the processor will wait for about 3 seconds before it
checks the registers content.  During this wait time, we expect the
registers to suffer some radiation effect.

After the busy wait, the registers are checked to see if any value is
different from the reference word.

Location:
    scalar/reg/reg.c
    vector/reg/reg.c

Usage:
    ./reg <repetitions>     ##  Repetitions set to 0, means infinite loop.



================================================================================
    LOG SYSTEM
================================================================================

The global log system is formed by the following methods:
    start_log_file(const char benchmark, char *msg)  // Create a new log file
    start_iteration();                               // Write the log header
    set_max_errors_iter(int error_repetitions);      // Set the maximum number of errors before forced exit
    log_error_detail(char log[x][y]);                // Write the error details
    end_log_file();                                  // Create the footer

Inside each application, in order to store the log of any error detected
during the benchmark execution we have an specific data structure. On every
iteration execution sent to the target, the host also sends an log
structure. This log structure is a matrix where each line is allocated to be
used by a different core. The columns can store multiple errors found by a
specific processor core.



================================================================================
    DEBUG MODE
================================================================================

All the benchmarks have a define called DEBUG. If set, it will generate an
error to check the log generation mechanism.

The global makefile already is set to create the normal and the debug
version, using the prefix _DEBUG.

The debug mode also checks if the threads are correctly placed into the
processor cores.
