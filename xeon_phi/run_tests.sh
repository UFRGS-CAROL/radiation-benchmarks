################################################################################
##
##      Commands to execute all the benchmarks
##
################################################################################
source /opt/intel/composerxe/bin/compilervars.sh intel64
#make
################################################################################
##
##      Infinite repetitions
##
################################################################################
./cache/cache 0 $((28672 * 56))     # L1 Cache (28KB per core)
./cache/cache 0 $((491520 * 56))    # L2 Cache (480KB per core)
###############################################
./scalar/reg/reg 0      # Scalar registers
./scalar/and/and 0      # Scalar AND + ADD + MUL + DIV
./scalar/or/or   0      # Scalar OR  + ADD + MUL + DIV
./scalar/slr/slr 0      # Scalar SLR + ADD + MUL + DIV
./scalar/rol/rol 0      # Scalar ROL + ADD + MUL + DIV
###############################################
./vector/reg/reg     0      # Vectorial registers:  32x ZMM#
./vector/and/and_int 0      # Vectorial Integer:    AND + ADD + MUL
./vector/and/and_fpd 0      # Vectorial FP Double:  AND + ADD + MUL
./vector/or/or_int   0      # Vectorial Integer:    OR  + ADD + MUL
./vector/or/or_fpd   0      # Vectorial FP Double:  OR  + ADD + MUL
./vector/slr/slr_int 0      # Vectorial Integer:    SLR + ADD + MUL
./vector/slr/slr_fpd 0      # Vectorial FP Double:  SLR + ADD + MUL
################################################################################
##
##      10 repetitions only
##
################################################################################
#time ./cache/cache 10 $((28672 * 56))     # L1 Cache (28KB per core)
#time ./cache/cache 10 $((491520 * 56))    # L2 Cache (480KB per core)
################################################
#time ./scalar/reg/reg 10      # Scalar registers
#time ./scalar/and/and 10      # Scalar AND + ADD + MUL + DIV
#time ./scalar/or/or   10      # Scalar OR  + ADD + MUL + DIV
#time ./scalar/slr/slr 10      # Scalar SLR + ADD + MUL + DIV
#time ./scalar/rol/rol 10      # Scalar ROL + ADD + MUL + DIV
################################################
#time ./vector/reg/reg     10      # Vectorial registers:  32x ZMM#
#time ./vector/and/and_int 10      # Vectorial Integer:    AND + ADD + MUL
#time ./vector/and/and_fpd 10      # Vectorial FP Double:  AND + ADD + MUL
#time ./vector/or/or_int   10      # Vectorial Integer:    OR  + ADD + MUL
#time ./vector/or/or_fpd   10      # Vectorial FP Double:  OR  + ADD + MUL
#time ./vector/slr/slr_int 10      # Vectorial Integer:    SLR + ADD + MUL
#time ./vector/slr/slr_fpd 10      # Vectorial FP Double:  SLR + ADD + MUL
#################################################################################
###
###      10 repetitions only + DEBUG
###
#################################################################################
#./cache/cache_debug 10 $((28672 * 56))     # L1 Cache (28KB per core)
#./cache/cache_debug 10 $((491520 * 56))    # L2 Cache (480KB per core)
################################################
#./scalar/reg/reg_debug 10      # Scalar registers
#./scalar/and/and_debug 10      # Scalar AND + ADD + MUL + DIV
#./scalar/or/or_debug   10      # Scalar OR  + ADD + MUL + DIV
#./scalar/slr/slr_debug 10      # Scalar SLR + ADD + MUL + DIV
#./scalar/rol/rol_debug 10      # Scalar ROL + ADD + MUL + DIV
################################################
#./vector/reg/reg_debug     10      # Vectorial registers:  32x ZMM#
#./vector/and/and_int_debug 10      # Vectorial Integer:    AND + ADD + MUL
#./vector/and/and_fpd_debug 10      # Vectorial FP Double:  AND + ADD + MUL
#./vector/or/or_int_debug   10      # Vectorial Integer:    OR  + ADD + MUL
#./vector/or/or_fpd_debug   10      # Vectorial FP Double:  OR  + ADD + MUL
#./vector/slr/slr_int_debug 10      # Vectorial Integer:    SLR + ADD + MUL
#./vector/slr/slr_fpd_debug 10      # Vectorial FP Double:  SLR + ADD + MUL
#################################################################################
