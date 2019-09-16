#!/bin/bash

metrics=ipc,issued_ipc,inst_executed,flop_count_dp,flop_count_dp_add,flop_count_dp_fma,flop_count_dp_mul,flop_count_sp,flop_count_sp_add,flop_count_sp_fma,flop_count_sp_mul,flop_count_sp_special,flop_count_hp,flop_count_hp_add,flop_count_hp_mul,flop_count_hp_fma

for p in half single double;
do
    for h in none dmr dmrmixed;
    do
        for m in add mul fma;
        do
            out_file="${p}_${h}_${m}.csv"
            nvprof --metrics $metrics --csv ./cuda_micro_mp_hardening --verbose --iterations 10 --precision $p --redundancy $h --inst $m > nvprof_out.txt 2>$out_file
            sed -i "1,4d" "$out_file"
            cat $out_file
	    exit 0
        done
    done
done

rm nvprof_out.txt
