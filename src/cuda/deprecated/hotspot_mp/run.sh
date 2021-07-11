#!/bin/bash

set -x

metrics=ipc,flop_count_dp,flop_count_dp_add,flop_count_dp_fma,flop_count_dp_mul,flop_count_sp,flop_count_sp_add,flop_count_sp_fma,flop_count_sp_mul,flop_count_sp_special,flop_count_hp,flop_count_hp_add,flop_count_hp_mul,flop_count_hp_fma

for p in half single double;
do
    for h in none dmr dmrmixed;
    do
        make PRECISION=$p REDUNDANCY=$h generate
 
        out_file="${p}_${h}.csv"
        nvprof --metrics $metrics --csv ./cuda_hotspot_mp -size=1024  -verbose -sim_time=1000 -input_temp=../../../data/hotspot/temp_1024 -input_power=../../../data/hotspot/power_1024 -gold_temp=./gold_1024 -streams=10 -iterations=5 -redundancy=$h -precision=$p > nvprof_out.txt 2>$out_file
        sed -i "1,4d" "$out_file"
        cat $out_file
    done
done

rm nvprof_out.txt
