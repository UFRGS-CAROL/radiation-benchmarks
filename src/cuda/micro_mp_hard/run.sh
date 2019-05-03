#!/bin/bash

metrics=ipc,issued_ipc,inst_executed

for p in half single double;
do
    for h in none dmr dmrmixed;
    do
        for m in add mul fma;
        do
            out_file="${p}_${h}_${m}.csv"
            nvprof --metrics $metrics --csv ./cuda_micro_mp_hardening --verbose --iterations 10 --precision $p --redundancy $h --inst $m > nvprof_out.txt 2>$out_file
            sed -i '1d;3d' $out_file
        done
    done
done

rm nvprof_out.txt 
