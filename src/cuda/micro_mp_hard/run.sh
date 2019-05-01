#!/bin/bash


for p in half single double;
do
    for h in none dmr dmrmixed;
    do
        for m in add mul fma;
        do
          echo ./cuda_micro_mp_hardening -verbose -iterations=10 -precision=$p -redundancy=$h -inst=$m
        done
    done
done
