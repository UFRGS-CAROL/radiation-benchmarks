#!/bin/bash
for ((i=1; i <= $#; i++)); do
	echo "size: ${!i} - FLOP - TIME"
	rm hmxm*_${!i}.matrix output.txt -f
	./generateMatricesHalf ${!i} > /dev/null
	./cudaHMXM ${!i} hmxmA_8192.matrix hmxmB_8192.matrix hmxmGOLD_${!i}.matrix 1 > output.txt 2>&1
	time= cat output.txt | grep "time: [0-9]\{1,\}.[0-9]\{1,\}" -o | cut -c7-
	nvprof -m flop_count_dp ./cudaHMXM ${!i} hmxmA_8192.matrix hmxmB_8192.matrix hmxmGOLD_${!i}.matrix 1 > output.txt 2>&1
	flop= cat output.txt | grep "Double Precisi  [0-9e+\.]\{1,\}" -o | cut -c17-
done
