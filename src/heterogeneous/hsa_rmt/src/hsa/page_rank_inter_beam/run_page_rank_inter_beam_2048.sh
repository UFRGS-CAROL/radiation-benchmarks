#!/bin/bash

while ((1))
do
        cd /home/carol/vinicius/radiation-benchmarks/bin/page_rank_inter_beam; sudo /home/carol/vinicius/radiation-benchmarks/bin/page_rank_intra_beam/page_rank_intra_beam -i input/csr_2048_10.txt
done
