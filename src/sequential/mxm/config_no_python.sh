#!/bin/sh

RADIATION_BENCHMARKS="/home/carol/radiation-benchmarks"
#RADIATION_BENCHMARKS="/mnt/4E0AEF320AEF15AD/radiation-benchmarks"
SIZE=1024
ARCH=x86
 
make ARCH=x86 LOGS=1 MATRIXSIZE=$SIZE generate

mkdir -p $RADIATION_BENCHMARKS/bin
mkdir -p $RADIATION_BENCHMARKS/data/mxm

mv -f matmul $RADIATION_BENCHMARKS/bin/
mv matmul_input_${SIZE}.txt  matmul_gold_${SIZE}.txt $RADIATION_BENCHMARKS/data/mxm/

str='[
{
    "killcmd": "killall -9 matmul", "exec": '${RADIATION_BENCHMARKS}'/bin/matmul  '${RADIATION_BENCHMARKS}'/data/mxm/matmul_input_'${SIZE}'.txt '${RADIATION_BENCHMARKS}'/data/mxm/matmul_gold_'${SIZE}'.txt 0 '${SIZE}'"
}
]'

echo $str > "${RADIATION_BENCHMARKS}/scripts/json_files/matmul.json"
