source /opt/intel/composerxe/bin/compilervars.sh intel64;
export OFFLOAD_REPORT=1
file=output
./and_int 0 >> $file 2>&1
./and_fpd 0 >> $file 2>&1

