source /opt/intel/composerxe/bin/compilervars.sh intel64;
export OFFLOAD_REPORT=1
file=output
./or_int 0 >> $file 2>&1
./or_fpd 0 >> $file 2>&1

