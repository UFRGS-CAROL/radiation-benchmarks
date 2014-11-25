source /opt/intel/composerxe/bin/compilervars.sh intel64;
export OFFLOAD_REPORT=1
file=output
./slr_int 0 >> $file 2>&1
./slr_fpd 0 >> $file 2>&1

