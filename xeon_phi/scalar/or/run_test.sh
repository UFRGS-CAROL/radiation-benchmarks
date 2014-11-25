source /opt/intel/composerxe/bin/compilervars.sh intel64;
export OFFLOAD_REPORT=1
file=output
./or 0 >> $file 2>&1
