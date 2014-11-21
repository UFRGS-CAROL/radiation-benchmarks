source /opt/intel/composerxe/bin/compilervars.sh intel64;
export OFFLOAD_REPORT=1
file=output
echo "28KB * cores => L1=32KB"
./cache 0 $((28672 * 56)) >> $file
echo "480KB * cores => L2=512KB"
./cache 0 $((491520 * 56)) >> $file
