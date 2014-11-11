source /opt/intel/composerxe/bin/compilervars.sh intel64;
export OFFLOAD_REPORT=1
file=output
echo "28KB * cores => L1=32KB"
./cache 100000 $((28672 * 56)) 2 >> $file
echo "224KB * cores => L2=512KB"
./cache 10000 $((229376 * 56)) 2 >> $file
echo "448KB * cores => L2=512KB"
./cache 5000 $((458752 * 56)) 2 >> $file
