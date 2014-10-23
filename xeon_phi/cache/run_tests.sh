source /opt/intel/composerxe/bin/compilervars.sh intel64;
export OFFLOAD_REPORT=1
file=output
#/// 8 em 8
load_ind_00008kb;./memory_load_ind 2048 128 >> $file
load_ind_00016kb;./memory_load_ind 1024 256 >> $file
load_ind_00024kb;./memory_load_ind 683  384 >> $file
load_ind_00032kb;./memory_load_ind 512  512 >> $file
load_ind_00048kb;./memory_load_ind 341  768 >> $file
load_ind_00056kb;./memory_load_ind 293  896 >> $file
load_ind_00064kb;./memory_load_ind 256  1024 >> $file
#/// 64 em 64                       
load_ind_00128kb;./memory_load_ind 128 2048 >> $file
load_ind_00192kb;./memory_load_ind 85  3072 >> $file
load_ind_00256kb;./memory_load_ind 64  4096 >> $file
load_ind_00320kb;./memory_load_ind 51  5120 >> $file
load_ind_00384kb;./memory_load_ind 43  6144 >> $file
load_ind_00448kb;./memory_load_ind 37  7168 >> $file
load_ind_00512kb;./memory_load_ind 32  8192 >> $file
#/// 256 em 256
load_ind_00768kb;./memory_load_ind 21 12288 >> $file
load_ind_01024kb;./memory_load_ind 16 16384 >> $file
load_ind_01280kb;./memory_load_ind 13 20480 >> $file
load_ind_01536kb;./memory_load_ind 11 24576 >> $file
load_ind_01792kb;./memory_load_ind 9  28672 >> $file
load_ind_02048kb;./memory_load_ind 8  32768 >> $file
