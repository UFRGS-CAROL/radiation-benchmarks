#!/bin/bash
#$1 == big execute CLAMR with big processors
#$1 == little processor
#$2 is the numbers of big processors

n_para_lit=1
n_para_big=2

big="big";
lit="little";

function right_par(){
	echo "Please enter with the right parameters <big | little>";
	exit;
}

if [ `id -u` != 0 ];
then 
	echo "Please run as root"
	exit;
fi

if (( "$#" == $n_para_lit || "$#" == $n_para_big ));
then
	#if it's little
	if [ $1 == $lit ];
	then
                echo 0 > /sys/devices/system/cpu/cpuquiet/tegra_cpuquiet/enable
                echo 0 > /sys/devices/system/cpu/cpu1/online
                echo 0 > /sys/devices/system/cpu/cpu2/online
                echo 0 > /sys/devices/system/cpu/cpu3/online
                echo LP > /sys/kernel/cluster/active
	fi
	
	if [ $1 == $big ];
	then
                echo 0 > /sys/devices/system/cpu/cpuquiet/tegra_cpuquiet/enable
                echo 1 > /sys/devices/system/cpu/cpu0/online
                echo 1 > /sys/devices/system/cpu/cpu1/online
                echo 1 > /sys/devices/system/cpu/cpu2/online
                echo 1 > /sys/devices/system/cpu/cpu3/online
                echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
	fi

else
	right_par;
fi
export OMP_NUM_THREADS=8
exit;
