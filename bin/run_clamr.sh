#!/bin/bash
#$1 == big execute CLAMR with big processors
#$1 == little processor

big="big";
lit="little";

function right_par(){
	echo "Please enter with the right parameters <big | little> <threads>";
	return;
}

if [ `id -u` != 0 ];
then 
	echo "Please run as root"
	return;
fi

if (( "$#" == 1 ));
then
	#if it's little
	if [ $1 == $lit ];
	then
                echo 0 > /sys/devices/system/cpu/cpuquiet/tegra_cpuquiet/enable
                echo 0 > /sys/devices/system/cpu/cpu1/online
                echo 0 > /sys/devices/system/cpu/cpu2/online
                echo 0 > /sys/devices/system/cpu/cpu3/online
                echo LP > /sys/kernel/cluster/active
                #configure how many threads will be executed
	        
                
	fi
	
	if [ $1 == $big ];
	then
                echo 0 > /sys/devices/system/cpu/cpuquiet/tegra_cpuquiet/enable
                echo 1 > /sys/devices/system/cpu/cpu0/online
                echo 1 > /sys/devices/system/cpu/cpu1/online
                echo 1 > /sys/devices/system/cpu/cpu2/online
                echo 1 > /sys/devices/system/cpu/cpu3/online
                echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
                #configure how many threads will be executed
	        #export OMP_NUM_THREADS=8;
	fi
       # sudo export OMP_NUM_THREADS='$2';
else    
	right_par;
fi
exit;
