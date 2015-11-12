#!/bin/bash
#$1 == big execute CLAMR with big processors
#$1 == little processor

big="big";
lit="little";

function right_par(){
	echo "Please enter with the right parameters <big | little>";
	return;
}

if [ `id -u` != 0 ];
then 
	echo "Please run as root"
	return;
fi

if (( "$#" == 2 ));
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
	        export OMP_NUM_THREADS=2;
                
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
	        export OMP_NUM_THREADS=8;
	fi
        
        #execute CLAMR with -n 256 -t 1500 -g 100 -G data -j md5files
        ./clamr_openmponly -n 256 -t 1500 -g 100 -G data -j md5files;
        	
else    
	right_par;
fi

return;
