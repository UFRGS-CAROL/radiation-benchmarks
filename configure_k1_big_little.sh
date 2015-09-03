#!/bin/bash
#$1 == big execute CLAMR with big processors
#$1 == little processor
#$2 is the numbers of big processors

n_para_lit=1
n_para_big=2

big="big";
lit="little";

function right_par(){
	echo "Please enter with the right parameters <big | little> <1..4 if it's big>";
	exit 1;
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
		echo 0 > /sys/devices/system/cpu/cpu0/online
		echo 0 > /sys/devices/system/cpu/cpu1/online
		echo 0 > /sys/devices/system/cpu/cpu2/online	
		echo 0 > /sys/devices/system/cpu/cpu3/online
		echo LP > /sys/kernel/cluster/active
	fi
	
	if [ $1 == $big ];
	then
		if (( $# < $n_para_big ));
		then
			right_par;
		fi
		
		if (( $2 <= 0 || $2 > 4 ));
		then
			right_par;
		else
			#disable power save CPU
			echo 0 > /sys/devices/system/cpu/cpuquiet/tegra_cpuquiet/enable;
			#how many big cpus are enable
			for (( i=0; $i < $2; i++ ))
			do
				echo 1 > "/sys/devices/system/cpu/cpu$i/online"
			done
			
			for (( i=$2; $i < 4; i++ )) 
			do
				echo 0 > "/sys/devices/system/cpu/cpu$i/online"
			done
			echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
		fi
	fi

else
	right_par;
fi
exit;
