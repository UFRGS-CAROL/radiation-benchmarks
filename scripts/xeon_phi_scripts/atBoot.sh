#!/bin/bash

#echo "where's my car?" > /tmp/duuuuude
#whoami >> /tmp/duuuuude

#micctrl --status mic0 > /tmp/aaaaaa

# make sure xeon phi goes to online state
micctrl --boot

# wait for mic to be online and able to receive ssh
/home/carol/waitForMic.sh 

#echo "time before ntpdate" >> /micNfs/timelog
#date >> /micNfs/timelog
#ntpdate -u 192.168.1.5
#echo "time after ntpdate" >> /micNfs/timelog
#date >> /micNfs/timelog

# disable ECC report, dump, cehckpoint-roolback, crash etc.
ssh mic0 /micNfs/msr/scriptMCA.sh 

#/home/carol/radiation-benchmarks/scripts/killtestFile.py /home/carol/radiation-benchmarks/scripts/xeon_phi_commands/xeon_phi_jsons &

exit 0
