#!/bin/bash

#echo "where's my car?" > /tmp/duuuuude
#whoami >> /tmp/duuuuude

#micctrl --status mic0 > /tmp/aaaaaa

###daqui pra baixo comentado por daniel
micctrl --boot

/home/carol/waitForMic.sh

cd /home/carol/msr
make

	#/home/carol/A_KillTest_SW &

exit 0
