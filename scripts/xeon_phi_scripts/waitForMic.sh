#!/bin/bash

DEBUG=/tmp/aaa

echo "Gambi" > $DEBUG

while true
do
	sleep 1
	echo "Waiting for mic..." >> $DEBUG
	micctrl --status mic0 | grep "online" > /dev/null

	if [ $? -eq 0 ]
	then
		echo "mic online!" >> $DEBUG
		break
	fi
done

