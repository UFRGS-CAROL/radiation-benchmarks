#!/bin/bash

TARGETDIR=$(cat targetdir)
PASS=$(cat pass)

# add other binaries here if needed
for i in rdmsr wrmsr scriptMCA.sh
do
	sshpass -p $PASS scp ${i} root@mic0:${TARGETDIR}/
done
