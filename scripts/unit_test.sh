#!/bin/bash

#echo "Testing unit Machine class"
python3 Machine.py

#echo "Testing unit RebootMachine"
python3 RebootMachine.py

cd server_package/
rm -r *.log logs
# Testing unit CopyLogs
python3 CopyLogs.py

cd -