#!/bin/bash

PASS=$(cat pass)

sshpass -p $PASS ssh root@mic0 ${1}
