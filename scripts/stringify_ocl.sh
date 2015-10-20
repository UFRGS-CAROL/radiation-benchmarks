#!/bin/bash

# Receives a opencl kernel file (my_kernel.cl) and tansform it to string, 
# to include with host source code, saving it to my_kernel.h

IN=$1
NAME=${IN%.cl}
OUT=$NAME.h
echo "const char *"$NAME"_ocl =" >$OUT
sed -e 's/\\/\\\\/g;s/"/\\"/g;s/^/"/;s/$/\\n"/' $IN >>$OUT
echo ";" >>$OUT 
