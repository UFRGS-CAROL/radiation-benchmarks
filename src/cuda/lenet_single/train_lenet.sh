sudo env PYTHONPATH=/home/carol/radiation-benchmarks/src/cuda/lenet_single/caffe/python:/home/carol/radiation-benchmarks/src/include/log_helper_swig_wraper:$PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} /home/carol/radiation-benchmarks/src/cuda/lenet_single/lenet_single.py   --ite 1  --testmode  3 

