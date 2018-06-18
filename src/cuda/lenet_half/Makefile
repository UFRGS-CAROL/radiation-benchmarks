PRECISION=half

ENV= PYTHONPATH=/home/carol/radiation-benchmarks/src/cuda/lenet_$(PRECISION)/caffe/python:/home/carol/radiation-benchmarks/src/include/log_helper_swig_wraper:$PYTHONPATH 

ENV+= LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

PYTHONFILE=/home/carol/radiation-benchmarks/src/cuda/lenet_$(PRECISION)/lenet_$(PRECISION).py 
ITERATIONS=1
PROTOTXT= /home/carol/radiation-benchmarks/src/cuda/lenet_$(PRECISION)/caffe/examples/mnist/lenet.prototxt 
LENETMODEL=/home/carol/radiation-benchmarks/src/cuda/lenet_$(PRECISION)/caffe/examples/mnist/lenet_iter_10000.caffemodel 
LMDB=/home/carol/radiation-benchmarks/src/cuda/lenet_$(PRECISION)/caffe/examples/mnist/mnist_test_lmdb/
GOLD=/home/carol/radiation-benchmarks/data/lenet/gold_lenet_$(PRECISION)_10k.gold


generate:
	sudo $(ENV) $(PYTHONFILE)  --ite $(ITERATIONS) --testmode 0  --prototxt $(PROTOTXT)  --lenet_model $(LENETMODEL) --lmdb  $(LMDB) --gold $(GOLD)

test:
	sudo $(ENV) $(PYTHONFILE)  --ite $(ITERATIONS) --testmode 1  --prototxt $(PROTOTXT)  --lenet_model $(LENETMODEL) --lmdb  $(LMDB) --gold $(GOLD)

train:
	sudo $(ENV) $(PYTHONFILE)  --ite $(ITERATIONS) --testmode 3 
