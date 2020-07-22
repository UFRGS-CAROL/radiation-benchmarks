IMG_LIST=fault_injection.txt
RADPATH=/home/carol/radiation-benchmarks
BIN=$(RADPATH)/src/cuda/py-faster-rcnn
GOLD=$(RADPATH)/data/py_faster_rcnn/fault_injection.test

test:
	$(BIN)/tools/py_faster_rcnn.py --ite 1 --iml $(RADPATH)/data/networks_img_list/$(IMG_LIST) --gld $(GOLD) --log daniel_logs --net zf

generate:
	$(BIN)/tools/py_faster_rcnn.py  --ite 1 --iml $(RADPATH)/data/networks_img_list/$(IMG_LIST) --gen $(GOLD) --net zf


