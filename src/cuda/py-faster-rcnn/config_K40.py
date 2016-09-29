#!/usr/bin/python

import os
import sys
import ConfigParser

print "Generating py-faster for CUDA"

confFile = '/etc/radiation-benchmarks.conf'
try:
	config = ConfigParser.RawConfigParser()
	config.read(confFile)

	installDir = config.get('DEFAULT', 'installdir')+"/"
	varDir =  config.get('DEFAULT', 'vardir')+"/"
	logDir =  config.get('DEFAULT', 'logdir')+"/"
	tmpDir =  config.get('DEFAULT', 'tmpdir')+"/"

except IOError as e:
	print >> sys.stderr, "Configuration setup error: "+str(e)
	sys.exit(1)

data_path=installDir+"data/py_faster_rcnn"
bin_path=installDir+"bin"
src_py_faster = installDir + "src/cuda/py_faster_rcnn"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);


iterations = '10000' #it is not so much, since each dataset have at least 10k of images
base_caltech_out = src_py_faster + '/gold/comp_caltech'
base_voc_out = src_py_faster + '/gold/comp_voc'


img_datasets = str.replace(data_path,'/py_faster_rcnn', '') + "/networks_img_list"
#all inputs
caltech_gold_FULL = data_path + '/gold_caltech_full.test'
caltech_img_list_FULL = img_datasets + 'caltech/K40/caltech.pedestrians.FULL.txt'
cal_FULL_str = caltech_gold_FULL + " -l " + caltech_img_list_FULL

#half of inputs
caltech_gold_HALF = data_path + '/gold_caltech_full.test'
caltech_img_list_HALF = img_datasets + '/caltech/K40/caltech.pedestrians.HALF.txt'
cal_HALF_str = caltech_gold_HALF + " -l " + caltech_img_list_HALF

#VOC
#all inputs
voc_gold_FULL = data_path + '/gold_voc_full.test'
voc_img_list_FULL = img_datasets + '/voc/K40/voc.2012.FULL.txt'
voc_FULL_str = voc_gold_FULL + " -l " + voc_img_list_FULL

#half of inputs
voc_gold_HALF = data_path + '/gold_voc_full.test'
voc_img_list_HALF = img_datasets + '/voc/K40/voc.2012.HALF.txt'
voc_HALF_str = voc_gold_HALF + " -l " + voc_img_list_HALF

#./py_faster_rcnn.py --gld test.test --iml /home/carol/radiation-benchmarks/data/networks_img_list/caltech/K40/caltech.pedestrians.DEBUG.txt --log daniel_logs
# ./py_faster_rcnn.py --gen test.test --iml /home/carol/radiation-benchmarks/data/networks_img_list/caltech/K40/caltech.pedestrians.DEBUG.txt
#voc
vc_half_gen =  "sudo ./tools/py_faster_rcnn.py --gen " + voc_gold_HALF + " --iml " +  voc_img_list_HALF

vc_full_gen = "sudo ./tools/py_faster_rcnn.py --gen " + voc_gold_FULL + " --iml " + voc_img_list_FULL

#caltech
cl_half_gen = "sudo ./tools/py_faster_rcnn.py --gen " + caltech_gold_HALF + " --iml " + caltech_img_list_HALF

cl_full_gen = "sudo ./tools/py_faster_rcnn.py --gen " + caltech_gold_FULL + " --iml " + caltech_img_list_FULL


#voc
vc_half_ex = "sudo "+ src_py_faster+ "/tools/py_faster_rcnn.py --gld " + voc_gold_HALF + " --iml " +  voc_img_list_HALF + " --log daniel_logs"

vc_full_ex = "sudo "+ src_py_faster+ "/tools/py_faster_rcnn.py --gld " + voc_gold_FULL + " --iml " + voc_img_list_FULL + " --log daniel_logs"


#caltech
cl_half_ex = "sudo "+ src_py_faster+ "/tools/py_faster_rcnn.py --gld " + caltech_gold_HALF + " --iml " + caltech_img_list_HALF + " --log daniel_logs"

cl_full_ex = "sudo "+ src_py_faster+ "/tools/py_faster_rcnn.py --gld " + caltech_gold_FULL + " --iml " + caltech_img_list_FULL + " --log daniel_logs"

os.system("cd " + src_py_faster)

# os.system("make clean")
# os.system("make -j 4 GPU=1")

os.system(vc_half_gen)
os.system(vc_full_gen)
os.system(cl_half_gen)
os.system(cl_full_gen)

# os.system("make clean")
# os.system("make -C ../../include/")
# os.system("make -j 4 GPU=1 LOGS=1")

# os.system("sudo chmod 777 gold_* ");
# os.system("mv gold_* "+data_path);
# os.system("mv ./ "+bin_path)


fp = open(installDir+"scripts/how_to_run_py_faster_rcnn_cuda_K40", 'w')

print >>fp, "[\""+vc_half_ex+"\", 1, \"py_faster_rcnn.py\"],"
print >>fp, "[\""+vc_full_ex+"\", 1, \"py_faster_rcnn.py\"],"
print >>fp, "[\""+cl_half_ex+"\", 1, \"py_faster_rcnn.py\"],"
print >>fp, "[\""+cl_full_ex+"\", 1, \"py_faster_rcnn.py\"],"


print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_py_faster_rcnn_cuda_K40\n"

sys.exit(0)
