#!/usr/bin/python

import os
import sys
import ConfigParser

print "Generating darknet for CUDA"

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

data_path=installDir+"data/darknet"
bin_path=installDir+"bin"
src_daknet = installDir+"src/cuda/"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

"""
 { { "execution_type",
            required_argument, NULL, 'e' }, //yolo/cifar/imagenet...
            { "execution_model", required_argument, NULL, 'm' }, //test/valid...
            { "config_file", required_argument, NULL, 'c' }, //<yolo, imagenet..>.cfg
            { "weights", required_argument, NULL, 'w' }, //<yolo, imagenet..>weights
//          { "input_data_path",    required_argument, NULL, 'i' },
            { "iterations", required_argument, NULL, 'n' }, //log data iterations
            { "generate", required_argument, NULL, 'g' }, //generate gold
            { "img_list_path", required_argument, NULL, 'l' }, //data path list input
            { "base_result_out", required_argument, NULL, 'b' }, //result output
            { "gpu_index", required_argument, NULL, 'x' }, //gpu index
            { "gold_input", required_argument, NULL, 'd'},
            { NULL, 0, NULL, 0 } };


test:darknet
./darknet -e yolo -m valid -c cfg/yolo.cfg -w yolo.weights -n 4 -d gold/gold_voc2012.test -l voc.2012.debug.txt -b gold/comp4_det_test_ -x -1

generate:darknet
./darknet -e yolo -m valid -c cfg/yolo.cfg -w yolo.weights -n 1 -g gold/gold_voc2012.test -l voc.2012.debug.txt -b gold/comp4_det_test_ -x 0
"""

execution_model = 'yolo'
config_file = data_path + 'yolo.cfg'
weights = data_path + 'yolo.weights'
gold_

os.system("cd "+src_hotspot)
os.system("sudo ./hotspot -size=1024 -generate -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file=gold_1024_1000 -sim_time=1000 -iterations=1")
os.system("sudo ./hotspot -size=1024 -generate -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file=gold_1024_10000 -sim_time=10000 -iterations=1")

os.system("sudo chmod 777 gold_* ");
os.system("mv gold_* "+data_path);
os.system("mv ./hotspot "+bin_path)

fp = open(installDir+"scripts/how_to_run_hotspot_cuda_K40", 'w')
print >>fp, "sudo "+bin_path+"/hotspot -size=1024 -sim_time=1000 -streams=1 -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file="+data_path+"/gold_1024_1000 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/hotspot -size=1024 -sim_time=10000 -streams=1 -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file="+data_path+"/gold_1024_10000 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/hotspot -size=1024 -sim_time=1000 -streams=4 -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file="+data_path+"/gold_1024_1000 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/hotspot -size=1024 -sim_time=10000 -streams=4 -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file="+data_path+"/gold_1024_10000 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/hotspot -size=1024 -sim_time=1000 -streams=8 -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file="+data_path+"/gold_1024_1000 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/hotspot -size=1024 -sim_time=10000 -streams=8 -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file="+data_path+"/gold_1024_10000 -iterations=10000000"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_hotspot_cuda\n"

sys.exit(0)
