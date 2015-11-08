#!/usr/bin/python

import os
import sys
import ConfigParser


print "Gold generating"

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

data_path=installDir+"data/histogram_ori_gradients"
bin_path=installDir+"bin"
src_hog = installDir+"src/cuda/histogram_ori_gradients"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

os.system("cd "+src_hog)
#os.system("sudo ./gold_gen 2000 1024 "+str(deviceType)+" ../../../data/hotspot/temp_1024 ../../../data/hotspot/power_1024 GOLD_2000_1024")
#executar de qualquer lugar para testar
os.system("sudo ./gold_gen ../../../data/histogram_ori_gradients/1x_pedestrians.jpg --dst_data GOLD_1x.data --hit_threshold 0.9 --gr_threshold 1 --nlevels 100")
os.system("sudo ./gold_gen ../../../data/histogram_ori_gradients/4x_pedestrians.jpg --dst_data GOLD_4x.data --hit_threshold 0.9 --gr_threshold 1 --nlevels 100")
os.system("sudo ./gold_gen ../../../data/histogram_ori_gradients/9x_pedestrians.jpg --dst_data GOLD_9x.data --hit_threshold 0.9 --gr_threshold 1 --nlevels 100")
os.system("sudo chmod 777 GOLD_* ");
os.system("mv GOLD_* "+data_path);
#move all binaries to bin path
os.system("mv ./hog ./gold_gen "+bin_path)

fp = open(installDir+"scripts/how_to_run_hog_cuda", 'w')
print >>fp, "sudo "+bin_path+"/hog "+data_path+"/1x_pedestrians.jpg --dst_data "+data_path+"/GOLD_1x.data --iterations 10000000"
print >>fp, "sudo "+bin_path+"/hog "+data_path+"/4x_pedestrians.jpg --dst_data "+data_path+"/GOLD_4x.data --iterations 10000000"
print >>fp, "sudo "+bin_path+"/hog "+data_path+"/9x_pedestrians.jpg --dst_data "+data_path+"/GOLD_9x.data --iterations 10000000"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_hog_cuda\n"

sys.exit(0)
