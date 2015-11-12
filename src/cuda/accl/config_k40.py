#!/usr/bin/python

import os
import sys
import ConfigParser

print "Generating accl for CUDA K40"

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

data_path=installDir+"data/accl"
bin_path=installDir+"bin"
src_accl = installDir+"src/cuda/accl"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

os.system("cd "+src_accl)

os.system("sudo ./accl_generate 2 2 "+data_path+"/2Frames.pgm gold_2_2")
os.system("sudo ./accl_generate 5 1 "+data_path+"/5Frames.pgm gold_5_1")
os.system("sudo ./accl_generate 5 5 "+data_path+"/5Frames.pgm gold_5_5")
os.system("sudo ./accl_generate 7 1 "+data_path+"/7Frames.pgm gold_7_1")
os.system("sudo ./accl_generate 7 4 "+data_path+"/7Frames.pgm gold_7_4")
os.system("sudo ./accl_generate 7 7 "+data_path+"/7Frames.pgm gold_7_7")
os.system("sudo chmod 777 accl accl_generate");
os.system("mv gold* "+data_path);
os.system("mv ./accl ./accl_generate "+bin_path)

fp = open(installDir+"scripts/how_to_run_accl_cuda_K40", 'w')
print >>fp, "sudo "+bin_path+"/accl 2 2 "+data_path+"/2Frames.pgm "+data_path+"/gold_2_2 10000000"
print >>fp, "sudo "+bin_path+"/accl 5 1 "+data_path+"/5Frames.pgm "+data_path+"/gold_5_1 10000000"
print >>fp, "sudo "+bin_path+"/accl 5 5 "+data_path+"/5Frames.pgm "+data_path+"/gold_5_5 10000000"
print >>fp, "sudo "+bin_path+"/accl 7 1 "+data_path+"/7Frames.pgm "+data_path+"/gold_7_1 10000000"
print >>fp, "sudo "+bin_path+"/accl 7 4 "+data_path+"/7Frames.pgm "+data_path+"/gold_7_4 10000000"
print >>fp, "sudo "+bin_path+"/accl 7 7 "+data_path+"/7Frames.pgm "+data_path+"/gold_7_7 10000000"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_accl_cuda\n"

sys.exit(0)
