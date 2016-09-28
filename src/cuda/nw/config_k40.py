#!/usr/bin/python

import os
import sys
import ConfigParser

print "Generating nw for CUDA K40"

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

data_path=installDir+"data/nw"
bin_path=installDir+"bin"
src_nw = installDir+"src/cuda/nw"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

os.system("cd "+src_nw)

os.system("sudo ./nw_generate 4096 5")
os.system("sudo ./nw_generate 8192 5")
os.system("sudo ./nw_generate 16384 5")
os.system("sudo chmod 777 nw nw_generate");
os.system("mv gold* "+data_path);
os.system("mv input* "+data_path);
os.system("mv ./nw ./nw_generate "+bin_path)

fp = open(installDir+"scripts/how_to_run_nw_cuda_K40", 'w')

print >>fp, "sudo "+bin_path+"/nw 4096 5 "+data_path+"/input_4096 "+data_path+"/gold_4096 1000000"
print >>fp, "sudo "+bin_path+"/nw 8192 5 "+data_path+"/input_8192 "+data_path+"/gold_8192 1000000"
print >>fp, "sudo "+bin_path+"/nw 16384 5 "+data_path+"/input_16384 "+data_path+"/gold_16384 1000000"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_nw_cuda\n"

sys.exit(0)
