#!/usr/bin/python

import os
import sys
import ConfigParser

print "Generating gold files."

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

data_path=installDir+"bin/fir"
bin_path=installDir+"bin/fir"
src_fir = installDir+"src/heterogeneous/opencl/src/opencl20/fir_cl20/bin/x86_64/Release"

os.system("sudo mkdir "+src_fir+"/input");
os.system("sudo mkdir "+src_fir+"/output");

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

#os.system("cd "+src_fir+"; pwd");
os.system("cd "+src_fir+";sudo ./fir_cl20 -b 2048 -d 1024 -g");
os.system("cd "+src_fir+";sudo ./fir_cl20 -b 4096 -d 1024 -g");
os.system("cd "+src_fir+";sudo ./fir_cl20 -b 8192 -d 1024 -g");
os.system("cd "+src_fir+";sudo chmod 777 input output input/* output/* ");
os.system("cd "+src_fir+";mv input output "+data_path);
os.system("cd "+src_fir+";mv ./fir_cl20 ./fir_cl20_kernel.cl "+bin_path);

fp = open(installDir+"scripts/how_to_run_fir_opencl", 'w')
print >>fp, "cd "+bin_path+"; sudo "+bin_path+"/fir_cl20 -b 2048 -d 1024"
print >>fp, "cd "+bin_path+"; sudo "+bin_path+"/fir_cl20 -b 4096 -d 1024"
print >>fp, "cd "+bin_path+"; sudo "+bin_path+"/fir_cl20 -b 8192 -d 1024"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_fir_opencl\n"

sys.exit(0)
