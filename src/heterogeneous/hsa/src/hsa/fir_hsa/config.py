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
src_fir = installDir+"src/heterogeneous/hsa/src/hsa/fir_hsa"

os.system("sudo mkdir "+src_fir+"/input");
os.system("sudo mkdir "+src_fir+"/output");

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

#os.system("cd "+src_fir);
os.system("cd "+src_fir+"; sudo ./fir_hsa -b 2048 -n 2048 -g");
os.system("cd "+src_fir+"; sudo ./fir_hsa -b 4096 -n 4096 -g");
os.system("cd "+src_fir+"; sudo ./fir_hsa -b 8192 -n 8192 -g");
os.system("sudo chmod 777 input output input/* output/* ");
os.system("mv input output "+data_path);
os.system("mv ./fir_hsa "+bin_path);

fp = open(installDir+"scripts/how_to_run_fir_hsa", 'w')
print >>fp, "cd "+bin_path+"; sudo "+bin_path+"/fir_hsa -b 2048 -n 2048"
print >>fp, "cd "+bin_path+"; sudo "+bin_path+"/fir_hsa -b 4096 -n 4096"
print >>fp, "cd "+bin_path+"; sudo "+bin_path+"/fir_hsa -b 8192 -n 8192"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_fir_hsa\n"

sys.exit(0)
