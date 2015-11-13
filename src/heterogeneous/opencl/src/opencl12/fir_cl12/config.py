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
src_fir = installDir+"src/heterogeneous/opencl/src/opencl12/fir_cl12/bin/x86_64/Release"

os.system("sudo mkdir "+src_fir+"/input");
os.system("sudo mkdir "+src_fir+"/output");

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

#os.system("cd "+src_fir);
os.system("cd "+src_fir+"; sudo ./fir_cl12 -b 2048 -d 2048 -g; pwd");
os.system("cd "+src_fir+"; sudo ./fir_cl12 -b 4096 -d 4096 -g");
os.system("cd "+src_fir+"; sudo ./fir_cl12 -b 8192 -d 8192 -g");
os.system("sudo chmod 777 input output input/* output/* ");
os.system("mv input output "+data_path);
os.system("mv ./fir_cl12 "+bin_path);
os.system("cp run_* "+bin_path);

fp = open(installDir+"scripts/how_to_run_fir_cl12", 'w')
print >>fp, "cd "+bin_path+"; bash "+bin_path+"/run_fir_2048.sh"
print >>fp, "cd "+bin_path+"; bash "+bin_path+"/run_fir_4096.sh"
print >>fp, "cd "+bin_path+"; bash "+bin_path+"/run_fir_8192.sh"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_fir_cl12\n"

sys.exit(0)
