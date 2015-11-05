#!/usr/bin/python

import os
import sys
import ConfigParser

print "Choose one device type to generate gold:"
print "\tDefault: 1"
print "\tCPU: 2"
print "\tGPU: 4"
print "\tACCELERATOR: 8"
print "\tALL: -1"

deviceType = raw_input("Enter device: ")
deviceType = int(deviceType)
if deviceType not in [1, 2, 4, 8, -1]:
	print "invalid device type: ",deviceType
	sys.exit(1)

print "Generating for device type: ",deviceType

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

data_path=installDir+"data/fft"
bin_path=installDir+"bin"
src_fft = installDir+"src/opencl/fft"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

os.system("cd "+src_fft)
os.system("sudo ./fft_genGold 4 "+str(deviceType))
os.system("sudo chmod 777 input_fft_* output_fft_* ");
os.system("mv input_fft_* output_fft_* "+data_path);
os.system("mv ./fft_genGold ./fft_nologs_timing ./fft_err_inj ./fft "+bin_path)

fp = open(installDir+"scripts/how_to_run_fft_ocl", 'w')
print >>fp, bin_path+"/fft 4 "+str(deviceType)+" 10 "+data_path+"/input_fft_size_4 "+data_path+"/output_fft_size_4 10000000"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_fft_ocl\n"

sys.exit(0)
