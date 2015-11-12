#!/usr/bin/python

import os
import sys
import ConfigParser

#works only on opencl
"""print "Choose one device type to generate gold:"
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
"""
print "Generating hotspot for CUDA"

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

data_path=installDir+"data/hotspot"
bin_path=installDir+"bin"
src_gemm = installDir+"src/cuda/hotspot"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

os.system("cd "+src_gemm)
#not finished yet
os.system("sudo ./hotspot -size=1024 -temp_file=../../../data/hotspot/temp_1024 -power_file=../../../data/hotspot/power_1024 -gold_file=GOLD_1024 -iterations=1")

os.system("sudo chmod 777 GOLD_* ");
os.system("mv GOLD_* "+data_path);
os.system("mv ./hotspot "+bin_path)

fp = open(installDir+"scripts/how_to_run_hotspot_cuda", 'w')
print >>fp, "sudo "+bin_path+"/hotspot -size=1024 -generate -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file="+data_path+"/GOLD_1024 -iterations=10000000"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_hotspot_cuda\n"

sys.exit(0)
