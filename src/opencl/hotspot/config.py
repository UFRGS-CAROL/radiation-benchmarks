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

data_path=installDir+"data/hotspot"
bin_path=installDir+"bin"
src_hotspot = installDir+"src/opencl/hotspot"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

os.system("cd "+src_hotspot)
os.system("sudo ./hotspot_genGold 2000 1024 "+str(deviceType)+" ../../../data/hotspot/temp_1024 ../../../data/hotspot/power_1024 GOLD_2000_1024")
os.system("sudo chmod 777 GOLD_* ");
os.system("mv GOLD_* "+data_path);
os.system("mv ./hotspot_genGold ./hotspot_nologs_timing ./hotspot_err_inj ./hotspot "+bin_path)

fp = open(installDir+"scripts/hotspot_ocl_how_to_run", 'w')
print >>fp, "sudo "+bin_path+"/hotspot 2000 1024 "+str(deviceType)+" "+data_path+"/temp_1024 "+data_path+"/power_1024 "+data_path+"/GOLD_2000_1024 10000000"

print "\nConfiguring done, to run check file: "+installDir+"scripts/hotspot_ocl_how_to_run\n"

sys.exit(0)

