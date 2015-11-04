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

data_path=installDir+"data/lavamd"
bin_path=installDir+"bin"
src_lavamd = installDir+"src/opencl/lavaMD"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777)
	os.chmod(data_path, 0777)

os.system("cd "+src_lavamd)
os.system("sudo ./lavamd_genGold 13 "+str(deviceType)+" 64")
os.system("sudo ./lavamd_genGold 15 "+str(deviceType)+" 64")
os.system("sudo ./lavamd_genGold 19 "+str(deviceType)+" 64")
os.system("sudo ./lavamd_genGold 23 "+str(deviceType)+" 64")
os.system("sudo chmod 777 input_* output_* ")
os.system("mv input_* output_* "+data_path)
os.system("mv ./lavamd_genGold ./lavamd_nologs_timing ./lavamd_err_inj ./lavamd "+bin_path)


fp = open(installDir+"scripts/how_to_run_lavamd_ocl", 'w')
print >>fp, bin_path+"/lavamd 13 "+str(deviceType)+" 64 "+data_path+"/input_distance_13_64 "+data_path+"/input_charges_13_64 "+data_path+"/output_gold_13_64 10000000"
print >>fp, bin_path+"/lavamd 15 "+str(deviceType)+" 64 "+data_path+"/input_distance_15_64 "+data_path+"/input_charges_15_64 "+data_path+"/output_gold_15_64 10000000"
print >>fp, bin_path+"/lavamd 19 "+str(deviceType)+" 64 "+data_path+"/input_distance_19_64 "+data_path+"/input_charges_19_64 "+data_path+"/output_gold_19_64 10000000"
print >>fp, bin_path+"/lavamd 23 "+str(deviceType)+" 64 "+data_path+"/input_distance_23_64 "+data_path+"/input_charges_23_64 "+data_path+"/output_gold_23_64 10000000"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_lavamd_ocl\n"

sys.exit(0)

