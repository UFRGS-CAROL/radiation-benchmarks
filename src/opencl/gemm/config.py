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

data_path=installDir+"data/gemm"
bin_path=installDir+"bin"
src_gemm = installDir+"src/opencl/gemm"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

os.system("cd "+src_gemm)
os.system("./genInputMatrices 8192");
os.system("sudo ./gemm_genGold 2048 "+str(deviceType)+" Double_A_8192.matrix Double_B_8192.matrix GOLD_2048")
os.system("sudo ./gemm_genGold 4096 "+str(deviceType)+" Double_A_8192.matrix Double_B_8192.matrix GOLD_4096")
os.system("sudo ./gemm_genGold 8192 "+str(deviceType)+" Double_A_8192.matrix Double_B_8192.matrix GOLD_8192")
os.system("sudo chmod 777 GOLD_* ");
os.system("mv GOLD_* Double_* "+data_path);
os.system("mv ./genInputMatrices ./gemm_genGold ./gemm_nologs_timing ./gemm_err_inj ./gemm "+bin_path)

fp = open(installDir+"scripts/how_to_run_gemm_ocl", 'w')
print >>fp, bin_path+"/gemm 2048 "+str(deviceType)+" "+data_path+"/Double_A_8192.matrix "+data_path+"/Double_B_8192.matrix "+data_path+"/GOLD_2048 10000000"
print >>fp, bin_path+"/gemm 4096 "+str(deviceType)+" "+data_path+"/Double_A_8192.matrix "+data_path+"/Double_B_8192.matrix "+data_path+"/GOLD_4096 10000000"
print >>fp, bin_path+"/gemm 8192 "+str(deviceType)+" "+data_path+"/Double_A_8192.matrix "+data_path+"/Double_B_8192.matrix "+data_path+"/GOLD_8192 10000000"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_gemm_ocl\n"

sys.exit(0)

