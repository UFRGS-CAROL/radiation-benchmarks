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
print "Generating gemmm for CUDA"

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
src_gemm = installDir+"src/cuda/gemm"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

os.system("cd "+src_gemm)

os.system("sudo ./generateMatrices -size=512 -input_a=Double_A_512.matrix -input_b=Double_B_512.matrix -gold=GOLD_512")
os.system("sudo ./generateMatrices -size=2048 -input_a=Double_A_2048.matrix -input_b=Double_B_2048.matrix -gold=GOLD_2048")
os.system("sudo ./generateMatrices -size=1024 -input_a=Double_A_1024.matrix -input_b=Double_B_1024.matrix -gold=GOLD_1024")
os.system("sudo chmod 777 GOLD_* ");
os.system("mv GOLD_* Double_* "+data_path);
os.system("mv ./generateMatrices ./cudaGEMM "+bin_path)

fp = open(installDir+"scripts/how_to_run_gemm_cuda", 'w')
print >>fp, "sudo "+bin_path+"/cudaGEMM -size=2048 -input_a="+data_path+"/Double_A_2048.matrix -input_b="+data_path+"/Double_B_2048.matrix -gold="+data_path+"/GOLD_2048 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/cudaGEMM -size=512 -input_a="+data_path+"/Double_A_512.matrix -input_b="+data_path+"/Double_B_512.matrix -gold="+data_path+"/GOLD_512 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/cudaGEMM -size=1024 -input_a="+data_path+"/Double_A_1024.matrix -input_b="+data_path+"/Double_B_1024.matrix -gold="+data_path+"/GOLD_1026 -iterations=10000000"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_gemm_cuda\n"

sys.exit(0)

