#!/usr/bin/python

import os
import sys
import ConfigParser

print "Generating for mxm CUDA"

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

data_path=installDir+"data/mxm"
bin_path=installDir+"bin"
src_mxm = installDir+"src/cuda/mxm"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

os.system("cd "+src_mxm)
os.system("sudo ./generateMatrices 512 A_matrix_512 B_matrix_512 GOLD_matrix_512")
os.system("sudo ./generateMatrices 1024 A_matrix_1024 B_matrix_1024 GOLD_matrix_1024")
os.system("sudo ./generateMatrices 2048 A_matrix_2048 B_matrix_2048 GOLD_matrix_2048")

os.system("sudo chmod 777 GOLD_* ");
os.system("mv *matrix* "+data_path);
os.system("mv ./generateMatrices cudaMxM "+bin_path)

fp = open(installDir+"scripts/how_to_run_mxm_cuda", 'w')
print >>fp, "sudo "+bin_path+"/cudaMxM 512 "+data_path+"/A_matrix_512 "+data_path+"/B_matrix_512 "+data_path+"/GOLD_matrix_512 10000000"
print >>fp, "sudo "+bin_path+"/cudaMxM 1024 "+data_path+"/A_matrix_1024 "+data_path+"/B_matrix_1024 "+data_path+"/GOLD_matrix_1024 10000000"
print >>fp, "sudo "+bin_path+"/cudaMxM 2048 "+data_path+"/A_matrix_2048 "+data_path+"/B_matrix_2048 "+data_path+"/GOLD_matrix_2048 10000000"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_mxm_cuda\n"

sys.exit(0)
