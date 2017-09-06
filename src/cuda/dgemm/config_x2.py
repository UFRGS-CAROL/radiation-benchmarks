#!/usr/bin/python

import os
import sys
import ConfigParser

print "Generating dgemm for CUDA Tegra X2"

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

data_path=installDir+"data/dgemm"
bin_path=installDir+"bin"
src_dgemm = installDir+"src/cuda/dgemm"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);
if not os.path.isdir(bin_path):
	os.mkdir(bin_path, 0777);
	os.chmod(bin_path, 0777);

os.system("cd "+src_dgemm)

os.system("sudo ./generateMatricesDouble -size=1024 -input_a=dgemmA_8192.matrix -input_b=dgemmB_8192.matrix -gold=dgemmGOLD_1024.matrix")
os.system("sudo ./generateMatricesDouble -size=2048 -input_a=dgemmA_8192.matrix -input_b=dgemmB_8192.matrix -gold=dgemmGOLD_2048.matrix")
os.system("sudo ./generateMatricesDouble -size=4096 -input_a=dgemmA_8192.matrix -input_b=dgemmB_8192.matrix -gold=dgemmGOLD_4096.matrix")
os.system("sudo chmod 777 dgemm*.matrix");
os.system("mv dgemm*.matrix "+data_path);
os.system("mv ./generateMatricesDouble ./cudaDGEMM "+bin_path)

fp = open(installDir+"scripts/how_to_run_dgemm_cuda_x2", 'w')
print >>fp, "sudo "+bin_path+"/cudaDGEMM -size=1024 -input_a="+data_path+"dgemmA_8192.matrix -input_b="+data_path+"/dgemmB_8192.matrix -gold="+data_path+"/dgemmGOLD_1024.matrix -iterations=10000000"
print >>fp, "sudo "+bin_path+"/cudaDGEMM -size=2048 -input_a="+data_path+"dgemmA_8192.matrix -input_b="+data_path+"/dgemmB_8192.matrix -gold="+data_path+"/dgemmGOLD_2048.matrix -iterations=10000000"
print >>fp, "sudo "+bin_path+"/cudaDGEMM -size=4096 -input_a="+data_path+"dgemmA_8192.matrix -input_b="+data_path+"/dgemmB_8192.matrix -gold="+data_path+"/dgemmGOLD_4096.matrix -iterations=10000000"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_dgemm_cuda_x2\n"

sys.exit(0)
