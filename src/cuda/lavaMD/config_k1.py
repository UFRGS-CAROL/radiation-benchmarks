#!/usr/bin/python

import os
import sys
import ConfigParser

print "Generating for lava CUDA"

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

data_path=installDir+"data/lava"
bin_path=installDir+"bin"
src_lava = installDir+"src/cuda/kmeans"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

os.system("cd "+src_lava)
os.system("sudo ./lava -boxes=6 -generate -output_gold=GOLD_6 -iterations=1 -streams=1")
os.system("sudo ./lava -boxes=7 -generate -output_gold=GOLD_7 -iterations=1 -streams=1")
os.system("sudo ./lava -boxes=5 -generate -output_gold=GOLD_5 -iterations=1 -streams=1")


os.system("sudo chmod 777 GOLD_* ");
os.system("mv GOLD_* input* "+data_path);
os.system("mv ./lava "+bin_path)

fp = open(installDir+"scripts/how_to_run_lava_cuda", 'w')
print >>fp, "sudo "+bin_path+"/lava -boxes=6 -input_distances="+data_path+"/input_distances_6 -input_charges="+data_path+"/input_charges_6 -output_gold="+data_path+"/GOLD_6 -iterations=10000000 -streams=1"
print >>fp, "sudo "+bin_path+"/lava -boxes=7 -input_distances="+data_path+"/input_distances_7 -input_charges="+data_path+"/input_charges_7 -output_gold="+data_path+"/GOLD_7 -iterations=10000000 -streams=1"
print >>fp, "sudo "+bin_path+"/lava -boxes=5 -input_distances="+data_path+"/input_distances_5 -input_charges="+data_path+"/input_charges_5 -output_gold="+data_path+"/GOLD_5 -iterations=10000000 -streams=1"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_lava_cuda\n"

sys.exit(0)

