#!/usr/bin/python

import os
import sys
import ConfigParser

print "Generating lava for CUDA"

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
src_lava = installDir+"src/cuda/lava"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

os.system("cd "+src_lava)
os.system("sudo ./lava -boxes=7 -generate -output_gold=gold_7 -iterations=1 -streams=1")
os.system("sudo ./lava -boxes=9 -generate -output_gold=gold_9 -iterations=1 -streams=1")
os.system("sudo ./lava -boxes=11 -generate -output_gold=gold_11 -iterations=1 -streams=1")

os.system("sudo chmod 777 gold_* ");
os.system("mv gold_* "+data_path);
os.system("mv ./lava "+bin_path)

fp = open(installDir+"scripts/how_to_run_lava_cuda_K40", 'w')
print >>fp, "sudo "+bin_path+"/lava -boxes=7 -input_distances="+data_path+"/input_distances_7 -input_charges="+data_path+"/input_charges_7 -output_gold="+data_path+"/gold_7 -iterations=10000000 -streams=1"
print >>fp, "sudo "+bin_path+"/lava -boxes=9 -input_distances="+data_path+"/input_distances_9 -input_charges="+data_path+"/input_charges_9 -output_gold="+data_path+"/gold_9 -iterations=10000000 -streams=1"
print >>fp, "sudo "+bin_path+"/lava -boxes=11 -input_distances="+data_path+"/input_distances_11 -input_charges="+data_path+"/input_charges_11 -output_gold="+data_path+"/gold_11 -iterations=10000000 -streams=1"
print >>fp, "sudo "+bin_path+"/lava -boxes=7 -input_distances="+data_path+"/input_distances_7 -input_charges="+data_path+"/input_charges_7 -output_gold="+data_path+"/gold_7 -iterations=10000000 -streams=8"
print >>fp, "sudo "+bin_path+"/lava -boxes=9 -input_distances="+data_path+"/input_distances_9 -input_charges="+data_path+"/input_charges_9 -output_gold="+data_path+"/gold_9 -iterations=10000000 -streams=8"
print >>fp, "sudo "+bin_path+"/lava -boxes=11 -input_distances="+data_path+"/input_distances_11 -input_charges="+data_path+"/input_charges_11 -output_gold="+data_path+"/gold_11 -iterations=10000000 -streams=8"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_lava_cuda\n"

sys.exit(0)
