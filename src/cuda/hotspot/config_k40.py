#!/usr/bin/python

import os
import sys
import ConfigParser

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
src_hotspot = installDir+"src/cuda/hotspot"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

os.system("cd "+src_hotspot)
os.system("sudo ./hotspot -size=1024 -generate -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file=gold_1024_1000 -sim_time=1000 -iterations=1")
os.system("sudo ./hotspot -size=1024 -generate -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file=gold_1024_10000 -sim_time=10000 -iterations=1")

os.system("sudo chmod 777 gold_* ");
os.system("mv gold_* "+data_path);
os.system("mv ./hotspot "+bin_path)

fp = open(installDir+"scripts/how_to_run_hotspot_cuda_K40", 'w')
print >>fp, "sudo "+bin_path+"/hotspot -size=1024 -sim_time=1000 -streams=1 -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file="+data_path+"/gold_1024_1000 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/hotspot -size=1024 -sim_time=10000 -streams=1 -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file="+data_path+"/gold_1024_10000 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/hotspot -size=1024 -sim_time=1000 -streams=8 -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file="+data_path+"/gold_1024_1000 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/hotspot -size=1024 -sim_time=10000 -streams=8 -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file="+data_path+"/gold_1024_10000 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/hotspot -size=1024 -sim_time=1000 -streams=16 -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file="+data_path+"/gold_1024_1000 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/hotspot -size=1024 -sim_time=10000 -streams=16 -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file="+data_path+"/gold_1024_10000 -iterations=10000000"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_hotspot_cuda\n"

sys.exit(0)
