#!/usr/bin/python

import os
import sys
import ConfigParser

print "Generating hhotspot for CUDA Tegra X2"

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

data_path=installDir+"data/hhotspot"
bin_path=installDir+"bin"
src_hhotspot = installDir+"src/cuda/hhotspot"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);
if not os.path.isdir(bin_path):
	os.mkdir(bin_path, 0777);
	os.chmod(bin_path, 0777);

os.system("cd "+src_hhotspot)
os.system("sudo ./hhotspot -size=1024 -generate -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file=gold_half_1024_1000 -sim_time=1000 -iterations=1")

os.system("sudo chmod 777 gold_* ");
os.system("mv gold_* "+data_path);
os.system("mv ./hhotspot "+bin_path)

fp = open(installDir+"scripts/how_to_run_hhotspot_cuda_x2", 'w')
print >>fp, "sudo "+bin_path+"/hhotspot -size=1024 -sim_time=1000 -streams=1 -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file="+data_path+"/gold_half_1024_1000 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/hhotspot -size=1024 -sim_time=1000 -streams=2 -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file="+data_path+"/gold_half_1024_1000 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/hhotspot -size=1024 -sim_time=1000 -streams=4 -temp_file="+data_path+"/temp_1024 -power_file="+data_path+"/power_1024 -gold_file="+data_path+"/gold_half_1024_1000 -iterations=10000000"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_hhotspot_cuda_x2\n"

sys.exit(0)
