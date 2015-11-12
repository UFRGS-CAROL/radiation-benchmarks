#!/usr/bin/python

import os
import sys
import ConfigParser

print "Generating quicksort for CUDA K40"

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

data_path=installDir+"data/quicksort"
bin_path=installDir+"bin"
src_quicksort = installDir+"src/cuda/quicksort"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

os.system("cd "+src_quicksort)

os.system("sudo ./quicksort -size=1048576 -input=quicksort_input_134217728 -gold=quicksort_gold_1048576 -generate -iterations=1")
os.system("sudo ./quicksort -size=33554432 -input=quicksort_input_134217728 -gold=quicksort_gold_33554432 -generate -iterations=1")
os.system("sudo ./quicksort -size=67108864 -input=quicksort_input_134217728 -gold=quicksort_gold_67104464 -generate -iterations=1")
os.system("sudo ./quicksort -size=134217728 -input=quicksort_input_134217728 -gold=quicksort_gold_134217728 -generate -iterations=1")
os.system("sudo chmod 777 quicksort_*");
os.system("mv quicksort_* "+data_path);
os.system("mv ./quicksort "+bin_path)

fp = open(installDir+"scripts/how_to_run_quicksort_cuda_K40", 'w')
print >>fp, "sudo "+bin_path+"/quicksort -size=1048576 -input="+data_path+"/quicksort_input_134217728 -gold="+data_path+"/quicksort_gold_1048576 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/quicksort -size=33554432 -input="+data_path+"/quicksort_input_134217728 -gold="+data_path+"/quicksort_gold_33554432 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/quicksort -size=67108864 -input="+data_path+"/quicksort_input_134217728 -gold="+data_path+"/quicksort_gold_67108864 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/quicksort -size=134217728 -input="+data_path+"/quicksort_input_134217728 -gold="+data_path+"/quicksort_gold_134217728 -iterations=10000000"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_quicksort_cuda\n"

sys.exit(0)
