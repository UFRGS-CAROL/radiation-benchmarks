#!/usr/bin/python

import os
import sys
import ConfigParser

print "Generating radixsort for CUDA K40"

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

data_path=installDir+"data/radixsort"
bin_path=installDir+"bin"
src_radixsort = installDir+"src/cuda/radixsort"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

os.system("cd "+src_radixsort)

os.system("sudo ./radixsort -size=1048576 -input=radixsort_input_134217728 -gold=radixsort_gold_1048576 -generate -iterations=1")
os.system("sudo ./radixsort -size=33554432 -input=radixsort_input_134217728 -gold=radixsort_gold_33554432 -generate -iterations=1")
os.system("sudo ./radixsort -size=67108864 -input=radixsort_input_134217728 -gold=radixsort_gold_67104464 -generate -iterations=1")
os.system("sudo ./radixsort -size=134217728 -input=radixsort_input_134217728 -gold=radixsort_gold_134217728 -generate -iterations=1")
os.system("sudo chmod 777 radixsort_*");
os.system("mv radixsort_* "+data_path);
os.system("mv ./radixsort "+bin_path)

fp = open(installDir+"scripts/how_to_run_radixsort_cuda_K40", 'w')
print >>fp, "sudo "+bin_path+"/radixsort -size=1048576 -input="+data_path+"/radixsort_input_134217728 -gold="+data_path+"/radixsort_gold_1048576 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/radixsort -size=33554432 -input="+data_path+"/radixsort_input_134217728 -gold="+data_path+"/radixsort_gold_33554432 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/radixsort -size=67108864 -input="+data_path+"/radixsort_input_134217728 -gold="+data_path+"/radixsort_gold_67108864 -iterations=10000000"
print >>fp, "sudo "+bin_path+"/radixsort -size=134217728 -input="+data_path+"/radixsort_input_134217728 -gold="+data_path+"/radixsort_gold_134217728 -iterations=10000000"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_radixsort_cuda\n"

sys.exit(0)
