#!/usr/bin/python

import os
import sys
import ConfigParser

print "Generating CLAMR for K1"

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

data_path=installDir+"data/CLAMR"
bin_path=installDir+"bin"
src_clamr = installDir+"src/CLAMR"

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

os.system("sudo rm -rf "+src_clamr+"/build")
os.system("mkdir "+src_clamr+"/build");
os.system("cd "+src_clamr+"/build && cmake -DGRAPHICS_TYPE=None -DPRECISION_TYPE=full_precision -DLOG=yes -DCMAKE_BUILD_TYPE=release ..");
os.system("cd "+src_clamr+"/build && make clamr_openmponly");
#os.system("cmake -DGRAPHICS_TYPE=None -DPRECISION_TYPE=full_precision -DLOG=yes -DCMAKE_BUILD_TYPE=release ..");

os.system("sudo "+installDir+"/scripts/run_clamr.sh big 8");
os.system("cd "+src_clamr+"/build && sudo ./clamr_openmponly -n 256 -t 2000 -g 100 -G data -J md5files")


fp = open(installDir+"scripts/how_to_run_clamr_k1", 'w')
print >>fp, "sudo "+src_clamr+"/build/run_clamr.sh <big | little> <threads>"
print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_clamr_k1\n"

sys.exit(0)

