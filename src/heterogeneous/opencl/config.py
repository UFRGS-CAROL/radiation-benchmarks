#!/usr/bin/python

import os
import sys
import ConfigParser

print "Setting up gold generation for all OpenCL codes."

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

os.system("cd "+installDir+"/src/heterogeneous/opencl/ ; cmake . ; make");
os.system("cd "+installDir+"/src/heterogeneous/opencl/src/opencl20/fir_cl20/ ; sudo ./config.py");
os.system("cd "+installDir+"/src/heterogeneous/opencl/src/opencl20/pagerank_cl20/ ; sudo ./config.py");

sys.exit(0)
