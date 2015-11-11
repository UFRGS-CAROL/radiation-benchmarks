#/usr/bin/python

import os
import sys
import ConfigParser

print "Setting up gold generation for all HSA codes."

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

os.system("cd "+installDir+"/src/heterogeneous/hsa/ ; cmake . -DLOGS=1 ; make");
os.system("cd "+installDir+"/src/heterogeneous/hsa/src/hsa/fir_hsa/ ; sudo ./config.py");
os.system("cd "+installDir+"/src/heterogeneous/hsa/src/hsa/page_rank_hsa/ ; sudo ./config.py");
os.system("cd "+installDir+"/src/heterogeneous/hsa/src/hsa/kmeans_hsa/ ; sudo ./config.py");

sys.exit(0)
