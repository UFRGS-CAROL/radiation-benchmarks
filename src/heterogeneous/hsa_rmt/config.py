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

os.system("cd "+installDir+"/src/heterogeneous/hsa_rmt/ ; cmake . -DLOGS=1 ; make");
os.system("cd "+installDir+"/src/heterogeneous/hsa_rmt/src/hsa/fir_intra_beam/ ; sudo ./config.py");
os.system("cd "+installDir+"/src/heterogeneous/hsa_rmt/src/hsa/page_rank_intra_beam/ ; sudo ./config.py");
os.system("cd "+installDir+"/src/heterogeneous/hsa_rmt/src/hsa/kmeans_intra_beam/ ; sudo ./config.py");

sys.exit(0)
