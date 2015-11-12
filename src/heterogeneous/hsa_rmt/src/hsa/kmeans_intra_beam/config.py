#!/usr/bin/python

import os
import sys
import ConfigParser

print "Generating gold files."

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

data_path=installDir+"bin/kmeans_intra_beam"
bin_path=installDir+"bin/kmeans_intra_beam"
src_kmeans_intra_beam = installDir+"src/heterogeneous/hsa_rmt/src/hsa/kmeans_intra_beam"

os.system("sudo mkdir "+src_kmeans_intra_beam+"/input");
os.system("sudo mkdir "+src_kmeans_intra_beam+"/output");

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

os.system("cd "+src_kmeans_intra_beam+"/datagen/ ; sudo ./gen_dataset.sh; mv 1* 3* ../input");
os.system("cd "+src_kmeans_intra_beam+"; sudo ./kmeans_intra_beam -i input/30000_34.txt -g");
os.system("cd "+src_kmeans_intra_beam+"; sudo ./kmeans_intra_beam -i input/100000_34.txt -g");
os.system("cd "+src_kmeans_intra_beam+"; sudo ./kmeans_intra_beam -i input/300000_34.txt -g");
os.system("sudo chmod 777 input output input/* output/* ");
os.system("mv input output "+data_path);
os.system("mv ./kmeans_intra_beam "+bin_path);
os.system("cp .run_* "+bin_path);

fp = open(installDir+"scripts/how_to_run_kmeans_intra_beam", 'w')
print >>fp, "cd "+bin_path+"; bash ./run_kmeans_intra_beam_30000.sh"
print >>fp, "cd "+bin_path+"; bash ./run_kmeans_intra_beam_100000.sh"
print >>fp, "cd "+bin_path+"; bash ./run_kmeans_intra_beam 300000.sh"


print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_kmeans_intra_beam\n"

sys.exit(0)
