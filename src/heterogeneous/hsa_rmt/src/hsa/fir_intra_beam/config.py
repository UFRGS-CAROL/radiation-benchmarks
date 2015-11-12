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

data_path=installDir+"bin/fir_intra_beam"
bin_path=installDir+"bin/fir_intra_beam"
src_fir_intra_beam = installDir+"src/heterogeneous/hsa_rmt/src/hsa/fir_intra_beam"

os.system("sudo mkdir "+src_fir_intra_beam+"/input");
os.system("sudo mkdir "+src_fir_intra_beam+"/output");

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

#os.system("cd "+src_fir);
os.system("cd "+src_fir_intra_beam+"; sudo ./fir_intra_beam -b 2048 -n 2048 -g");
os.system("cd "+src_fir_intra_beam+"; sudo ./fir_intra_beam -b 4096 -n 4096 -g");
os.system("cd "+src_fir_intra_beam+"; sudo ./fir_intra_beam -b 8192 -n 8192 -g");
os.system("sudo chmod 777 input output input/* output/* ");
os.system("mv input output "+data_path);
os.system("mv ./fir_intra_beam "+bin_path);
os.system("cp run_* "+bin_path);

fp = open(installDir+"scripts/how_to_run_fir_intra_beam", 'w')
print >>fp, "cd "+bin_path+"; python "+bin_path+"/run_fir_intra_beam_2048.sh"
print >>fp, "cd "+bin_path+"; python "+bin_path+"/run_fir_intra_beam_4096.sh"
print >>fp, "cd "+bin_path+"; python "+bin_path+"/run_fir_intra_beam_8192.sh"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_fir_intra_beam\n"

sys.exit(0)
