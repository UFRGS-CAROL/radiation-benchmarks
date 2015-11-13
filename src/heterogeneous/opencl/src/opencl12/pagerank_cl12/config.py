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

data_path=installDir+"bin/page_rank"
bin_path=installDir+"bin/page_rank"
src_page_rank = installDir+"src/heterogeneous/opencl/src/opencl12/pagerank_cl12/bin/x86_64/Release"

os.system("sudo mkdir "+src_page_rank+"/input");
os.system("sudo mkdir "+src_page_rank+"/output");

if not os.path.isdir(data_path):
	os.mkdir(data_path, 0777);
	os.chmod(data_path, 0777);

os.system("cd "+src_page_rank+" ; python PageRank_generateCsrMatrix.py ; mv csr_* input/");
#os.system("sudo cd "+src_page_rank);
os.system("cd "+src_page_rank+"; sudo ./pagerank_cl12 -m input/csr_2048_10.txt -g");
os.system("cd "+src_page_rank+"; sudo ./pagerank_cl12 -m input/csr_3072_10.txt -g");
os.system("cd "+src_page_rank+"; sudo ./pagerank_cl12 -m input/csr_4096_10.txt -g");
os.system("sudo chmod 777 input output input/* output/* ");
os.system("sudo mv input output "+data_path);
os.system("sudo mv ./pagerank_cl12 "+bin_path);
os.system("sudo cp run_* "+bin_path);

fp = open(installDir+"scripts/how_to_run_pagerank_cl12", 'w')
print >>fp, "cd "+bin_path+"; bash "+bin_path+"/run_page_rank_2048.sh"
print >>fp, "cd "+bin_path+"; bash "+bin_path+"/run_page_rank_3072.sh"
print >>fp, "cd "+bin_path+"; bash "+bin_path+"/run_page_rank_4096.sh"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_pagerank_cl12\n"

sys.exit(0)
