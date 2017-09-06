#!/usr/bin/python

import os
import sys
import ConfigParser

print "Generating hlava for CUDA Tegra X2"

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

data_path=installDir+"data/hlava"
bin_path=installDir+"bin"
src_hlava = installDir+"src/cuda/hlavaMD"

if not os.path.isdir(data_path):
    os.mkdir(data_path, 0777);
    os.chmod(data_path, 0777);
if not os.path.isdir(bin_path):
    os.mkdir(bin_path, 0777);
    os.chmod(bin_path, 0777);

os.system("cd "+src_hlava)
os.system("sudo ./hlava -boxes=10 -generate -output_gold=gold_10 -iterations=1 -streams=1")
os.system("sudo ./hlava -boxes=15 -generate -output_gold=gold_15 -iterations=1 -streams=1")

os.system("sudo chmod 777 gold_* ");
os.system("sudo chmod 777 input* ");
os.system("mv gold_* "+data_path);
os.system("mv input* "+data_path);
os.system("mv ./hlava "+bin_path)

fp = open(installDir+"scripts/how_to_run_hlava_cuda_x2", 'w')
print >>fp, "sudo "+bin_path+"/hlava -boxes=10 -input_distances="+data_path+"/input_distances_half_10 -input_charges="+data_path+"/input_charges_half_10 -output_gold="+data_path+"/gold_half_10 -iterations=10000000 -streams=1"
print >>fp, "sudo "+bin_path+"/hlava -boxes=15 -input_distances="+data_path+"/input_distances_half_15 -input_charges="+data_path+"/input_charges_half_15 -output_gold="+data_path+"/gold_half_15 -iterations=10000000 -streams=1"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_hlava_cuda_x2\n"

sys.exit(0)
