#!/usr/bin/python

import os
import sys
import ConfigParser

print "Generating lava for CUDA"

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

data_path=installDir+"data/lava"
bin_path=installDir+"bin"
src_lava = installDir+"src/cuda/lava"

if not os.path.isdir(data_path):
    os.mkdir(data_path, 0777);
    os.chmod(data_path, 0777);

os.system("cd "+src_lava)
os.system("sudo ./lava -boxes=10 -generate -output_gold=gold_10 -iterations=1 -streams=1")
os.system("sudo ./lava -boxes=15 -generate -output_gold=gold_15 -iterations=1 -streams=1")
os.system("sudo ./lava -boxes=20 -generate -output_gold=gold_20 -iterations=1 -streams=1")
os.system("sudo ./lava -boxes=25 -generate -output_gold=gold_25 -iterations=1 -streams=1")

os.system("sudo chmod 777 gold_* ");
os.system("mv gold_* "+data_path);
os.system("mv ./lava "+bin_path)

fp = open(installDir+"scripts/how_to_run_lava_cuda_K40", 'w')
print >>fp, "sudo "+bin_path+"/lava -boxes=10 -input_distances="+data_path+"/input_distances_10 -input_charges="+data_path+"/input_charges_10 -output_gold="+data_path+"/gold_10 -iterations=10000000 -streams=1"
print >>fp, "sudo "+bin_path+"/lava -boxes=15 -input_distances="+data_path+"/input_distances_15 -input_charges="+data_path+"/input_charges_15 -output_gold="+data_path+"/gold_15 -iterations=10000000 -streams=1"
print >>fp, "sudo "+bin_path+"/lava -boxes=20 -input_distances="+data_path+"/input_distances_20 -input_charges="+data_path+"/input_charges_20 -output_gold="+data_path+"/gold_20 -iterations=10000000 -streams=1"
print >>fp, "sudo "+bin_path+"/lava -boxes=25 -input_distances="+data_path+"/input_distances_25 -input_charges="+data_path+"/input_charges_25 -output_gold="+data_path+"/gold_25 -iterations=10000000 -streams=1"


#8 streams
print >>fp, "sudo "+bin_path+"/lava -boxes=10 -input_distances="+data_path+"/input_distances_10 -input_charges="+data_path+"/input_charges_10 -output_gold="+data_path+"/gold_10 -iterations=10000000 -streams=8"
print >>fp, "sudo "+bin_path+"/lava -boxes=15 -input_distances="+data_path+"/input_distances_15 -input_charges="+data_path+"/input_charges_15 -output_gold="+data_path+"/gold_15 -iterations=10000000 -streams=8"
print >>fp, "sudo "+bin_path+"/lava -boxes=20 -input_distances="+data_path+"/input_distances_20 -input_charges="+data_path+"/input_charges_20 -output_gold="+data_path+"/gold_20 -iterations=10000000 -streams=8"
print >>fp, "sudo "+bin_path+"/lava -boxes=25 -input_distances="+data_path+"/input_distances_25 -input_charges="+data_path+"/input_charges_25 -output_gold="+data_path+"/gold_25 -iterations=10000000 -streams=8"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_lava_cuda\n"

sys.exit(0)
