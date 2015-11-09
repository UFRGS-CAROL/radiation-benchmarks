#!/usr/bin/python

import os
import sys
import ConfigParser


print "Gold generating"


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

data_path=installDir+"data/histogram_ori_gradients"
bin_path=installDir+"bin"
src_hog = installDir+"src/opencl/histogram_ori_gradients"

if not os.path.isdir(data_path):
            os.mkdir(data_path, 0777);
            os.chmod(data_path, 0777);

os.system("cd "+src_hog)
os.system("sudo ./gold_gen_hog_ocl -i=../../../data/histogram_ori_gradients/1x_pedestrians.jpg -o=gold1x_ocl")
os.system("sudo ./gold_gen_hog_ocl -i=../../../data/histogram_ori_gradients/4x_pedestrians.jpg -o=gold4x_ocl")
os.system("sudo ./gold_gen_hog_ocl -i=../../../data/histogram_ori_gradients/9x_pedestrians.jpg -o=gold9x_ocl")
os.system("sudo chmod 777 gold* ");
os.system("mv gold1x_ocl.data gold4x_ocl.data gold9x_ocl.data "+data_path);
os.system("mv ./hog_ocl ./gold_gen_hog_ocl "+bin_path)

fp = open(installDir+"scripts/how_to_run_hog_ocl", 'w')
print >>fp, "sudo "+bin_path+"/hog_ocl -i="+data_path+"/1x_pedestrians.jpg -o="+data_path+"/gold1x_ocl.data"
print >>fp, "sudo "+bin_path+"/hog_ocl -i="+data_path+"/4x_pedestrians.jpg -o="+data_path+"/gold4x_ocl.data"
print >>fp, "sudo "+bin_path+"/hog_ocl -i="+data_path+"/9x_pedestrians.jpg -o="+data_path+"/gold9x_ocl.data"

print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_hog_ocl\n"

sys.exit(0)
