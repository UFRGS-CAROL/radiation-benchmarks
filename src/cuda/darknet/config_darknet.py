#!/usr/bin/python

import os
import sys
import ConfigParser
import sys

def main(board):

    print "Generating darknet for CUDA, board:"+board

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

    data_path=installDir+"data/darknet"
    bin_path=installDir+"bin"
    src_darknet = installDir+"src/cuda/darknet"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777);
        os.chmod(data_path, 0777);


    execution_type = 'yolo'
    execution_model = 'valid'
    config_file = data_path + '/yolo.cfg'
    weights = data_path + '/yolo.weights'
    iterations = '1000' #it is not so much, since each dataset have at least 10k of images
    base_caltech_out = src_darknet
    base_voc_out = src_darknet


    img_datasets = str.replace(data_path,'/darknet', '') + "/networks_img_list"
    #all inputs
    caltech_gold_FULL = data_path + '/gold_caltech_full.test'
    caltech_img_list_FULL = img_datasets + '/caltech/'+board+'/caltech.pedestrians.1K.txt'
    cal_FULL_str = caltech_gold_FULL + " -l " + caltech_img_list_FULL

    #half of inputs
    caltech_gold_HALF = data_path + '/gold_caltech_full.test'
    caltech_img_list_HALF = img_datasets + '/caltech/'+board+'/caltech.pedestrians.5K.txt'
    cal_HALF_str = caltech_gold_HALF + " -l " + caltech_img_list_HALF

    #VOC
    #all inputs
    voc_gold_FULL = data_path + '/gold_voc_full.test'
    voc_img_list_FULL = img_datasets + '/voc/'+board+'/voc.2012.1K.txt'
    voc_FULL_str = voc_gold_FULL + " -l " + voc_img_list_FULL

    #half of inputs
    voc_gold_HALF = data_path + '/gold_voc_full.test'
    voc_img_list_HALF = img_datasets + '/voc/'+board+'/voc.2012.5K.txt'
    voc_HALF_str = voc_gold_HALF + " -l " + voc_img_list_HALF


    #voc
    vc_full_gen = "sudo "+ bin_path+"/darknet -e " + execution_type + " -m " + execution_model + " -c " + \
          config_file + " -w " +  weights + " -n 1 -g " + voc_FULL_str + " -b " + base_voc_out + " -x 0"


    #caltech
    cl_full_gen = "sudo "+ bin_path+"/darknet -e " + execution_type + " -m " + execution_model + " -c " + \
          config_file + " -w " +  weights + " -n 1 -g " + cal_FULL_str + " -b " + base_caltech_out + " -x 0"

    #execute################################################################
    #voc
    vc_half_ex =  "[\"sudo "+ bin_path + "/darknet -e " + execution_type + " -m " + execution_model + " -c " + \
          config_file + " -w " +  weights + " -n 10000 -d " + voc_HALF_str + " -b " + base_voc_out + " -x 0 \" , 1, \"darknet\"],"

    vc_full_ex = "[\"sudo "+ bin_path + "/darknet -e " + execution_type + " -m " + execution_model + " -c " + \
          config_file + " -w " +  weights + " -n 10000 -d " + voc_FULL_str + " -b " + base_voc_out + " -x 0 \" , 1, \"darknet\"],"


    #caltech
    cl_half_ex = "[\"sudo "+ bin_path + "/darknet -e " + execution_type + " -m " + execution_model + " -c " + \
          config_file + " -w " +  weights + " -n 10000 -d " + cal_HALF_str + " -b " + base_caltech_out + " -x 0 \" , 1, \"darknet\"],"

    cl_full_ex = "[\"sudo "+ bin_path + "/darknet -e " + execution_type + " -m " + execution_model + " -c " + \
          config_file + " -w " +  weights + " -n 10000 -d " + cal_FULL_str + " -b " + base_caltech_out + " -x 0 \" , 1, \"darknet\"],"

    os.system("cd "+src_darknet)

    os.system("make clean")
    os.system("make -j 4 GPU=1 ARCH_I=53")
    os.system("mv ./darknet "+bin_path)

    #os.system(vc_half_gen)
    os.system(vc_full_gen)
    #os.system(cl_half_gen)
    os.system(cl_full_gen)

    os.system("make clean")
    os.system("make -C ../../include/")
    os.system("make -j 4 GPU=1 LOGS=1 ARCH_I=53")

    # os.system("sudo chmod 777 gold_* ");
    # os.system("mv gold_* "+data_path);
    os.system("mv ./darknet "+bin_path)


    fp = open(installDir+"scripts/how_to_run_darknet_cuda_"+board, 'w')

    print >>fp, vc_half_ex
    print >>fp, vc_full_ex
    print >>fp, cl_half_ex
    print >>fp, cl_full_ex


    print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_darknet_cuda_"+board+"\n"

    sys.exit(0)


if __name__ == "__main__":
    parameter = sys.argv[1:]
    if len(parameter) < 1:
        print "./config_generic <k1/x1/k40>"
    else:
        main(str(parameter[0]).upper())
