#!/usr/bin/python

import os
import sys
import ConfigParser

def main(board):
    print "Config generating for HOG on " + str(board)

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
    src_hog = installDir+"src/cuda/histogram_ori_gradients"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    os.system("cd "+src_hog)
    os.system("make clean")
    make_str = "make -j 4 LOGS=1 "
    if 'X1' in board:
        make_str += "ARCH_I=53"

    os.system(make_str)
    gold_txt = installDir + 'data/networks_img_list/voc.2012.1K.txt'

    #../../../../data/CALTECH/set10/V000/ --dst_data dataset.txt --hit_threshold 0.9 --gr_threshold 1 --nlevels 100
    generate_hog = ["sudo ", bin_path + "/gold_gen ",  gold_txt, " --dst_data " + data_path + "/ ", " --hit_threshold 0.9 --gr_threshold 1 --nlevels 100"]

    #$(HOG_EXT_DIR)/hog_ext $(HOG_OCV_DIR)/hog_opencv $(HOG_HAR_DIR)/hog_har_eccon  $(HOG_EOF_DIR)/hog_har_eccoff
    HOG_EXT = "hog_ext "
    HOG_HAR = "hog_har_eccon "
    HOG_EOF = "hog_har_eccoff "

    execute_hog = {
        'hog_ext':["sudo ",  bin_path + HOG_EXT,  gold_txt, " --dst_data " + data_path + "/ ", " --iterations 10000000"],
        'hog_ecc':["sudo ",  bin_path + HOG_HAR,  gold_txt, " --dst_data " + data_path + "/ ", " --iterations 10000000"],
        'hog_off':["sudo ",  bin_path + HOG_EOF,  gold_txt, " --dst_data " + data_path + "/ ", " --iterations 10000000"],
    }

    #move all binaries to bin path
    os.system("mv hog_extrated/hog_ext hog_hardened_eccoff/hog_har_eccoff hog_hardened_eccon/hog_har_eccon "+bin_path)

    fp = open(installDir+"scripts/how_to_run_hog_cuda_" + str(board), 'w')

    execute = []
    execute.append(" ".join([''.join(map(str, value)) for key, value in execute_hog.iteritems()]))
    for i in execute:
        print >> fp, "[\"" + str(i) + "\" , 0.016, \"hog\"]," #it is wrong

    print "\nConfiguring done, to run check file: "+installDir+"scripts/how_to_run_hog_cuda_" + board +"\n"

    sys.exit(0)
