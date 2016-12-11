#!/usr/bin/python

import os
import sys
import ConfigParser
import sys
import copy


DATASETS = [
    # normal
    #{'txt': 'caltech.pedestrians.critical.1K.txt', 'gold': 'gold.caltech.critical.1K.test'},
    #{'txt': 'caltech.pedestrians.1K.txt', 'gold': 'gold.caltech.1K.test'},
    # abft
    {'txt': 'caltech.pedestrians.1K.txt', 'gold': 'gold.caltech.abft.1K.test'},
    {'txt': 'caltech.pedestrians.critical.1K.txt', 'gold': 'gold.caltech.critical.abft.1K.test'},

    #{'txt': 'voc.2012.1K.txt', 'gold': 'gold.voc.2012.1K.test'},
    {'txt': 'voc.2012.1K.txt', 'gold': 'gold.voc.2012.abft.1K.test'},
]


def main(board):
    print "Generating darknet for CUDA, board:" + board

    confFile = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(confFile)

        installDir = config.get('DEFAULT', 'installdir') + "/"
        varDir = config.get('DEFAULT', 'vardir') + "/"
        logDir = config.get('DEFAULT', 'logdir') + "/"
        tmpDir = config.get('DEFAULT', 'tmpdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    data_path = installDir + "data/darknet"
    bin_path = installDir + "bin"
    src_darknet = installDir + "src/cuda/darknet"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777);
        os.chmod(data_path, 0777);


    #	./darknet -e yolo -m valid -c cfg/yolo.cfg -w data/yolo.weights -n 3 -d data/gold_voc2012.test
    # -l data/voc.2012.DEBUG.txt -b ~/radiation-benchmarks/src/cuda/darknet -x 0 -s 1 -a 1
    generate = ["cd " + src_darknet, "make clean GPU=1", "make -j 4 GPU=1 ", "mv ./darknet " + bin_path + "/"]
    execute = []
    for i in DATASETS:
        abft = 0
        if 'abft' in i['gold']:
            abft = 1

        gold = data_path + '/' + i['gold']
        txt_list = installDir + 'data/networks_img_list/' + i['txt']
        gen = {
            'bin': [bin_path, "/darknet"],
            'e': [' -e ', 'yolo'],  # execution_type =
            'm': [' -m ', 'valid'],  # execution_model =
            'c': [' -c ', data_path + '/yolo.cfg'],  # config_file =
            'w': [' -w ', data_path + '/yolo.weights'],  # weights =
            'n': [' -n ', '1'],  # iterations =  #it is not so much, since each dataset have at least 10k of images
            'g': [' -g ', gold],  # base_caltech_out = base_voc_out = src_darknet
            'l': [' -l ', txt_list],
            'b': [' -b ', src_darknet],
            'x': [' -x ', 0],
            's': [' -s ', 0],
            'a': [' -a ', abft],
        }

        exe = copy.deepcopy(gen)
        exe['n'][1] = 10000
        exe['g'][0] = ' -d '

        exe_save = copy.deepcopy(exe)
        exe_save['s'][1] = 1

        generate.append(" ".join([''.join(map(str, value)) for key, value in gen.iteritems()]))

        execute.append(" ".join([''.join(map(str, value)) for key, value in exe.iteritems()]))
        execute.append(" ".join([''.join(map(str, value)) for key, value in exe_save.iteritems()]))



    # end for generate

    generate.append("make clean GPU=1 ")
    generate.append("make -C ../../include/ cuda")
    generate.append("make -j 4 GPU=1 LOGS=1")
    generate.append("mv ./darknet " + bin_path + "/")

    for i in generate:
        if os.system(str(i)) != 0:
            print "Something went wrong with generate of ", str(i)
            exit(1)
        print i

    fp = open(installDir + "scripts/how_to_run_darknet_cuda_" + board, 'w')

    for i in execute:
        print >> fp, "[\"""sudo " + str(i) + "\" , 0.016, \"darknet\"],"

    print "\nConfiguring done, to run check file: " + installDir + "scripts/how_to_run_darknet_cuda_" + board + "\n"

    sys.exit(0)


if __name__ == "__main__":
    parameter = sys.argv[1:]
    if len(parameter) < 1:
        print "./config_generic <k1/x1/k40>"
    else:
        main(str(parameter[0]).upper())



    # voc_gen_abft = copy.deepcopy(voc_gen_clean)
    # gold_abft = data_path + '/gold.voc.2012.1K.abft.test'
    # voc_gen_abft['a'][1] = 1
    # voc_gen_abft['g'][1] = gold_abft
    #
    # #caltech gen
    # gold = data_path + '/gold.caltech.1K.test'
    # txt_list = installDir + 'data/networks_img_list/caltech.pedestrians.1K.txt'
    # caltech_gen_clean = copy.deepcopy(voc_gen_clean)
    # caltech_gen_clean['l'][1] = txt_list
    # caltech_gen_clean['g'][1] = gold
    #
    # gold_abft = data_path + '/gold.caltech.1K.abft.test'
    # caltech_gen_abft =copy.deepcopy(caltech_gen_clean)
    # caltech_gen_abft['a'][1] = 1
    # caltech_gen_abft['g'][1] = gold_abft
    # execute################################################################
    # caltech----------
    # clean
    # abft
    # caltech_exe_abft = copy.deepcopy(caltech_gen_abft)
    # caltech_exe_abft['n'][1] = 10000
    # caltech_exe_abft['g'][0] = ' -d '
    # #save layers
    # caltech_exe_save = copy.deepcopy(caltech_exe_clean)
    # caltech_exe_save['s'][1] = 1
    #
    # #voc---------------
    # #clean
    # voc_exe_clean = copy.deepcopy(voc_gen_clean)
    # voc_exe_clean['n'][1] = 10000
    # voc_exe_clean['g'][0] = ' -d '
    #
    # #abft
    # voc_exe_abft = copy.deepcopy(voc_gen_abft)
    # voc_exe_abft['n'][1] = 10000
    # voc_exe_abft['g'][0] = ' -d '
    # save layers
    # generate.append(" ".join([''.join(map(str, value)) for key,value in caltech_gen_clean.iteritems()]))
    # generate.append(" ".join([''.join(map(str, value)) for key,value in  caltech_gen_abft.iteritems()]))
    # generate.append(" ".join([''.join(map(str, value)) for key,value in  voc_gen_clean.iteritems()]))
    # execute.append (" ".join([''.join(map(str, value))for key,value in caltech_exe_abft.iteritems()]))
    # execute.append (" ".join([''.join(map(str, value))for key,value in caltech_exe_save.iteritems()]))
    #
    # execute.append (" ".join([''.join(map(str, value))for key,value in voc_exe_clean.iteritems()]))
    # execute.append (" ".join([''.join(map(str, value))for key,value in voc_exe_abft.iteritems()]))
    # execute.append (" ".join([''.join(map(str, value))for key,value in voc_exe_save.iteritems()]))
