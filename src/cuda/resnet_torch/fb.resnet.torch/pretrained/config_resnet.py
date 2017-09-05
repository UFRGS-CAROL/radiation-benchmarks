#!/usr/bin/python

import os
import sys
import ConfigParser
import copy

DATASETS = [
    # normal
    {'txt': 'caltech.pedestrians.1K.txt', 'gold': 'gold.caltech.1K.csv', 'mode': 'full'},
    {'txt': 'urban.street.1.1K.txt', 'gold': 'gold.urban.street.1.1K.csv', 'mode': 'full'},
    {'txt': 'voc.2012.1K.txt', 'gold': 'gold.voc.2012.1K.csv', 'mode': 'full'},


]

WEIGHTS = ['resnet-200.t7'] #, 'lenet_L1.weights', 'lenet_L2.weights']


def main(board):
    print "Generating ResNet for CUDA on " + str(board)

    confFile = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(confFile)

        installDir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    data_path = installDir + "data/resnet_torch"
    src_resnet = installDir + "src/cuda/resnet_torch/fb.resnet.torch/pretrained"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    generate = ["make -C ../../../../include/log_helper_swig_wraper/ log_helper_lua",
                "mkdir -p /home/carol/radiation-benchmarks/data/resnet_torch/",
                "cd " + src_resnet]
    execute = []

    for i in DATASETS:
        gold = data_path + '/' + i['gold']
        set = data_path + '/' + i['txt']

        weights = data_path + '/' + WEIGHTS[0]

        gen = [None] * 6
        gen[0] = ['sudo ', src_resnet + "/classify_radiation.lua "]
        gen[1] = [weights]
        gen[2] = [' generate ']
        gen[3] = [set]
        gen[4] = [gold]
        gen[5] = [1, '']

        exe = copy.deepcopy(gen)
        exe[2] = [' rad_test ']
        exe[5] = [1000, 'log']

        generate.append(' '.join(str(r) for v in gen for r in v))
        execute.append(' '.join(str(r) for v in exe for r in v))




    for i in generate:
        if os.system(str(i)) != 0:
            print "Something went wrong with generate of ", str(i)
            exit(1)
        print i

    fp = open(installDir + "scripts/how_to_run_resnet_torch_" + board, 'w')

    for i in execute:
        print >> fp, "[\"" + str(i) + "\" , 0.016, \"luajit\"],"
        print "[\"" + str(i) + "\" , 0.016, \"luajit\"],"

    print "\nConfiguring done, to run check file: " + installDir + "scripts/how_to_run_resnet_cuda_" + str(
        board) + "\n"

    sys.exit(0)


if __name__ == "__main__":
    parameter = sys.argv[1:]
    if len(parameter) < 1:
        print "./config_generic <k1/x1/k20/k40/tx>"
    else:
        main(str(parameter[0]).upper())