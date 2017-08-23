#!/usr/bin/python

import os
import sys
import ConfigParser
import copy

DATASETS = [
    # normal
    {'set': 't10k-images-idx3-ubyte', 'label': 't10k-labels-idx1-ubyte', 'gold': 'gold_t10k_images.test'},

]

WEIGHTS = ['lenet_base.weights'] #, 'lenet_L1.weights', 'lenet_L2.weights']


def main(board):
    print "Generating Lenet for CUDA on " + str(board)

    confFile = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(confFile)

        installDir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    data_path = installDir + "data/lenet"
    bin_path = installDir + "bin"
    src_lenet = installDir + "src/cuda/LeNetCUDA"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    generate = ["cd " + src_lenet, "make clean GPU=1", "make -j4 GPU=1 ", "mv ./leNetCUDA " + bin_path + "/"]
    execute = []

    for w in WEIGHTS:
        for i in DATASETS:
            gold = data_path + '/' + i['gold']
            set = data_path + '/' + i['set']
            labels = data_path + '/' + i['label']
            weights = data_path + '/' + w
            gen = [None] * 6
            gen[0] = ['sudo ', src_lenet + "/leNetCUDA "]
            gen[1] = [' gold_gen ']
            gen[2] = [set, labels]
            gen[3] = [weights]
            gen[4] = [gold]
            gen[5] = [1000, 0, 1]

            exe = copy.deepcopy(gen)
            exe[1] = [' rad_test ']
            exe[5][2] = 1000

            generate.append(' '.join(str(r) for v in gen for r in v))
            execute.append(' '.join(str(r) for v in exe for r in v))



    # end for generate
    generate.append("make clean GPU=1 ")
    generate.append("make -C ../../include/")
    generate.append("make -j 4 GPU=1 LOGS=1")
    generate.append("mv ./leNetCUDA " + bin_path + "/")

    for i in generate:
        if os.system(str(i)) != 0:
            print "Something went wrong with generate of ", str(i)
            exit(1)
        print i

    fp = open(installDir + "scripts/how_to_run_py_faster_rcnn_cuda_" + board, 'w')

    for i in execute:
        print >> fp, "[\"" + str(i) + "\" , 0.016, \"py_faster_rcnn.py\"],"
        print "[\"" + str(i) + "\" , 0.016, \"py_faster_rcnn.py\"],"

    print "\nConfiguring done, to run check file: " + installDir + "scripts/how_to_run_py_faster_rcnn_cuda_" + str(
        board) + "\n"

    sys.exit(0)


if __name__ == "__main__":
    parameter = sys.argv[1:]
    if len(parameter) < 1:
        print "./config_generic <k1/x1/k40>"
    else:
        main(str(parameter[0]).upper())