#!/usr/bin/python

import os
import sys
import ConfigParser
import copy

DATASETS = [
    # normal
    {'set': 't10k-images-idx3-ubyte', 'label': 't10k-labels-idx1-ubyte', 'gold': 'gold_t10k_images.test'},

]

WEIGHTS = ['lenet_base.weights', 'lenet_L1.weights', 'lenet_L2.weights']


def main(board):
    print "Generating py-faster for CUDA on " + str(board)

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

    generate = [str("cd " + src_lenet), ]
    execute = []

    for w in WEIGHTS:
        for i in DATASETS:
            gold = data_path + '/' + i['gold']
            set = data_path + '/' + i['set']
            labels = data_path + '/' + i['label']
            weights = data_path + '/' + w
            gen = [None] * 6
            gen[0] = [' gold_gen ']
            gen[1] = [set, labels]
            gen[2] = [weights]
            gen[3] = [gold]
            gen[4] = [1000, 0, 1]
            gen[5] = ['sudo ', src_lenet + "/leNetCUDA "]

            exe = copy.deepcopy(gen)
            exe[0][0] = ' rad_test '
            exe[4][2] = 1000

            generate.append()
            execute.append(" ".join([''.join(str(value)) for value in exe]))

    # os.system("cd " + src_py_faster)
    for i in generate:
        # if os.system(str(i)) != 0:
        #     print "Something went wrong with generate of ", str(i)
        #     exit(1)
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