#!/usr/bin/python

import os
import sys
import ConfigParser
import copy

INPUT = ['image.pgm']
ITERATIONS = 100000

def main(board):
    print "Generating SRADV1 for CUDA on " + str(board)

    confFile = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(confFile)

        installDir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    data_path = installDir + "data/srad_v1"
    bin_path = installDir + "bin"
    src_srad = installDir + "src/cuda/srad_v1"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    generate = ["mkdir -p " + data_path, "cd " + src_srad, "make clean", "make",  "mv ./srad_v1 " + bin_path + "/"]
    execute = []

    for i in INPUT:
        inputImg = data_path + "/" + i
        #1000 0.5 image.pgm 1 image_out.gold

        gen = [None] * 6
        gen[0] = ['sudo ', bin_path + "/srad_v1 "]
        gen[1] = [1]
        gen[2] = [0.5]
        gen[3] = [inputImg]
        gen[4] = [0]
        gen[5] = [inputImg + "_out.gold"]

        #change mode and iterations for exe
        exe = copy.deepcopy(gen)
        exe[4] = [1]
        exe[1] = [ITERATIONS]

        generate.append(' '.join(str(r) for v in gen for r in v))
        execute.append(' '.join(str(r) for v in exe for r in v))


    # end for generate
    generate.append("make clean ")
    generate.append("make -C ../../include/")
    generate.append("make LOGS=1")
    generate.append("sudo mv ./srad_v1 " + bin_path + "/")

    for i in generate:
        # if os.system(str(i)) != 0:
        #     print "Something went wrong with generate of ", str(i)
        #     exit(1)
        print i

    fp = open(installDir + "scripts/how_to_run_srad_v1_cuda_" + board, 'w')

    for i in execute:
        print >> fp, "[\"" + str(i) + "\" , 0.016, \"srad_v1\"],"
        print "[\"" + str(i) + "\" , 0.016, \"srad_v1\"],"

    print "\nConfiguring done, to run check file: " + installDir + "scripts/how_to_run_srad_v1_cuda_" + str(
        board) + "\n"

    sys.exit(0)


if __name__ == "__main__":
    parameter = sys.argv[1:]
    if len(parameter) < 1:
        print "./config_generic <k1/x1/k40>"
    else:
        main(str(parameter[0]).upper())
