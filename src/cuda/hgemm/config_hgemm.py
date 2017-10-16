#!/usr/bin/python


import ConfigParser
import copy
import os

import sys

SIZES=[1024, 2048, 4096, 8192]
ITERATIONS=10000


def main(board, debug=None):

    benchmark_bin = "cudaHGEMM"
    print "Generating "+ benchmark_bin + " for CUDA, board:" + board

    confFile = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(confFile)
        installDir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)


    data_path = installDir + "data/hgemm"
    bin_path = installDir + "bin"
    src_dgemm = installDir + "src/cuda/hgemm"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)


    # change it for lava
    generate = ["cd " + src_dgemm, "make clean", "make -C ../../include ", "make", "mkdir -p " + data_path,
                "mv ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    for i in SIZES:
        inputFile = data_path + "/"

        gen = [None] * 6
        gen[0] = ['sudo ', src_dgemm + "/generateMatricesHalf "]
        gen[1] = ['-size=' + str(i)]
        gen[2] = ['-input_a=' + inputFile + 'hgemmA_' + str(i) + '.matrix']
        gen[3] = ['-input_b=' + inputFile + 'hgemmB_' + str(i) + '.matrix']
        gen[4] = ['-gold=' + inputFile + "H_GOLD_" + str(i) + ".matrix"]  # change for execute
        gen[5] = []

        # change mode and iterations for exe
        exe = copy.deepcopy(gen)
        exe[0][1] = bin_path + '/' + benchmark_bin
        exe[5] = ['-iterations=' + str(ITERATIONS)]


        generate.append(' '.join(str(r) for v in gen for r in v))
        execute.append(' '.join(str(r) for v in exe for r in v))


    execute_and_write_how_to_file(execute, generate, installDir, benchmark_bin, debug)

def execute_and_write_how_to_file(execute, generate, installDir, benchmark_bin, debug):
    for i in generate:
        if debug == None and os.system(str(i)) != 0:
            print "Something went wrong with generate of ", str(i)
            exit(1)
        print i
    fp = open(installDir + "scripts/json_files/" + benchmark_bin + ".json", 'w')

    list_to_print = ["["]
    for ii, i in enumerate(execute):
        command = "{\"killcmd\": \"killall -9 " + benchmark_bin + "\", \"exec\": \"" + str(i) + "\"}"
        if ii != len(execute) - 1:
            command += ', '
        list_to_print.append(command)
    list_to_print.append("]")

    for i in list_to_print:
        print >> fp, i
        print i
    fp.close()
    print "\nConfiguring done, to run check file: " + installDir + "scripts/json_files/" + benchmark_bin + ".json"



if __name__ == "__main__":
    parameter = sys.argv[1:]
    if len(parameter) < 1:
        print "./config_generic <k1/x1/x2/k40/titan> <1/true/True if you want to debug the application>"
    else:
        try:
            main(str(parameter[0].upper()), bool(parameter[1]))
        except:
            main(str(parameter[0]).upper())
