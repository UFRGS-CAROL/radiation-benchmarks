#!/usr/bin/python

import ConfigParser
import copy
import os
import sys

SIZES = [2048, 4096, 8192]
ITERATIONS = 10000

DEBUG_MODE = False
BENCHMARK_BIN = "cudaDGEMM"
DATA_PATH_BASE = "dgemm"
GENERATE_BIN_NAME = "generateMatricesSingle"


def main(board):
    benchmark_bin = BENCHMARK_BIN
    print "Generating " + benchmark_bin + " for CUDA, board:" + board

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    data_path = install_dir + "data/" + DATA_PATH_BASE
    bin_path = install_dir + "bin"
    src_dgemm = install_dir + "src/cuda/" + DATA_PATH_BASE

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    # change it for lava
    generate = ["cd " + src_dgemm, "make clean", "make -C ../../include ", "make -j 4", "mkdir -p " + data_path,
                "mv -f ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    # gen only for max size
    max_size = max(SIZES)
    for i in SIZES:
        input_file = data_path + "/"

        gen = [None] * 6
        gen[0] = ['sudo ', src_dgemm + "/" + GENERATE_BIN_NAME + " "]
        gen[1] = ['-size=' + str(i)]
        gen[2] = ['-input_a=' + input_file + benchmark_bin + 'A_' + str(max_size) + '.matrix']
        gen[3] = ['-input_b=' + input_file + benchmark_bin + 'B_' + str(max_size) + '.matrix']
        gen[4] = ['-gold=' + input_file + "GOLD_" + str(max_size) + ".matrix"]  # change for execute
        gen[5] = []

        # change mode and iterations for exe
        exe = copy.deepcopy(gen)
        exe[0][1] = bin_path + '/' + benchmark_bin + " "
        exe[5] = ['-iterations=' + str(ITERATIONS)]

        if i == max_size:
            generate.append(' '.join(str(r) for v in gen for r in v))
        execute.append(' '.join(str(r) for v in exe for r in v))

    execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin)


def execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin):
    for i in generate:
        print i
        if not DEBUG_MODE:
            if os.system(str(i)) != 0:
                print "Something went wrong with generate of ", str(i)
                exit(1)

    list_to_print = ["[\n"]
    for ii, i in enumerate(execute):
        command = "{\"killcmd\": \"killall -9 " + benchmark_bin + "\", \"exec\": \"" + str(i) + "\"}"
        if ii != len(execute) - 1:
            command += ',\n'
        list_to_print.append(command)
    list_to_print.append("\n]")

    with open(install_dir + "scripts/json_files/" + benchmark_bin + ".json", 'w') as fp:
        fp.writelines(list_to_print)

    print "\nConfiguring done, to run check file: " + install_dir + "scripts/json_files/" + benchmark_bin + ".json"


if __name__ == "__main__":
    global DEBUG_MODE

    parameter = sys.argv[1:]
    try:
        DEBUG_MODE = sys.argv[2:]
    except:
        DEBUG_MODE = False
    if len(parameter) < 1:
        print "./config_generic <k1/x1/x2/k40/titan>"
    else:
        main(str(parameter[0]).upper())