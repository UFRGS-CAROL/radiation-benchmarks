#!/usr/bin/python


import ConfigParser
import copy
import os
import sys

BOXES = [10, 15, 20, 25, 30]
ITERATIONS = 10000
DEBUG_MODE = False
BENCHMARK_BIN = "slava"
DATA_PATH_BASE = "slava"
EMBEDDED_HOSTS = ['K1', 'X1', 'X2', 'APU']


def main(board):
    print "Generating Lava for CUDA, board:" + board

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    benchmark_bin = BENCHMARK_BIN
    data_path = install_dir + "data/" + DATA_PATH_BASE
    bin_path = install_dir + "bin"
    src_lava = install_dir + "src/cuda/" + DATA_PATH_BASE

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    # change it for lava
    generate = ["sudo mkdir -p " + bin_path, "cd " + src_lava, "make clean", "make -C ../../include ", "make",
                "mkdir -p " + data_path,
                "mv ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    # if board in EMBEDDED_HOSTS:
    #     BOXES.pop()

    for i in BOXES:
        input_file = data_path + "/"
        gen = [None] * 8
        gen[0] = ['sudo ', bin_path + "/" + benchmark_bin + " "]
        gen[1] = ['-boxes=' + str(i)]
        gen[2] = ['-generate ']
        gen[3] = ['-output_gold=' + input_file + "gold_" + str(i) + ""]
        gen[4] = ['-iterations=1']  # change for execute
        gen[5] = ['-streams=1']
        gen[6] = ['-input_charges=' + input_file + "input_charges_" + str(i)]
        gen[7] = ['-input_distances=' + input_file + "input_distances_" + str(i)]

        # change mode and iterations for exe
        exe = copy.deepcopy(gen)
        exe[2] = []
        exe[4] = ['-iterations=' + str(ITERATIONS)]

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
