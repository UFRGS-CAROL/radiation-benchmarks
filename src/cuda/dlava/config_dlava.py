#!/usr/bin/python
import ConfigParser
import copy
import os
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file


BOXES = [16, 20, 25]
ITERATIONS = 10000
BENCHMARK_BIN = "dlava"
DATA_PATH_BASE = "dlava"
EMBEDDED_HOSTS = ['K1', 'X1', 'X2', 'APU']


def config(board, debug):
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

    if board in EMBEDDED_HOSTS:
        BOXES.pop()

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

    #execute, generate, install_dir, benchmark_bin, debug
    execute_and_write_json_to_file(execute=execute, generate=generate,
                                    install_dir=install_dir, 
                                    benchmark_bin=benchmark_bin, 
                                    debug=debug)


if __name__ == "__main__":
    try:
        parameter = str(sys.argv[1:][0]).upper() 
        if parameter == 'DEBUG':
            debug_mode = True
    except:
        debug_mode = False
    
    board, _ = discover_board()
    config(board=board, debug=debug_mode)
