#!/usr/bin/python

import ConfigParser
import copy
import os
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

SIZES = [4, 6]
THREADS = 4
ITERATIONS = [100000]


def config(board, debug):
    DATA_PATH_BASE = "lavamd"

    benchmark_bin = DATA_PATH_BASE + "_check"
    print("Generating {0} for OPENMP, board:{1}".format(benchmark_bin, board))

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print("Configuration setup error: {0}".format(str(e)))
        sys.exit(1)

    # arch define
    architecture = 'x86'
    if 'x2' in board or 'raspberry' in board:
        architecture = 'arm'
    elif 'xeon' in board:
        architecture = 'xeon'

    data_path = install_dir + "data/" + DATA_PATH_BASE
    bin_path = install_dir + "bin"
    src_benchmark = install_dir + "src/openmp/lavaMD"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    input_distance = "{}/input_distance_{}_{}"
    input_charges = "{}/input_charges_{}_{}"
    gold_output = "{}/output_gold_{}_{}"
    

    generate = ["mkdir -p " + bin_path,
                "cd " + src_benchmark,
                "make clean",
                "make -C ../../include ",
                "make -j 4 LOGS=1 general",
                "mkdir -p " + data_path,
                "mv -f ./" + benchmark_bin + " " + bin_path + "/"]

    execute = []

    for size in SIZES:
        #./lavamd_gen <# cores> <# boxes 1d>
        gen = [""] * 3
        gen[0] = ['{}/lavamd_gen'.format(src_benchmark)]
        gen[1] = [THREADS]
        gen[2] = [size]
        
        # change mode and iterations for exe
        # ./lavamd_check <# cores> <# boxes 1d> <input_distances> <input_charges> <gold_output> <#iterations>

        exe = copy.deepcopy(gen)
        exe[0] = ['sudo {}/lavamd_check'.format(bin_path)]
        current_input_distance = input_distance.format(data_path, THREADS, size)
        current_input_charge = input_charges.format(data_path, THREADS, size)
        current_gold = gold_output.format(data_path, THREADS, size)
        exe.append([current_input_distance])
        exe.append([current_input_charge])
        exe.append([current_gold])
        exe.append(ITERATIONS)


        generate.append(' '.join(str(r) for v in gen for r in v))
        generate.append("mv {} {} {} {}/".format(input_distance.format(".", THREADS, size), input_charges.format(".", THREADS, size), gold_output.format(".", THREADS, size), data_path))

        execute.append(' '.join(str(r) for v in exe for r in v))
        

    execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin, debug=debug)


if __name__ == "__main__":
    debug_mode = False
    try:
        parameter = str(sys.argv[0:][1]).upper()
        if parameter == 'DEBUG':
            debug_mode = True
    except:
        debug_mode = False

    board, _ = discover_board()
    config(board=board, debug=debug_mode)
    print("Json may have been generated.")
