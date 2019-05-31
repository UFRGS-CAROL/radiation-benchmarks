#!/usr/bin/python

import ConfigParser
import copy
import os
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

SIZES = [512]
THREADS = 4
GRID_COLS = 1024
GRID_ROWS = 1024
ITERATIONS = [100000]


def config(board, debug):
    DATA_PATH_BASE = "hotspot"

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
    src_benchmark = install_dir + "src/openmp/hotspot"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    #./hotspot_gen <grid_rows> <grid_cols> <sim_time> <no. of threads><temp_file> <power_file> <output_file>
    temp_file = "{}/temp_1024"
    power_file = "{}/power_1024"
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
        #../hotspot_check <grid_rows> <grid_cols> <sim_time> <no. of threads><temp_file> <power_file> <output_file> <# iterations>
        gen = [""] * 8
        gen[0] = ['{}/hotspot_gen '.format(src_benchmark)]
        gen[1] = [GRID_ROWS]
        gen[2] = [GRID_COLS]
        gen[3] = [size]
        gen[4] = [THREADS]
        gen[5] = [temp_file.format(data_path, THREADS, size)]
        gen[6] = [power_file.format(data_path, THREADS, size)]
        gen[7] = [gold_output.format(data_path, THREADS, size)]
        
        exe = copy.deepcopy(gen)
        exe[0] = ['sudo {}/hotspot_check'.format(bin_path)]
        exe.append(ITERATIONS)


        generate.append(' '.join(str(r) for v in gen for r in v))
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
