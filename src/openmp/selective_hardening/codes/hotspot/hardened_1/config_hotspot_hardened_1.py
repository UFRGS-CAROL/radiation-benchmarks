#!/usr/bin/python

import ConfigParser
import sys

sys.path.insert(0, '../../../../../include')
from common_config import discover_board, execute_and_write_json_to_file

NUM_THREADS = 4
MATRIX_ORDER = 1024
ITER_SIZE = 512
TEMP_MATRIX = "../temp_" + str(MATRIX_ORDER)
POWER_MATRIX = "../power_" + str(MATRIX_ORDER)
GOLD_MATRIX = "../gold_" + str(MATRIX_ORDER)
ITERATIONS = 100000

def config(board, debug):

    benchmark_bin = "hotspot_check_hardened_1"
    print("Generating {} for OpenMP, board:{}".format(benchmark_bin, board))

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        sys.stderr.write("Configuration setup error: " + str(e))
        sys.exit(1)

    bin_path = install_dir + "bin"
    src_benchmark = install_dir + "src/openmp/selective_hardening/codes/hotspot/hardened_1"
    selective_hardening_dir = "/var/selective_hardening/"

    generate = ["sudo mkdir -p " + bin_path,
                "sudo mkdir -p " + selective_hardening_dir,
                "cd " + src_benchmark, 
                "make clean",
                "make",
                "./hotspot_gen {} {} {} {} {} {} {}".format(str(MATRIX_ORDER), str(MATRIX_ORDER), str(ITER_SIZE), str(NUM_THREADS), TEMP_MATRIX, POWER_MATRIX, GOLD_MATRIX),
                "sudo mv -f ./" + benchmark_bin + " " + bin_path + "/",
                "make clean"]

    execute = []

    exe = [None] * 9
    exe[0] = [bin_path + "/" + benchmark_bin]
    exe[1] = [str(MATRIX_ORDER)]
    exe[2] = [str(MATRIX_ORDER)]
    exe[3] = [str(ITER_SIZE)]
    exe[4] = [str(NUM_THREADS)]
    exe[5] = [TEMP_MATRIX]
    exe[6] = [POWER_MATRIX]
    exe[7] = [GOLD_MATRIX]
    exe[8] = [str(ITERATIONS)]

    execute = [(' '.join(str(r) for v in exe for r in v))]
    execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin, debug=debug)  


if __name__ == "__main__":
    debug_mode = False
    try:
        parameter = str(sys.argv[1:][1]).upper() 
        if parameter == 'DEBUG':
            debug_mode = True
    except:
        debug_mode = False
    
    board, _ = discover_board()
    config(board=board, debug=debug_mode)
    print("A json has been generated.")
